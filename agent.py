"""
BibTeX Extraction Agent.

Orchestrates LLM-driven PDF analysis using function calling.
The LLM decides which PDF tools to call via MCP. This module handles:
  - Tool discovery from the MCP server
  - Schema conversion to OpenAI/Gemini format
  - The agent conversation loop
  - Status callbacks for the GUI

Architecture:
    GUI ──> BibAgent.run(intent) ──> LLM API (function calling)
                                 ──> MCPClient.call_tool() (executes LLM's choices)
"""

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Callable

from llm_helper import LLMHelper, LLMResponse, ToolCall
from mcp_client import MCPClient

log = logging.getLogger(__name__)


# ── Tool Schema Conversion ──────────────────────────────────────────

def mcp_schemas_to_openai(mcp_tools: list[dict]) -> list[dict]:
    """Convert MCP tool schemas to OpenAI function-calling format."""
    return [
        {
            "type": "function",
            "function": {
                "name": tool["name"],
                "description": tool["description"],
                "parameters": tool["inputSchema"],
            },
        }
        for tool in mcp_tools
    ]


def mcp_schemas_to_gemini(mcp_tools: list[dict]) -> list[dict]:
    """
    Convert MCP tool schemas to Gemini function_declarations format.
    Gemini requires uppercase types and rejects extra fields like 'title'.
    """
    TYPE_MAP = {
        "object": "OBJECT", "string": "STRING", "integer": "INTEGER",
        "number": "NUMBER", "boolean": "BOOLEAN", "array": "ARRAY",
    }

    def convert_schema(schema: dict) -> dict:
        result = {}
        if "type" in schema:
            result["type_"] = TYPE_MAP.get(schema["type"], schema["type"].upper())
        if "description" in schema:
            result["description"] = schema["description"]
        if "properties" in schema:
            result["properties"] = {
                k: convert_schema(v) for k, v in schema["properties"].items()
            }
        if "required" in schema:
            result["required"] = schema["required"]
        if "items" in schema:
            result["items"] = convert_schema(schema["items"])
        if "enum" in schema:
            result["enum"] = schema["enum"]
        # Intentionally drop "title", "default", etc. that Gemini rejects
        return result

    return [
        {
            "name": tool["name"],
            "description": tool["description"],
            "parameters": convert_schema(tool["inputSchema"]),
        }
        for tool in mcp_tools
    ]


# ── System Prompt ────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are BibTeX Resolver, an expert AI agent for extracting bibliography \
information from academic PDFs.

You have access to PDF tools that let you read text from loaded PDF documents. \
A PDF has already been loaded for you.

## Your Capabilities
You can: read full PDF text, read specific page ranges, extract text from \
rectangular regions on specific pages, and get the page count.

## Task Types You Handle

### 1. ANALYZE_PDF (called once when a PDF is loaded)
Goal: Find the bibliography section and detect the citation style.

Steps:
1. Call `get_full_text` to read the entire PDF.
2. Analyze the text to find the bibliography/references section:
   - Look for headings: "References", "Bibliography", "Literature Cited"
   - IGNORE the Table of Contents. A ToC line like "References ...... 42" \
is NOT the bibliography.
   - VERIFY: The pages must contain actual citation entries \
(e.g., "[1] Author, Title, Journal..." or "Author (Year). Title...")
   - Determine start_page and end_page (1-based).
3. Call `get_text_range` with those page numbers to get the bibliography text.
4. Detect the citation style from the first ~5 pages of the document. \
Call `get_text_range(start_page=1, end_page=5)`.
   Look at the MAIN BODY text (skip abstract, ToC, title). Identify one of:
   - "Numeric Brackets": [1], [2-5]
   - "Author-Year": (Smith 2020), (Jones et al., 2021)
   - "Superscript": Word^1 or Word1
   - "Alpha-Numeric": [Smi20]
   - "Footnotes"
   - "Unknown/Generic"
5. Return a JSON summary (no markdown fences, just raw JSON):
   {"bibliography_start": N, "bibliography_end": M, \
"bibliography_text": "...", "citation_style": "...", "reason": "..."}

### 2. RESOLVE_CITATION (called when user selects text from the PDF)
You will receive the selected text and the bibliography context. Your job:

1. ANALYZE the selection to identify citation handles:
   - Valid: "[1]", "[1-3]", "[1, 5]", "(Smith 2020)", "(Doe, 2021)", \
"(Jones et al. 2022)", "Ref. 12", "(OpenAI 2023)"
   - INVALID (ignore these): "Fig. 2", "2D", "equation (5)", \
"Section 3", plain numbers
   - Expand ranges: "[1-3]" means references 1, 2, and 3
   - Range separators include: -, \u2013, \u2014, \u2011, \u2212, \
\uFE63, \uFF0D (treat all as equivalent)
   - If selection contains bibliography entries directly, parse all lines.
2. If NO valid citation handles found, respond: \
"% No valid citation handles found in selection."
   Do NOT hallucinate. Do NOT guess based on topic.
3. LOCATE each handle in the bibliography context text.
4. CONVERT each found reference to a valid BibTeX entry with correct: \
author, title, journal, volume, pages, year, DOI.
   Do not hallucinate missing metadata.
5. Return ONLY BibTeX entries separated by newlines. No markdown. \
No conversation. Error messages must be commented with "%".

## Important Rules
- Never hallucinate references. If you cannot find a reference in the \
bibliography, say so with a % comment.
- When a citation style hint is provided, strictly enforce that style \
when identifying handles.
- Be thorough: always expand numeric ranges and process all citations \
in a selection.
"""

# Tools the agent is allowed to use (excludes load_pdf, get_page_pixmap
# which are GUI-only operations)
AGENT_TOOL_NAMES = {
    "get_full_text",
    "get_text_range",
    "get_text_in_rect",
    "get_page_count",
}


# ── Agent ────────────────────────────────────────────────────────────

@dataclass
class AgentResult:
    """Result of an agent run."""
    text: str
    tool_calls_made: int = 0
    error: str | None = None


class BibAgent:
    """
    Agent that uses LLM function calling to orchestrate PDF analysis.

    One instance per PDF session. Maintains conversation history
    so the LLM has context from previous operations (e.g., the bibliography
    location is remembered for subsequent citation resolution calls).
    """

    MAX_ITERATIONS = 15

    def __init__(
        self,
        llm: LLMHelper,
        mcp_client: MCPClient,
        on_status: Callable[[str], None] | None = None,
    ):
        self.llm = llm
        self.mcp_client = mcp_client
        self.on_status = on_status or (lambda s: None)

        # Conversation history persists across run() calls
        self.messages: list[dict] = [
            {"role": "system", "content": SYSTEM_PROMPT}
        ]

        # Discover and convert tool schemas from MCP server
        self._mcp_tools: list[dict] = []
        self._provider_tools: list[dict] = []
        self._discover_tools()

    def _discover_tools(self):
        """Fetch tool schemas from MCP server and convert to provider format."""
        all_tools = self.mcp_client.list_tools_with_schemas()

        # Filter to agent-relevant tools only
        self._mcp_tools = [
            t for t in all_tools if t["name"] in AGENT_TOOL_NAMES
        ]

        if self.llm.provider == "openai":
            self._provider_tools = mcp_schemas_to_openai(self._mcp_tools)
        elif self.llm.provider == "gemini":
            self._provider_tools = mcp_schemas_to_gemini(self._mcp_tools)

        log.info(
            f"Agent discovered {len(self._mcp_tools)} tools: "
            f"{[t['name'] for t in self._mcp_tools]}"
        )

    def run(self, user_message: str) -> AgentResult:
        """
        Run the agent loop for a user intent.

        The LLM may make multiple tool calls before returning a final
        text answer. Blocks until the LLM produces a text response.

        Args:
            user_message: e.g. "Analyze this PDF" or
                          "Resolve citation: '[1-3]' with context: ..."
        """
        self.messages.append({"role": "user", "content": user_message})

        total_tool_calls = 0

        for iteration in range(self.MAX_ITERATIONS):
            self.on_status(f"Agent thinking... (step {iteration + 1})")

            try:
                response: LLMResponse = self.llm.chat_with_tools(
                    messages=self.messages,
                    tools=self._provider_tools,
                    temperature=0,
                )
            except Exception as e:
                error_msg = f"LLM API error: {e}"
                log.error(error_msg)
                return AgentResult(text="", error=error_msg)

            if not response.has_tool_calls:
                # Final text answer
                text = response.text or ""
                self.messages.append({"role": "assistant", "content": text})
                self.on_status("Agent complete.")
                return AgentResult(text=text, tool_calls_made=total_tool_calls)

            # LLM wants to call tools — add assistant message to history
            self._append_assistant_tool_calls(response)

            for tc in response.tool_calls:
                total_tool_calls += 1
                self.on_status(f"Calling tool: {tc.name}...")
                log.info(f"Tool call: {tc.name}({tc.arguments})")

                try:
                    result = self.mcp_client.call_tool(tc.name, tc.arguments)
                except Exception as e:
                    result = f"Error: {e}"
                    log.error(f"Tool {tc.name} failed: {e}")

                self._append_tool_result(tc, result)

        # Safety: hit max iterations
        return AgentResult(
            text="% Agent reached maximum iterations without a final answer.",
            tool_calls_made=total_tool_calls,
            error="Max iterations reached",
        )

    def reset(self):
        """Clear conversation history (keep system prompt)."""
        self.messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    # ── History helpers ──────────────────────────────────────────────

    def _append_assistant_tool_calls(self, response: LLMResponse):
        """Add assistant message with tool calls to conversation history."""
        if self.llm.provider == "openai":
            msg = response.raw_message
            self.messages.append({
                "role": "assistant",
                "content": msg.content,
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in msg.tool_calls
                ],
            })
        elif self.llm.provider == "gemini":
            self.messages.append({
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.name,
                            "arguments": json.dumps(tc.arguments),
                        },
                    }
                    for tc in response.tool_calls
                ],
            })

    def _append_tool_result(self, tc: ToolCall, result: str):
        """Add tool result to conversation history."""
        self.messages.append({
            "role": "tool",
            "tool_call_id": tc.id,
            "name": tc.name,
            "content": result,
        })
