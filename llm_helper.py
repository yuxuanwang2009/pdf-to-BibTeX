import json
import os
import re
from dataclasses import dataclass, field
from typing import Any

# Try importing openai
try:
    from openai import OpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False

# Try importing google-generativeai
try:
    import google.generativeai as genai
    HAS_GENAI = True
except ImportError:
    HAS_GENAI = False


# ── Data types for function calling ─────────────────────────────────

@dataclass
class ToolCall:
    """Represents one function call the LLM wants to make."""
    id: str
    name: str
    arguments: dict


@dataclass
class LLMResponse:
    """Unified response from chat_with_tools."""
    text: str | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)
    raw_message: Any = None

    @property
    def has_tool_calls(self) -> bool:
        return len(self.tool_calls) > 0


class LLMHelper:
    AVAILABLE_MODELS = {
        "openai": ["gpt-4o", "gpt-5.2", "gpt-4-turbo"],
        "gemini": ["gemini-1.5-flash", "gemini-1.5-pro", "gemini-1.0-pro"]
    }

    def __init__(self, api_key=None, provider="auto", model_name=None):
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY") or os.getenv("OPENAI_API_KEY")
        self.is_configured = bool(self.api_key)
        self.provider = provider
        self.client = None
        self.model_name = model_name

        if self.is_configured:
            # Auto-detect provider if default
            if self.provider == "auto":
                if self.api_key.startswith("sk-"):
                    self.provider = "openai"
                else:
                    self.provider = "gemini"

            if self.provider == "gemini" and HAS_GENAI:
                genai.configure(api_key=self.api_key)
                if not self.model_name:
                    self.model_name = "gemini-1.5-flash"
                self.model = genai.GenerativeModel(self.model_name)
            elif self.provider == "openai" and HAS_OPENAI:
                if not self.model_name:
                    self.model_name = "gpt-4o"
                self.client = OpenAI(api_key=self.api_key)

    def set_model(self, model_name):
        """Updates the active model."""
        self.model_name = model_name
        if self.provider == "gemini" and HAS_GENAI:
             self.model = genai.GenerativeModel(self.model_name)

    def validate_connection(self):
        """
        Tests if the current API Key is valid by making a minimal API call.
        Returns: (True, "Message") or (False, "Error Message")
        """
        if not self.is_configured:
            return False, "No API Key provided."

        try:
            if self.provider == "gemini" and HAS_GENAI:
                try:
                    next(genai.list_models())
                    return True, "Success: Gemini API Connected."
                except Exception as inner_e:
                    raise inner_e

            elif self.provider == "openai" and self.client:
                try:
                    self.client.models.list()
                    return True, "Success: OpenAI API Connected."
                except Exception as inner_e:
                    raise inner_e
            elif not HAS_OPENAI and self.provider == "openai":
                 return False, "Error: 'openai' library not installed."
            elif not HAS_GENAI and self.provider == "gemini":
                 return False, "Error: 'google.generativeai' library not installed."

            return False, f"Unknown provider or missing library for {self.provider}"

        except Exception as e:
            return False, f"Connection Failed: {str(e)}"

    def custom_query(self, prompt, json_mode=False, temperature=0):
        """Executes a raw prompt against the configured LLM."""
        if not self.is_configured:
            return None
        return self._query_llm(prompt, json_mode=json_mode, temperature=temperature)

    # ═════════════════════════════════════════════════════════════════
    #  Function calling (agent loop support)
    # ═════════════════════════════════════════════════════════════════

    def chat_with_tools(
        self,
        messages: list[dict],
        tools: list[dict],
        temperature: float = 0,
    ) -> LLMResponse:
        """
        Send a conversation with tool definitions.

        Args:
            messages: Conversation history in OpenAI format
                      (system/user/assistant/tool roles).
            tools: Tool definitions in provider-native format.
            temperature: LLM temperature.

        Returns:
            LLMResponse with either text or tool_calls populated.
        """
        if self.provider == "openai":
            return self._openai_chat_with_tools(messages, tools, temperature)
        elif self.provider == "gemini":
            return self._gemini_chat_with_tools(messages, tools, temperature)
        else:
            raise ValueError(f"Unknown provider: {self.provider}")

    def _openai_chat_with_tools(self, messages, tools, temperature) -> LLMResponse:
        kwargs = {
            "model": self.model_name,
            "messages": messages,
            "temperature": temperature,
        }
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"

        completion = self.client.chat.completions.create(**kwargs)
        choice = completion.choices[0]
        msg = choice.message

        if msg.tool_calls:
            tool_calls = []
            for tc in msg.tool_calls:
                tool_calls.append(ToolCall(
                    id=tc.id,
                    name=tc.function.name,
                    arguments=json.loads(tc.function.arguments),
                ))
            return LLMResponse(text=msg.content, tool_calls=tool_calls, raw_message=msg)
        else:
            return LLMResponse(text=msg.content, raw_message=msg)

    def _gemini_chat_with_tools(self, messages, tools, temperature) -> LLMResponse:
        # Extract system instruction and convert messages
        system_text, gemini_contents = self._convert_messages_to_gemini(messages)

        # Build Gemini tools
        gemini_tools = None
        if tools:
            gemini_tools = [genai.protos.Tool(function_declarations=[
                genai.protos.FunctionDeclaration(
                    name=t["name"],
                    description=t["description"],
                    parameters=t.get("parameters"),
                )
                for t in tools
            ])]

        model = genai.GenerativeModel(
            self.model_name,
            tools=gemini_tools,
            system_instruction=system_text if system_text else None,
            generation_config={"temperature": temperature},
        )

        response = model.generate_content(gemini_contents)

        # Parse response parts
        tool_calls = []
        text_parts = []
        for part in response.candidates[0].content.parts:
            if hasattr(part, "function_call") and part.function_call.name:
                fc = part.function_call
                tool_calls.append(ToolCall(
                    id=f"gemini_{fc.name}_{id(fc)}",
                    name=fc.name,
                    arguments=dict(fc.args) if fc.args else {},
                ))
            elif hasattr(part, "text") and part.text:
                text_parts.append(part.text)

        if tool_calls:
            return LLMResponse(
                text="\n".join(text_parts) if text_parts else None,
                tool_calls=tool_calls,
                raw_message=response,
            )
        else:
            return LLMResponse(
                text=self._clean_llm_output("\n".join(text_parts)),
                raw_message=response,
            )

    def _convert_messages_to_gemini(self, messages: list[dict]) -> tuple[str, list]:
        """
        Convert OpenAI-format messages to Gemini contents.
        Returns (system_text, contents_list).
        """
        system_text = ""
        contents = []

        for msg in messages:
            role = msg["role"]

            if role == "system":
                system_text = msg["content"]

            elif role == "user":
                contents.append({"role": "user", "parts": [msg["content"]]})

            elif role == "assistant":
                if msg.get("tool_calls"):
                    parts = []
                    for tc in msg["tool_calls"]:
                        args = tc["function"]["arguments"]
                        if isinstance(args, str):
                            args = json.loads(args)
                        parts.append(genai.protos.Part(
                            function_call=genai.protos.FunctionCall(
                                name=tc["function"]["name"],
                                args=args,
                            )
                        ))
                    contents.append({"role": "model", "parts": parts})
                else:
                    contents.append({
                        "role": "model",
                        "parts": [msg["content"] or ""]
                    })

            elif role == "tool":
                contents.append({"role": "user", "parts": [
                    genai.protos.Part(
                        function_response=genai.protos.FunctionResponse(
                            name=msg.get("name", "tool"),
                            response={"result": msg["content"]},
                        )
                    )
                ]})

        return system_text, contents

    # ═════════════════════════════════════════════════════════════════
    #  Original query method (kept for backward compat)
    # ═════════════════════════════════════════════════════════════════

    def _query_llm(self, prompt, json_mode=False, temperature=0):
        """Helper to handle provider differences"""
        try:
            if self.provider == "gemini" and HAS_GENAI:
                response = self.model.generate_content(
                    prompt,
                    generation_config={"temperature": temperature},
                )
                return self._clean_llm_output(response.text)
            elif self.provider == "openai" and self.client:
                kwargs = {
                    "model": self.model_name,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": temperature,
                }
                if json_mode:
                    kwargs["response_format"] = { "type": "json_object" }

                completion = self.client.chat.completions.create(**kwargs)
                return self._clean_llm_output(completion.choices[0].message.content)
        except Exception as e:
            print(f"LLM Query Error: {e}")
            raise e

    def _clean_llm_output(self, text):
        # Remove markdown code blocks
        text = re.sub(r'```(?:json|bibtex)?', '', text)
        text = re.sub(r'```', '', text)
        return text.strip()
