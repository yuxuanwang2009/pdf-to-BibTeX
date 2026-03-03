# AI BibTeX Extractor

An LLM-powered tool for extracting BibTeX citations from academic PDFs. Uses an agent architecture where the LLM reasons about your document and decides how to extract references — no regex, no hardcoded parsing.

![Demo Application Interface](demo.png)
*The interface showing a PDF with selected text (left) and the extracted BibTeX result (right).*

## How It Works

The app has two layers:

- **MCP Server** — a simple PDF tool server (load, read text, render pages). No intelligence. Exposed via the [Model Context Protocol](https://modelcontextprotocol.io/) so any AI client can use it.
- **Agent** — an LLM (OpenAI or Gemini) that receives your request, discovers the available PDF tools, and decides which to call. It finds the bibliography, detects the citation style, and resolves your selections to BibTeX.

When you open a PDF, the agent reads the full text, locates the bibliography section, and identifies the citation style — all on its own. When you select text, the agent resolves the citation handles to BibTeX entries using the bibliography context it already has.

## Features

- **Agent-driven**: The LLM orchestrates the entire workflow via function calling. No hardcoded sequences.
- **Any citation style**: Numeric brackets `[1]`, author-year `(Smith 2020)`, superscript, footnotes — the LLM figures it out.
- **Smart selection**: Ranges `[1-3]`, disjoint `[1] and [5]`, or entire bibliography blocks.
- **Two ways to use it**:
  - **GUI** — visual PDF viewer with red-box selection
  - **Claude Desktop** — register the MCP server and use it conversationally
- **Model switching**: Switch between OpenAI (`gpt-4o`, `gpt-5.2`) and Gemini (`1.5-flash`, `1.5-pro`) on the fly.

## Requirements

- Python 3.10+
- An API key from Google (Gemini) or OpenAI

## Installation

```bash
pip install pymupdf Pillow google-generativeai openai mcp
```

## Usage

### Option 1: GUI App

```bash
python bib_app_mcp.py
```

1. Enter your API key and click **Connect**.
2. Click **Open PDF**. The agent analyzes the document automatically.
3. Draw a red box around any citation. BibTeX appears in the right panel.

### Option 2: Claude Desktop

Add this to `~/Library/Application Support/Claude/claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "bib-extractor": {
      "command": "python",
      "args": ["/path/to/mcp_server.py"]
    }
  }
}
```

Restart Claude Desktop. Then in Chat:

> "Use load_pdf to open /path/to/paper.pdf, find the bibliography, and give me BibTeX for citations [1-3]. Return only raw BibTeX."

### Option 3: Standalone (no MCP)

The original direct-call GUI is still available:

```bash
python bib_app.py
```

## Architecture

```
bib_app_mcp.py (GUI)
 ├── direct MCP calls (page rendering, navigation)
 └── BibAgent (agent.py)
      ├── LLM API with function calling (the brain)
      └── MCP tool execution (PDF operations)
           │
      mcp_server.py ──> pdf_engine.py ──> PyMuPDF
```

| File | Role |
|------|------|
| `mcp_server.py` | PDF tool server (MCP protocol, stdio transport) |
| `mcp_client.py` | Sync wrapper for the async MCP client SDK |
| `agent.py` | LLM agent loop with function calling + system prompt |
| `llm_helper.py` | Provider abstraction (OpenAI / Gemini) with tool calling support |
| `pdf_engine.py` | PyMuPDF wrapper (load, render, extract text) |
| `bib_app_mcp.py` | Tkinter GUI (MCP + Agent edition) |
| `bib_app.py` | Tkinter GUI (standalone, no MCP) |

## Troubleshooting

- **"Agent not ready"**: Make sure you've connected your API key before opening a PDF.
- **Claude Desktop doesn't use tools**: Fully quit and reopen Claude Desktop after editing the config. Check for the tools icon in Chat.
- **Unresolved reference**: The bibliography text must be searchable (not a scanned image).
