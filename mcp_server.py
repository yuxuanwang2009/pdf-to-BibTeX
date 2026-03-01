"""
MCP Server for the BibTeX Extractor.

Exposes PDFEngine and LLMController functionality as MCP tools,
communicating over stdio (JSON-RPC). The GUI client launches this
as a subprocess and calls tools through the MCP protocol.

Run standalone for testing:
    python mcp_server.py            # stdio mode (for MCP clients)
    mcp dev mcp_server.py           # interactive inspector UI

Architecture:
    bib_app_mcp.py ──MCP client──> [stdio] ──> THIS SERVER ──> PDFEngine
                                                            ──> LLMController
"""

import sys
import json
import base64
import io
import logging

from mcp.server.fastmcp import FastMCP

from pdf_engine import PDFEngine
from llm_controller import LLMController
from llm_helper import LLMHelper

# ── Logging to stderr (stdout is reserved for MCP JSON-RPC messages) ──
logging.basicConfig(stream=sys.stderr, level=logging.DEBUG,
                    format="[MCP-Server] %(levelname)s: %(message)s")
log = logging.getLogger(__name__)

# ── Server-side state (persists across tool calls within one session) ──
pdf_engine = PDFEngine()
llm_controller: LLMController | None = None

# ── FastMCP server instance ──
mcp = FastMCP("BibExtractor")


# ═══════════════════════════════════════════════════════════════════════
#  LLM Connection Tools
# ═══════════════════════════════════════════════════════════════════════

@mcp.tool()
def validate_connection(api_key: str) -> str:
    """
    Validate an LLM API key and initialize the LLM controller.
    Returns JSON: {success, message, provider, model, available_models}
    """
    global llm_controller
    log.info("validate_connection called")

    try:
        ctrl = LLMController(api_key=api_key)
        success, msg = ctrl.llm.validate_connection()

        if success:
            llm_controller = ctrl
            provider = getattr(ctrl.llm, "provider", "gemini")
            model = getattr(ctrl.llm, "model_name", None) or "gemini-1.5-flash"
            available = LLMHelper.AVAILABLE_MODELS.get(provider, [model])
            return json.dumps({
                "success": True,
                "message": msg,
                "provider": provider,
                "model": model,
                "available_models": available,
            })
        else:
            return json.dumps({"success": False, "message": msg})

    except Exception as e:
        return json.dumps({"success": False, "message": str(e)})


@mcp.tool()
def set_model(model_name: str) -> str:
    """Switch the active LLM model (e.g. 'gpt-4o', 'gemini-1.5-pro')."""
    if llm_controller is None:
        return "Error: Not connected. Call validate_connection first."
    llm_controller.llm.set_model(model_name)
    log.info(f"Model switched to {model_name}")
    return f"Model switched to {model_name}"


# ═══════════════════════════════════════════════════════════════════════
#  PDF Tools
# ═══════════════════════════════════════════════════════════════════════

@mcp.tool()
def load_pdf(path: str) -> str:
    """
    Load a PDF file into memory.
    Returns JSON: {page_count}
    """
    log.info(f"load_pdf: {path}")
    pdf_engine.load_pdf(path)
    count = pdf_engine.get_page_count()
    return json.dumps({"page_count": count})


@mcp.tool()
def get_page_count() -> int:
    """Return the number of pages in the currently loaded PDF."""
    return pdf_engine.get_page_count()


@mcp.tool()
def get_page_pixmap(page_num: int, zoom: float = 1.0) -> str:
    """
    Render a PDF page as a PNG image.
    Returns a base64-encoded PNG string (decode with base64.b64decode).
    """
    pix = pdf_engine.get_page_pixmap(page_num, zoom)
    if pix is None:
        return ""

    # Convert PyMuPDF pixmap → PNG bytes → base64 string
    png_bytes = pix.tobytes("png")
    return base64.b64encode(png_bytes).decode("ascii")


@mcp.tool()
def get_text_in_rect(page_num: int, x0: float, y0: float,
                     x1: float, y1: float) -> str:
    """Extract text from a rectangular region on a PDF page."""
    import fitz
    rect = fitz.Rect(x0, y0, x1, y1)
    return pdf_engine.get_text_in_rect(page_num, rect)


@mcp.tool()
def get_full_text() -> str:
    """Return the full text of the loaded PDF (all pages, with page markers)."""
    return pdf_engine.get_context_text(page_count=None, force_full=True)


@mcp.tool()
def get_text_range(start_page: int, end_page: int) -> str:
    """Return text for a specific 1-based inclusive page range."""
    return pdf_engine.get_context_text_range(start_page, end_page)


# ═══════════════════════════════════════════════════════════════════════
#  LLM Analysis Tools
# ═══════════════════════════════════════════════════════════════════════

@mcp.tool()
def resolve_bibliography_range(full_text: str) -> str:
    """
    Use the LLM to identify the bibliography page range in the PDF.
    Returns JSON: {start_page, end_page, reason} or "null".
    """
    if llm_controller is None:
        return json.dumps({"error": "Not connected"})

    result = llm_controller.resolve_bibliography_range(full_text)
    if result is None:
        return "null"
    return json.dumps(result)


@mcp.tool()
def detect_citation_style(first_pages_text: str) -> str:
    """
    Analyze the first few pages to identify the citation style
    (e.g. "Numeric Brackets", "Author-Year", "Superscript").
    """
    if llm_controller is None:
        return "Error: Not connected"
    return llm_controller.detect_citation_style(first_pages_text)


@mcp.tool()
def resolve_citation(selection_text: str, context_text: str,
                     style_hint: str = "") -> str:
    """
    Resolve a user-selected citation snippet into BibTeX entries.
    - selection_text: the text the user highlighted (e.g. "[1-3]")
    - context_text: the bibliography section text
    - style_hint: detected citation style (optional)
    """
    if llm_controller is None:
        return "% Error: Not connected to LLM."
    return llm_controller.resolve_citation(
        selection_text, context_text,
        style_hint=style_hint if style_hint else None
    )


# ═══════════════════════════════════════════════════════════════════════
#  Entry point
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    log.info("Starting BibExtractor MCP Server (stdio)")
    mcp.run(transport="stdio")
