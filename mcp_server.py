"""
MCP Server for the BibTeX Extractor.

Exposes PDFEngine functionality as MCP tools, communicating over stdio
(JSON-RPC). The agent (agent.py) and GUI call these tools through
the MCP protocol via MCPClient.

This server provides only PDF operations — all LLM reasoning is handled
by the agent loop, which decides which tools to call via function calling.

Run standalone for testing:
    python mcp_server.py            # stdio mode (for MCP clients)
    mcp dev mcp_server.py           # interactive inspector UI

Architecture:
    bib_app_mcp.py ──┬── direct calls (rendering) ──> MCPClient ──> [stdio] ──> THIS SERVER
                     └── BibAgent (reasoning)     ──┘
"""

import sys
import json
import base64
import logging

from mcp.server.fastmcp import FastMCP

from pdf_engine import PDFEngine

# ── Logging to stderr (stdout is reserved for MCP JSON-RPC messages) ──
logging.basicConfig(stream=sys.stderr, level=logging.DEBUG,
                    format="[MCP-Server] %(levelname)s: %(message)s")
log = logging.getLogger(__name__)

# ── Server-side state (persists across tool calls within one session) ──
pdf_engine = PDFEngine()

# ── FastMCP server instance ──
mcp = FastMCP("BibExtractor")


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
#  Entry point
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    log.info("Starting BibExtractor MCP Server (stdio)")
    mcp.run(transport="stdio")
