"""
Synchronous MCP Client wrapper for the BibTeX Extractor GUI.

This module bridges the gap between tkinter's synchronous event loop
and the MCP SDK's async protocol. It:

  1. Launches the MCP server (`mcp_server.py`) as a subprocess over stdio.
  2. Runs a persistent asyncio event loop in a background thread.
  3. Provides synchronous `call_tool(name, args)` that the GUI can call
     from any thread — it submits async work to the background loop
     and blocks until the result is ready.

Usage from the GUI:
    client = MCPClient()
    client.connect()                              # launches server subprocess
    result = client.call_tool("load_pdf", {"path": "/tmp/paper.pdf"})
    print(result)                                 # '{"page_count": 42}'
    client.disconnect()                           # tears down everything
"""

import asyncio
import json
import os
import sys
import threading
from typing import Any

from mcp import ClientSession, StdioServerParameters, types
from mcp.client.stdio import stdio_client


class MCPClient:
    """
    Synchronous wrapper around an MCP ClientSession.

    Internally manages:
      - A background thread running an asyncio event loop
      - The stdio transport to the MCP server subprocess
      - The MCP ClientSession (initialized once, reused for all calls)
    """

    def __init__(self):
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._session: ClientSession | None = None
        # AsyncExitStack manages the nested async context managers
        # (stdio_client and ClientSession) so they stay open across calls.
        self._exit_stack: asyncio.tasks.Task | None = None
        self._connected = False
        self._connect_event = threading.Event()
        self._connect_error: str | None = None

    # ── Public API (all synchronous, safe to call from any thread) ────

    def connect(self, server_script: str | None = None):
        """
        Launch the MCP server subprocess and establish a session.
        Blocks until the connection is ready or fails.

        Args:
            server_script: Path to mcp_server.py. If None, auto-detects
                           from the same directory as this file.
        """
        if self._connected:
            return

        if server_script is None:
            server_script = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "mcp_server.py"
            )

        self._connect_event.clear()
        self._connect_error = None

        # Start background event loop thread
        self._thread = threading.Thread(
            target=self._run_loop, args=(server_script,), daemon=True
        )
        self._thread.start()

        # Wait for connection to complete (or fail)
        self._connect_event.wait(timeout=30)

        if self._connect_error:
            raise ConnectionError(
                f"MCP server connection failed: {self._connect_error}"
            )
        if not self._connected:
            raise ConnectionError("MCP server connection timed out.")

    def disconnect(self):
        """Shut down the MCP session, server subprocess, and background loop."""
        if self._loop and self._connected:
            # Signal the background loop to stop
            self._loop.call_soon_threadsafe(self._loop.stop)
        if self._thread:
            self._thread.join(timeout=5)
        self._connected = False
        self._session = None
        self._loop = None
        self._thread = None

    def call_tool(self, name: str, arguments: dict[str, Any] | None = None,
                  timeout: float = 120) -> str:
        """
        Call an MCP tool synchronously. Blocks until the result is returned.

        Args:
            name: Tool name (e.g. "load_pdf", "resolve_citation")
            arguments: Tool arguments dict
            timeout: Max seconds to wait for the result

        Returns:
            The text content of the tool result (string).

        Raises:
            RuntimeError: If not connected or call fails.
        """
        if not self._connected or not self._loop or not self._session:
            raise RuntimeError("Not connected. Call connect() first.")

        future = asyncio.run_coroutine_threadsafe(
            self._async_call_tool(name, arguments or {}),
            self._loop,
        )
        return future.result(timeout=timeout)

    def list_tools(self) -> list[str]:
        """List the names of all tools available on the server."""
        if not self._connected or not self._loop or not self._session:
            raise RuntimeError("Not connected.")

        future = asyncio.run_coroutine_threadsafe(
            self._async_list_tools(), self._loop
        )
        return future.result(timeout=10)

    @property
    def is_connected(self) -> bool:
        return self._connected

    # ── Internal async methods (run on the background event loop) ─────

    async def _async_call_tool(self, name: str, arguments: dict) -> str:
        """Async implementation of call_tool."""
        result = await self._session.call_tool(name, arguments=arguments)

        # Extract text from the result content blocks
        texts = []
        for block in result.content:
            if isinstance(block, types.TextContent):
                texts.append(block.text)
            elif isinstance(block, types.ImageContent):
                # For image content, return the base64 data
                texts.append(block.data)
        return "\n".join(texts)

    async def _async_list_tools(self) -> list[str]:
        """Async implementation of list_tools."""
        result = await self._session.list_tools()
        return [tool.name for tool in result.tools]

    # ── Background thread entry point ─────────────────────────────────

    def _run_loop(self, server_script: str):
        """
        Runs in a background daemon thread.
        Creates an asyncio event loop, connects to the MCP server,
        and keeps the loop running until disconnect() is called.
        """
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)

        try:
            self._loop.run_until_complete(self._connect_async(server_script))
            # Keep the loop running so it can process call_tool futures
            self._loop.run_forever()
        except Exception as e:
            self._connect_error = str(e)
            self._connect_event.set()
        finally:
            # Cleanup: close the exit stack (shuts down server subprocess)
            if self._cleanup_coro:
                try:
                    self._loop.run_until_complete(self._cleanup_coro())
                except Exception:
                    pass
            self._loop.close()
            self._connected = False

    async def _connect_async(self, server_script: str):
        """
        Async connection sequence:
        1. Launch server subprocess via stdio_client
        2. Create ClientSession
        3. Initialize the MCP session
        """
        from contextlib import AsyncExitStack

        self._stack = AsyncExitStack()
        await self._stack.__aenter__()

        # Determine the Python executable (same one running this process)
        python_exe = sys.executable

        server_params = StdioServerParameters(
            command=python_exe,
            args=[server_script],
            env=None,  # inherits current environment (API keys, etc.)
        )

        # Enter the stdio_client context (launches subprocess)
        transport = await self._stack.enter_async_context(
            stdio_client(server_params)
        )
        read_stream, write_stream = transport

        # Enter the ClientSession context
        self._session = await self._stack.enter_async_context(
            ClientSession(read_stream, write_stream)
        )

        # Initialize the MCP protocol handshake
        await self._session.initialize()

        # Store cleanup coroutine for later
        self._cleanup_coro = self._stack.aclose

        self._connected = True
        self._connect_event.set()

        # Log available tools
        tools = await self._session.list_tools()
        tool_names = [t.name for t in tools.tools]
        print(f"[MCP-Client] Connected. Available tools: {tool_names}",
              file=sys.stderr)
