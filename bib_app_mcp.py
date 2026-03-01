"""
PDF Bib Extractor — MCP Edition.

Functionally identical to bib_app.py, but all PDF and LLM operations
go through the MCP protocol instead of direct in-process calls.

Architecture:
    This GUI ──MCPClient──> [stdio pipe] ──> mcp_server.py ──> PDFEngine
                                                            ──> LLMController

The MCPClient (mcp_client.py) manages the server subprocess and provides
synchronous call_tool() wrappers for the async MCP protocol.
"""

import tkinter as tk
from tkinter import filedialog, ttk, scrolledtext, messagebox
import fitz
from PIL import Image, ImageTk
import threading
import os
import json
import base64
import io

from mcp_client import MCPClient


class BibApp:
    def __init__(self, root):
        self.root = root
        self.root.title("PDF Bib Extractor (MCP Edition)")
        self.root.geometry("1400x900")

        # --- Visual Style & Theme ---
        self.colors = {
            "bg_root": "#2E2E2E",
            "bg_panel": "#383838",
            "fg_text": "#E0E0E0",
            "accent": "#61AFEF",
            "btn_bg": "#444444",
            "btn_fg": "#FFFFFF",
            "entry_bg": "#252525",
            "canvas_bg": "#404040",
            "success": "#98C379",
            "error": "#E06C75"
        }

        self.root.configure(bg=self.colors["bg_root"])

        self.style = ttk.Style()
        self.style.theme_use('clam')

        self.style.configure("TFrame", background=self.colors["bg_panel"])
        self.style.configure("Root.TFrame", background=self.colors["bg_root"])
        self.style.configure("TLabel", background=self.colors["bg_panel"], foreground=self.colors["fg_text"], font=("Helvetica", 11))
        self.style.configure("Header.TLabel", font=("Helvetica", 12, "bold"), foreground=self.colors["accent"])
        self.style.configure("Status.TLabel", font=("Helvetica", 10, "italic"), foreground="gray")
        self.style.configure("TButton",
                             font=("Helvetica", 11),
                             background=self.colors["btn_bg"],
                             foreground=self.colors["btn_fg"],
                             borderwidth=1,
                             focusthickness=3,
                             focuscolor="none")
        self.style.map("TButton", background=[("active", "#555555"), ("pressed", "#222222")])
        self.style.configure("TLabelframe", background=self.colors["bg_panel"], foreground=self.colors["fg_text"], borderwidth=1)
        self.style.configure("TLabelframe.Label", background=self.colors["bg_panel"], foreground=self.colors["accent"], font=("Helvetica", 10, "bold"))
        self.style.configure("TEntry", fieldbackground=self.colors["entry_bg"], foreground=self.colors["fg_text"], insertcolor="white", borderwidth=0)

        # ── MCP Client (replaces direct PDFEngine + LLMController) ──
        self.mcp_client = MCPClient()
        self._mcp_connected = False  # True after server subprocess is up

        # We still need a local PDFEngine for rendering pages in the canvas,
        # because shipping raw pixmap bytes over MCP for every scroll/resize
        # would be too slow. The server handles LLM + text extraction.
        # However, to stay true to the MCP demo, we route page rendering
        # through MCP too — the base64 overhead is acceptable for a demo.

        self.current_context = ""
        self.current_page = 0
        self.image_ref = None
        self.citation_rects = []
        self.citation_style_hint = None
        self._pdf_page_count = 0

        # State
        self.config_file = os.path.join(os.path.expanduser("~"), ".bib_extractor_config.json")

        initial_key = os.getenv("GOOGLE_API_KEY", "")
        if not initial_key:
            initial_key = self.load_config().get("api_key", "")

        self.api_key_var = tk.StringVar(value=initial_key)
        self.selection_start = None
        self.zoom_level = 1.5

        self._setup_ui()
        self._start_mcp_server()

    def _start_mcp_server(self):
        """Launch the MCP server subprocess in the background."""
        def start():
            try:
                self.mcp_client.connect()
                self._mcp_connected = True
                tools = self.mcp_client.list_tools()
                self.update_status(f"MCP Server ready. Tools: {len(tools)}")
            except Exception as e:
                self.update_status(f"MCP Server failed: {e}")

        threading.Thread(target=start, daemon=True).start()

    def load_config(self):
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, "r") as f:
                    return json.load(f)
            except Exception:
                pass
        return {}

    def save_config(self, key):
        try:
            with open(self.config_file, "w") as f:
                json.dump({"api_key": key}, f)
        except Exception as e:
            print(f"Failed to save config: {e}")

    def _setup_ui(self):
        # --- Left: PDF Viewer (Canvas) ---
        self.viewer_frame = tk.Frame(self.root, bg=self.colors["bg_root"])
        self.viewer_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.canvas = tk.Canvas(self.viewer_frame, bg=self.colors["canvas_bg"], highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)

        self.canvas.bind("<ButtonPress-1>", self.on_canvas_click)
        self.canvas.bind("<B1-Motion>", self.on_canvas_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_canvas_release)
        self.canvas.bind("<Configure>", self.on_resize)

        self.root.bind("<Left>", lambda e: self.prev_page())
        self.root.bind("<Right>", lambda e: self.next_page())

        # --- Right: Controls ---
        self.control_frame = tk.Frame(self.root, width=400, bg=self.colors["bg_panel"])
        self.control_frame.pack(side=tk.RIGHT, fill=tk.Y)
        self.control_frame.pack_propagate(False)

        content_box = ttk.Frame(self.control_frame, style="TFrame")
        content_box.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

        # 1. Navigation & Open
        nav_frame = ttk.Frame(content_box)
        nav_frame.pack(fill=tk.X, pady=(0, 20))

        self.btn_open = ttk.Button(nav_frame, text="Open PDF", command=self.open_pdf, state="disabled", width=12)
        self.btn_open.pack(side=tk.LEFT, padx=(0, 10))

        ttk.Button(nav_frame, text="<", width=3, command=self.prev_page).pack(side=tk.LEFT)
        self.lbl_page = ttk.Label(nav_frame, text="Page: 0/0", width=12, anchor="center")
        self.lbl_page.pack(side=tk.LEFT, padx=5)
        ttk.Button(nav_frame, text=">", width=3, command=self.next_page).pack(side=tk.LEFT)

        # 2. Setup / API Key
        setup_frame = ttk.LabelFrame(content_box, text="Connection Setup", padding=15)
        setup_frame.pack(fill=tk.X, pady=(0, 20))

        self.key_container = ttk.Frame(setup_frame)
        self.key_container.pack(fill=tk.X)
        self._build_key_input_state()

        self.key_status_label = ttk.Label(setup_frame, text="", font=("Helvetica", 9))
        self.key_status_label.pack(anchor="w", pady=(5, 0))

        ttk.Label(setup_frame, text="Valid API Key required.", style="Status.TLabel").pack(anchor="w", pady=(10, 0))

        # 3. Output Area
        ttk.Label(content_box, text="Extracted BibTeX", style="Header.TLabel").pack(anchor="w", pady=(0, 5))

        self.output_text = scrolledtext.ScrolledText(content_box, height=20, bg=self.colors["entry_bg"], fg=self.colors["fg_text"], insertbackground="white", borderwidth=0, font=("Consolas", 10))
        self.output_text.pack(fill=tk.BOTH, expand=True, pady=(0, 20))

        self.context_menu = tk.Menu(self.root, tearoff=0, bg=self.colors["bg_panel"], fg=self.colors["fg_text"])
        self.context_menu.add_command(label="Copy Selection", command=self.copy_selection)
        self.output_text.bind("<Button-3>", self.show_context_menu)
        self.output_text.bind("<Button-2>", self.show_context_menu)
        self.output_text.bind("<Control-Button-1>", self.show_context_menu)

        # 4. Action Buttons
        action_frame = ttk.Frame(content_box)
        action_frame.pack(fill=tk.X)

        ttk.Button(action_frame, text="Copy All", command=self.copy_to_clipboard, width=15).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(action_frame, text="Clear", command=self.clear_output, width=10).pack(side=tk.LEFT)

        # 5. Global Status Bar
        self.status_var = tk.StringVar(value="Starting MCP Server...")
        self.status_bar = tk.Label(self.control_frame, textvariable=self.status_var, bg=self.colors["bg_root"], fg="gray", anchor="w", padx=10, pady=5, font=("Helvetica", 9))
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)

    def _build_key_input_state(self):
        for widget in self.key_container.winfo_children():
            widget.destroy()

        ttk.Label(self.key_container, text="API Key:").pack(side=tk.LEFT, padx=(0, 5))

        self.key_entry = ttk.Entry(self.key_container, textvariable=self.api_key_var, show="*", width=20)
        self.key_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))

        self.enter_btn = ttk.Button(self.key_container, text="Connect", command=self.check_api_key, width=8)
        self.enter_btn.pack(side=tk.LEFT)

    # ═══════════════════════════════════════════════════════════════════
    #  API Key Validation — via MCP tool "validate_connection"
    # ═══════════════════════════════════════════════════════════════════

    def check_api_key(self):
        key = self.api_key_var.get().strip()
        if not key:
            messagebox.showerror("Error", "Please enter an API Key.")
            return

        if not self._mcp_connected:
            messagebox.showerror("Error", "MCP Server not ready yet. Please wait.")
            return

        self.key_status_label.config(text="Verifying...", foreground=self.colors["accent"])
        self.root.update_idletasks()

        def verify_wrapper():
            result = {}

            def target():
                try:
                    # ── MCP TOOL CALL ──
                    raw = self.mcp_client.call_tool(
                        "validate_connection", {"api_key": key}
                    )
                    data = json.loads(raw)
                    result['data'] = data
                except Exception as e:
                    result['error'] = str(e)

            t = threading.Thread(target=target)
            t.start()
            t.join(timeout=15)

            if t.is_alive():
                self.root.after(0, lambda: self._on_key_error("Timeout (15s). Check Network."))
                return

            if 'error' in result:
                self.root.after(0, lambda: self._on_key_error(result['error']))
            elif 'data' in result:
                data = result['data']
                if data.get("success"):
                    self.root.after(0, lambda: self._on_key_success(data))
                else:
                    self.root.after(0, lambda: self._on_key_error(data.get("message", "Unknown error")))
            else:
                self.root.after(0, lambda: self._on_key_error("Unknown Error"))

        threading.Thread(target=verify_wrapper, daemon=True).start()

    def _on_key_error(self, msg):
        short_msg = (msg[:40] + '...') if len(msg) > 40 else msg
        self.key_status_label.config(text=f"Error: {short_msg}", foreground=self.colors["error"])
        print(f"Key Error: {msg}")

    def _on_key_success(self, data):
        provider = data.get("provider", "gemini")
        current_model = data.get("model", "gemini-1.5-flash")
        models = data.get("available_models", [current_model])

        self.key_status_label.config(text=f"Connected ({provider})", foreground=self.colors["success"])

        self.save_config(self.api_key_var.get().strip())

        self.btn_open.config(state="normal")
        self.status_var.set(f"Ready. Using {current_model}.")

        # Switch to Active UI: [Label "Model:"] [Combobox] [Button "Change Key"]
        for widget in self.key_container.winfo_children():
            widget.destroy()

        ttk.Label(self.key_container, text="Model:").pack(side=tk.LEFT, padx=(0, 5))

        self.model_combo = ttk.Combobox(self.key_container, values=models, width=18, state="readonly")
        self.model_combo.set(current_model)
        self.model_combo.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))
        self.model_combo.bind("<<ComboboxSelected>>", self.on_model_changed)

        btn = ttk.Button(self.key_container, text="Change", width=8, command=self.reset_api_ui)
        btn.pack(side=tk.LEFT)

    def on_model_changed(self, event):
        new_model = self.model_combo.get()

        def switch():
            try:
                # ── MCP TOOL CALL ──
                self.mcp_client.call_tool("set_model", {"model_name": new_model})
                self.update_status(f"Switched to {new_model}")
            except Exception as e:
                self.update_status(f"Model switch failed: {e}")

        threading.Thread(target=switch, daemon=True).start()

    def reset_api_ui(self):
        self.api_key_var.set("")
        self.btn_open.config(state="disabled")
        self.key_status_label.config(text="")
        self._build_key_input_state()

    # ═══════════════════════════════════════════════════════════════════
    #  PDF Loading — via MCP tools
    # ═══════════════════════════════════════════════════════════════════

    def open_pdf(self):
        path = filedialog.askopenfilename(filetypes=[("PDF Files", "*.pdf")])
        if path:
            self.status_var.set(f"Loading {os.path.basename(path)}...")
            self.root.update()
            try:
                # ── MCP TOOL CALL: load_pdf ──
                raw = self.mcp_client.call_tool("load_pdf", {"path": path})
                data = json.loads(raw)
                self._pdf_page_count = data.get("page_count", 0)
                self._pdf_path = path  # remember for local rendering

                self.current_page = 0
                self.fit_to_page()
                self.update_page_label()
                self.status_var.set("PDF Loaded. Pre-loading context...")

                # Fetch Context in Background
                threading.Thread(target=self._fetch_context_thread, daemon=True).start()

            except Exception as e:
                self.status_var.set(f"Error loading PDF: {e}")

    def _fetch_context_thread(self):
        try:
            # ── MCP TOOL CALL: get_full_text ──
            full_text = self.mcp_client.call_tool("get_full_text")
            if not full_text:
                self.update_status("Context Load Failed (Empty).")
                return

            self.update_status(f"Context Loaded ({len(full_text)} chars). Locating bibliography...")

            # Default to full text (robust against LLM failures)
            self.current_context = full_text

            try:
                # ── MCP TOOL CALL: resolve_bibliography_range ──
                raw = self.mcp_client.call_tool(
                    "resolve_bibliography_range", {"full_text": full_text}
                )
                if raw and raw != "null":
                    range_info = json.loads(raw)
                    if "error" not in range_info:
                        start_page = range_info.get("start_page")
                        end_page = range_info.get("end_page")
                        if isinstance(start_page, int) and isinstance(end_page, int):
                            # ── MCP TOOL CALL: get_text_range ──
                            narrowed = self.mcp_client.call_tool(
                                "get_text_range",
                                {"start_page": start_page, "end_page": end_page}
                            )
                            if narrowed:
                                self.current_context = narrowed
                                self.update_status(f"Context narrowed to pages {start_page}-{end_page}. Ready.")
            except Exception as e:
                print(f"[WARN] Bibliography narrowing failed: {e}")
                self.update_status("Bibliography Auto-Locate Failed (Using Full Text).")

            # Detect citation style
            self._detect_style_in_background()
        except Exception as e:
            print(f"Context error: {e}")
            self.update_status("Context Load Failed (Check Console).")

    def _detect_style_in_background(self):
        try:
            page_limit = min(5, self._pdf_page_count)
            # ── MCP TOOL CALL: get_text_range ──
            first_pages_text = self.mcp_client.call_tool(
                "get_text_range", {"start_page": 1, "end_page": page_limit}
            )

            if first_pages_text:
                # ── MCP TOOL CALL: detect_citation_style ──
                style = self.mcp_client.call_tool(
                    "detect_citation_style", {"first_pages_text": first_pages_text}
                )
                self.citation_style_hint = style
                print(f"[DEBUG] Detected Citation Style: {style}")
                self.update_status(f"{self.status_var.get()} [Style: {style}]")
        except Exception as e:
            print(f"[WARN] Style detection failed: {e}")

    # ═══════════════════════════════════════════════════════════════════
    #  Page Rendering — via MCP tool "get_page_pixmap"
    # ═══════════════════════════════════════════════════════════════════

    def render_page(self):
        try:
            # ── MCP TOOL CALL: get_page_pixmap ──
            b64_png = self.mcp_client.call_tool(
                "get_page_pixmap",
                {"page_num": self.current_page, "zoom": self.zoom_level}
            )
            if not b64_png:
                return

            # Decode base64 PNG → PIL Image → ImageTk
            png_bytes = base64.b64decode(b64_png)
            img = Image.open(io.BytesIO(png_bytes))
            self.image_ref = ImageTk.PhotoImage(img)

            self.canvas.delete("all")
            canvas_w = self.canvas.winfo_width()
            canvas_h = self.canvas.winfo_height()
            self.canvas.create_image(canvas_w // 2, canvas_h // 2, anchor=tk.CENTER, image=self.image_ref)
        except Exception as e:
            print(f"[WARN] render_page failed: {e}")

    def prev_page(self):
        if self.current_page > 0:
            self.current_page -= 1
            self.fit_to_page()
            self.update_page_label()

    def next_page(self):
        if self.current_page < self._pdf_page_count - 1:
            self.current_page += 1
            self.fit_to_page()
            self.update_page_label()

    def update_page_label(self):
        self.lbl_page.config(text=f"Page: {self.current_page + 1}/{self._pdf_page_count}")

    def on_resize(self, event):
        self.fit_to_page()

    def fit_to_page(self):
        if self._pdf_page_count == 0:
            return
        canvas_w = self.canvas.winfo_width()
        canvas_h = self.canvas.winfo_height()
        if canvas_w > 10 and canvas_h > 10:
            try:
                # We need the page dimensions to calculate zoom.
                # For this, we do a quick render at zoom=1.0 to get dimensions,
                # or we can use fitz locally just for geometry.
                # To keep it pure-MCP, we'll request a pixmap at zoom=1.0
                # and compute from the image size. But that's wasteful.
                #
                # Pragmatic compromise: use fitz locally ONLY for page rect,
                # since geometry doesn't involve LLM or text extraction.
                # This avoids a round-trip for every resize event.
                import fitz as fitz_local
                if not hasattr(self, '_local_doc') or self._local_doc is None:
                    if hasattr(self, '_pdf_path'):
                        self._local_doc = fitz_local.open(self._pdf_path)
                    else:
                        return

                page = self._local_doc[self.current_page]
                scale_w = (canvas_w - 20) / page.rect.width
                scale_h = (canvas_h - 20) / page.rect.height
                self.zoom_level = min(scale_w, scale_h)
                if self.zoom_level < 0.1:
                    self.zoom_level = 0.1
                self.render_page()
            except Exception:
                pass

    # ═══════════════════════════════════════════════════════════════════
    #  Selection & Citation Resolution — via MCP tools
    # ═══════════════════════════════════════════════════════════════════

    def on_canvas_click(self, event):
        self.selection_start = (self.canvas.canvasx(event.x), self.canvas.canvasy(event.y))

    def on_canvas_drag(self, event):
        if not self.selection_start:
            return
        x, y = self.canvas.canvasx(event.x), self.canvas.canvasy(event.y)
        self.canvas.delete("selection_box")
        self.canvas.create_rectangle(self.selection_start[0], self.selection_start[1], x, y, outline="red", width=2, tags="selection_box")

    def on_canvas_release(self, event):
        if not self.selection_start:
            return
        x, y = self.canvas.canvasx(event.x), self.canvas.canvasy(event.y)
        start_x, start_y = self.selection_start

        if abs(x - start_x) > 5 or abs(y - start_y) > 5:
            canvas_w = self.canvas.winfo_width()
            canvas_h = self.canvas.winfo_height()
            if self.image_ref:
                img_w = self.image_ref.width()
                img_h = self.image_ref.height()
                offset_x = (canvas_w - img_w) // 2
                offset_y = (canvas_h - img_h) // 2
            else:
                offset_x = 0
                offset_y = 0

            x0 = (min(start_x, x) - offset_x) / self.zoom_level
            y0 = (min(start_y, y) - offset_y) / self.zoom_level
            x1 = (max(start_x, x) - offset_x) / self.zoom_level
            y1 = (max(start_y, y) - offset_y) / self.zoom_level

            # ── MCP TOOL CALL: get_text_in_rect ──
            def extract_and_resolve():
                try:
                    text = self.mcp_client.call_tool(
                        "get_text_in_rect",
                        {"page_num": self.current_page,
                         "x0": x0, "y0": y0, "x1": x1, "y1": y1}
                    )
                    if text and text.strip():
                        print(f"[DEBUG] User Selection: '{text}'")
                        self.update_status("Resolving selection with LLM...")
                        self._process_selection_mcp(text)
                    else:
                        self.update_status("Empty selection.")
                except Exception as e:
                    self.update_status(f"Selection error: {e}")

            threading.Thread(target=extract_and_resolve, daemon=True).start()

        self.canvas.delete("selection_box")
        self.selection_start = None

    def _process_selection_mcp(self, text):
        """Resolve a citation selection via the MCP resolve_citation tool."""
        try:
            args = {
                "selection_text": text,
                "context_text": self.current_context,
            }
            if self.citation_style_hint:
                args["style_hint"] = self.citation_style_hint

            # ── MCP TOOL CALL: resolve_citation ──
            result = self.mcp_client.call_tool("resolve_citation", args)

            if result:
                self.append_to_output(str(result) + "\n\n")
                self.update_status("Resolution Complete.")
            else:
                self.append_to_output("% No result returned.\n\n")
                self.update_status("Resolution Complete (Empty).")
        except Exception as e:
            err_str = str(e).lower()
            if "429" in err_str or "rate limit" in err_str or "quota" in err_str:
                msg = "% [Error] LLM rate limit exceeded. Please wait a moment."
            else:
                msg = f"% [Error] {e}"

            self.append_to_output(msg + "\n\n")
            self.update_status("Error: Rate Limit Exceeded" if "rate limit" in err_str or "429" in err_str else f"Error: {e}")

    # ═══════════════════════════════════════════════════════════════════
    #  Output & Utilities (unchanged from original)
    # ═══════════════════════════════════════════════════════════════════

    def append_to_output(self, text):
        self.root.after_idle(lambda: self._insert_text(text))

    def _insert_text(self, text):
        self.output_text.insert(tk.END, text)
        self.output_text.see(tk.END)

    def update_status(self, text):
        self.root.after_idle(lambda: self.status_var.set(text))

    def copy_to_clipboard(self):
        self.root.clipboard_clear()
        self.root.clipboard_append(self.output_text.get("1.0", tk.END))
        self.status_var.set("Copied to clipboard.")

    def clear_output(self):
        self.output_text.delete("1.0", tk.END)

    def show_context_menu(self, event):
        try:
            self.context_menu.tk_popup(event.x_root, event.y_root)
        finally:
            self.context_menu.grab_release()

    def copy_selection(self):
        try:
            sel = self.output_text.selection_get()
            self.root.clipboard_clear()
            self.root.clipboard_append(sel)
        except tk.TclError:
            pass


if __name__ == "__main__":
    root = tk.Tk()
    app = BibApp(root)
    root.mainloop()
