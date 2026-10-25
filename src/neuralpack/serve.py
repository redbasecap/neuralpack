"""``npk serve``: a tiny static server for the browser viewer.

Serves the packaged ``web/`` directory at ``/`` and the exported priors at ``/models/``.
Nothing is uploaded anywhere: the viewer decodes in the browser; this server only hands
out static files. For a public deployment copy ``web/`` and the models directory to any
static host (GitHub Pages, S3, nginx).
"""

from __future__ import annotations

import http.server
import mimetypes
import threading
import webbrowser
from functools import partial
from importlib import resources
from pathlib import Path

from .webexport import default_models_dir, exportable

mimetypes.add_type("application/javascript", ".js")
mimetypes.add_type("application/wasm", ".wasm")
mimetypes.add_type("application/octet-stream", ".onnx")
mimetypes.add_type("application/octet-stream", ".bin")
mimetypes.add_type("application/octet-stream", ".npk")


class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, web_dir: Path, models_dir: Path, **kw):
        self.web_dir, self.models_dir = web_dir, models_dir
        super().__init__(*args, directory=str(web_dir), **kw)

    def translate_path(self, path: str) -> str:
        path = path.split("?", 1)[0].split("#", 1)[0]
        if path.startswith("/models/"):
            rel = Path(path[len("/models/"):])
            if ".." in rel.parts:
                return str(self.models_dir / "__forbidden__")
            return str(self.models_dir / rel)
        return super().translate_path(path)

    def end_headers(self) -> None:
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Access-Control-Allow-Origin", "*")
        super().end_headers()

    def log_message(self, fmt, *args):  # quieter
        if "models/" in (args[0] if args else "") or self.command != "GET":
            return
        super().log_message(fmt, *args)


def web_dir() -> Path:
    return Path(str(resources.files("neuralpack").joinpath("web")))


def ensure_exports(models_dir: Path) -> None:
    from .webexport import export_model

    for model, q in exportable():
        if not (models_dir / f"{model}-q{q}" / "manifest.json").exists():
            print(f"exporting {model} q{q} for the browser (one-off) …", flush=True)
            export_model(models_dir, model, q, device="cpu")


def serve(port: int = 8765, models_dir: Path | None = None, open_browser: bool = True,
          export_missing: bool = True) -> int:
    models_dir = models_dir or default_models_dir()
    models_dir.mkdir(parents=True, exist_ok=True)
    if export_missing:
        ensure_exports(models_dir)
    handler = partial(Handler, web_dir=web_dir(), models_dir=models_dir)
    httpd = None
    for candidate in range(port, port + 20):
        try:
            httpd = http.server.ThreadingHTTPServer(("127.0.0.1", candidate), handler)
            break
        except OSError:
            continue
    if httpd is None:
        raise RuntimeError(f"no free port in {port}-{port + 19}")
    if candidate != port:
        print(f"port {port} is busy, using {candidate}")
    url = f"http://127.0.0.1:{candidate}/"
    print(f"npk viewer at {url}  (models from {models_dir}; Ctrl-C to stop)")
    if open_browser:
        threading.Timer(0.5, webbrowser.open, args=(url,)).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()
    return 0
