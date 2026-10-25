"""Serve the local avalanche research prototype UI."""
import argparse
import json
from datetime import date
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd

from config import MODEL_FILE
from predict import predict

UI_DIR = Path(__file__).resolve().parent / "ui"


class Handler(SimpleHTTPRequestHandler):
    model_path = MODEL_FILE

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(UI_DIR), **kwargs)

    def do_POST(self):
        if urlparse(self.path).path != "/api/predict":
            self.send_error(404)
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length))
            day = date.fromisoformat(payload["date"])
            aspect = float(payload["aspect"])
            if not 0 <= aspect <= 360:
                raise ValueError("Aspect must be between 0° and 360°.")
            result = predict(self.model_path, {"date": day.isoformat(), "aspect": aspect})
            result["date"] = day.isoformat()
            result["aspect"] = aspect
            body = json.dumps(result).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)
        except (ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
            self._json_error(400, str(exc))
        except FileNotFoundError:
            self._json_error(503, "Model not found. Train it first with: python train_model.py")
        except Exception as exc:
            self._json_error(500, f"Prediction failed: {exc}")

    def _json_error(self, status, message):
        body = json.dumps({"error": message}).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt, *args):
        print(f"[ui] {self.address_string()} - {fmt % args}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1", help="Bind address (default: localhost only)")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--model", type=Path, default=MODEL_FILE)
    args = parser.parse_args()
    Handler.model_path = args.model
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"Research prototype UI: http://{args.host}:{args.port}")
    print("Press Ctrl+C to stop. This is not for real safety decisions.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping UI server.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
