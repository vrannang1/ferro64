#!/usr/bin/env python3
"""Optional MLX image backend. Default Rust image generation uses Ollama.

This worker loads FLUX on demand and releases weights after every job.
Run with an environment that already has mflux installed; no startup download.
"""
import gc
import json
import os
import sys
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

PORT = int(os.environ.get("FLUX_WORKER_PORT", "8001"))
model = None


def unload():
    global model
    model = None
    gc.collect()
    try:
        import mlx.core as mx
        mx.synchronize()
        mx.clear_cache()
    except ImportError:
        pass


def load():
    global model
    if model is None:
        from mflux.models.common.resolution.config_resolution import ConfigResolution
        from mflux.models.flux2.cli.flux2_generate import DEFAULT_MODEL, FAMILY_MODELS
        from mflux.models.flux2.variants import Flux2Klein
        config = ConfigResolution.resolve_restricted(
            "flux2-klein-4b", DEFAULT_MODEL, extra_keys=FAMILY_MODELS
        )
        model = Flux2Klein(model_config=config, quantize=4)
    return model


class FluxWorkerHandler(BaseHTTPRequestHandler):
    def respond(self, status, value):
        body = json.dumps(value).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/health":
            self.respond(200, {"status": "ready", "model": "flux2-klein-4b", "resident": model is not None})
        else:
            self.respond(404, {"error": "Route not found"})

    def do_POST(self):
        if self.path == "/unload":
            try:
                unload()
                self.respond(200, {"success": True, "resident": False})
            except Exception as error:
                self.respond(500, {"error": str(error)})
            return
        if self.path != "/generate":
            self.respond(404, {"error": "Route not found"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= 1_048_576:
                raise ValueError("Request must be between 1 byte and 1 MB")
            data = json.loads(self.rfile.read(length))
            prompt = str(data.get("prompt", "")).strip()
            output = data.get("output")
            if not prompt or not isinstance(output, str):
                raise ValueError("prompt and output are required")
            width, height = int(data.get("width", 512)), int(data.get("height", 512))
            steps = int(data.get("steps", 4))
            if not (64 <= width <= 2048 and 64 <= height <= 2048 and 1 <= steps <= 50):
                raise ValueError("Invalid dimensions or step count")
            seed = int(data.get("seed", time.time_ns() % 1_000_000))
            start = time.monotonic()
            try:
                image = load().generate_image(
                    seed=seed, prompt=prompt, width=width, height=height, num_inference_steps=steps
                )
                image.save(path=output)
                del image
            finally:
                unload()
            self.respond(200, {"status": "ok", "width": width, "height": height,
                               "seed": seed, "duration_ms": int((time.monotonic() - start) * 1000)})
        except Exception as error:
            print(f"FLUX generation failed: {error}", file=sys.stderr)
            self.respond(500, {"error": str(error)})

    def log_message(self, format, *args):
        pass


if __name__ == "__main__":
    print(f"Optional MLX worker listening on 127.0.0.1:{PORT}; weights load on demand.")
    HTTPServer(("127.0.0.1", PORT), FluxWorkerHandler).serve_forever()
