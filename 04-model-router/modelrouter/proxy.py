"""
An OpenAI-compatible HTTP proxy, built on Python's stdlib http.server so it
has zero extra dependencies. Point any OpenAI-client-library app at this
server's URL instead of api.openai.com, and every request gets transparently
routed/escalated -- no code changes needed in the calling app beyond the
base_url.

For production use, a real ASGI framework (FastAPI/Starlette) is recommended
for concurrency; this stdlib version is intentionally simple so the whole
proxy can be read and understood (and run) without installing anything.
"""
from __future__ import annotations
import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from .router import Router


def make_handler(router: Router):
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            if self.path.rstrip("/") != "/v1/chat/completions":
                self.send_response(404)
                self.end_headers()
                return

            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length))
            messages = body.get("messages", [])
            prompt = "\n".join(m.get("content", "") for m in messages)

            decision = router.route(prompt)

            response = {
                "id": "chatcmpl-routed",
                "object": "chat.completion",
                "model": decision.provider_used,
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": decision.result.text},
                    "finish_reason": "stop",
                }],
                "usage": {
                    "prompt_tokens": decision.result.input_tokens,
                    "completion_tokens": decision.result.output_tokens,
                    "total_tokens": decision.result.input_tokens + decision.result.output_tokens,
                },
                "x_router_meta": {
                    "escalated": decision.escalated,
                    "confidence": decision.confidence,
                    "cost": decision.cost,
                },
            }
            payload = json.dumps(response).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, fmt, *args):
            pass  # keep stdout clean; swap for real logging in production

    return Handler


def serve(router: Router, port: int = 8000):
    server = HTTPServer(("0.0.0.0", port), make_handler(router))
    print(f"modelrouter proxy listening on http://localhost:{port}/v1")
    server.serve_forever()
