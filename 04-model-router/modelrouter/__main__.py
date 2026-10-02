"""`python -m modelrouter` starts the proxy server with env-configured providers."""
import os
import sys
from .router import Router
from .providers import OpenAICompatibleProvider
from .proxy import serve


def main():
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("Set OPENAI_API_KEY to run the live proxy (see README for config options).")
        sys.exit(1)

    cheap = OpenAICompatibleProvider("cheap", "gpt-4o-mini", api_key, 0.00015, 0.0006)
    expensive = OpenAICompatibleProvider("expensive", "gpt-4o", api_key, 0.0025, 0.01)
    router = Router(cheap, expensive, confidence_threshold=float(os.environ.get("CONFIDENCE_THRESHOLD", 0.6)))
    serve(router, port=int(os.environ.get("PORT", 8000)))


if __name__ == "__main__":
    main()
