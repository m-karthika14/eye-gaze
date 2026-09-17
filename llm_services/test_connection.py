"""Phase 0 deliverable: verify a basic Groq chat-completion call works end to end.

Usage:
    python llm_services/test_connection.py
"""

import sys

from groq_client import DEFAULT_SUGGESTION_MODEL, GroqClient


def main() -> int:
    try:
        client = GroqClient()
    except RuntimeError as exc:
        print(f"[FAIL] {exc}")
        return 1

    try:
        reply = client.chat_completion(
            prompt="Reply with exactly the single word: OK",
            model=DEFAULT_SUGGESTION_MODEL,
            max_tokens=5,
            temperature=0.0,
        )
    except Exception as exc:  # noqa: BLE001 - surface any SDK/network error plainly
        print(f"[FAIL] Groq API call failed: {exc}")
        return 1

    print(f"[OK] model={DEFAULT_SUGGESTION_MODEL} reply={reply!r}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
