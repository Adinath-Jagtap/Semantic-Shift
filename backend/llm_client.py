"""
Groq LLM client — sends chat completion requests and returns the response text.

# NOTE: Streaming responses are a known v1 gap. This module sends a single
# request and waits for the full response. See KNOWN LIMITATIONS in main.py.
"""

from groq import Groq, APIError, APIConnectionError

from backend.config import settings


class GroqAPIError(Exception):
    """Raised when the Groq API returns an error or is unreachable."""

    def __init__(self, status_code: int, body: str) -> None:
        self.status_code = status_code
        self.body = body
        super().__init__(f"Groq API error (HTTP {status_code}): {body}")


# Reusable client — initialized once, shares a connection pool internally.
_client: Groq | None = None


def _get_client() -> Groq:
    """Lazy-initialize the Groq SDK client."""
    global _client
    if _client is None:
        # The Groq SDK reads the key from the constructor argument.
        # We never log or expose settings.groq_api_key beyond this point.
        _client = Groq(api_key=settings.groq_api_key)
    return _client


def call_groq(query: str) -> str:
    """
    Send a single user message to the Groq chat completions endpoint.

    Args:
        query: The user's question / prompt text.

    Returns:
        The assistant's response text.

    Raises:
        GroqAPIError: If the Groq API returns an error or is unreachable.
            The caller (main.py) catches this and returns a 502 JSON body.
    """
    client = _get_client()

    try:
        response = client.chat.completions.create(
            model=settings.groq_model,
            messages=[{"role": "user", "content": query}],
        )
        return response.choices[0].message.content or ""

    except APIConnectionError as exc:
        raise GroqAPIError(
            status_code=503,
            body=f"Could not connect to Groq API: {exc}",
        ) from exc

    except APIError as exc:
        raise GroqAPIError(
            status_code=exc.status_code or 500,
            body=str(exc.body) if exc.body else str(exc),
        ) from exc

    except Exception as exc:
        # Catch-all so a Groq outage never crashes the whole server.
        raise GroqAPIError(
            status_code=500,
            body=f"Unexpected error calling Groq: {exc}",
        ) from exc
