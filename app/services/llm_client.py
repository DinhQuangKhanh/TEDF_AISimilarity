"""Tiny OpenAI-compatible chat client for the "explain duplication" wording.

**The LLM is OPT-IN.** If ``LLM_BASE_URL`` is not set, this returns ``None`` immediately and the caller
uses the grounded, deterministic template — free, offline, and always aligned with the highlights.
Set the three env vars below only if you want an LLM to phrase the explanations more fluently.

Free, hosted, OpenAI-compatible options (0 RAM on our server) — pick one that works from your region:

* **OpenRouter** — has genuinely-free models (no prepaid billing), reachable from Vietnam:
      LLM_BASE_URL=https://openrouter.ai/api/v1
      LLM_API_KEY=<free key from https://openrouter.ai/keys>
      LLM_MODEL=meta-llama/llama-3.3-70b-instruct:free
* **Groq** — free & fast, but **geo-blocked in Vietnam** ("Access denied"), so not usable from here.
* **Google Gemini** — OpenAI-compatible, but its free tier is not available on every account/region
  (returns 429 "prepayment credits are depleted"); needs paid billing there.

Fully-local option (offline, no key): **Ollama** — `ollama serve` + a small model. A 3B model wants
~3–4 GB RAM (not a 2 GB box); `qwen2.5:0.5b-instruct` (~0.5 GB) fits but is slower on CPU:
      LLM_BASE_URL=http://localhost:11434/v1
      LLM_MODEL=qwen2.5:0.5b-instruct

Any failure (unset, unreachable, 401/429, timeout, bad response) returns ``None`` so the caller falls
back to the template — the feature must work even with no LLM at all.

Configure via env:
    LLM_BASE_URL   unset ⇒ LLM disabled (template is used); set to enable one of the options above
    LLM_MODEL      default qwen2.5:0.5b-instruct  (override to match your provider's model id)
    LLM_API_KEY    required for hosted providers; ignored by Ollama
    LLM_TIMEOUT    seconds, default 40
"""

from __future__ import annotations

import os
import time

import httpx

from app.core.logging import logger

LLM_BASE_URL = os.getenv("LLM_BASE_URL", "").strip()
LLM_MODEL = os.getenv("LLM_MODEL", "qwen2.5:0.5b-instruct")
LLM_API_KEY = os.getenv("LLM_API_KEY", "").strip()
LLM_TIMEOUT = float(os.getenv("LLM_TIMEOUT", "40"))
# Free/shared models throttle intermittently (HTTP 429 "retry shortly"); one short retry recovers many
# of those without dropping to the template. Kept small so a user-triggered explain never hangs.
LLM_RETRIES = int(os.getenv("LLM_RETRIES", "1"))
LLM_RETRY_DELAY = float(os.getenv("LLM_RETRY_DELAY", "2.0"))


def llm_available() -> bool:
    """Best-effort reachability check of the chat server (for status/diagnostics)."""
    if not LLM_BASE_URL:
        return False
    try:
        base = LLM_BASE_URL.rstrip("/")
        root = base[: -len("/v1")] if base.endswith("/v1") else base
        return httpx.get(root, timeout=3).status_code < 500
    except Exception:  # noqa: BLE001
        return False


def call_llm(prompt: str) -> str | None:
    """Send one user prompt to the chat endpoint; return the text, or ``None`` on any failure.

    Short-circuits to ``None`` when no ``LLM_BASE_URL`` is configured (LLM is opt-in), so the default
    path costs nothing and never makes a doomed network call. On a transient throttle (429/503) it
    retries up to ``LLM_RETRIES`` times after a short delay before giving up to the template.
    """
    if not LLM_BASE_URL:
        return None  # LLM opt-in: not configured → caller uses the grounded template

    headers = {"Content-Type": "application/json"}
    if LLM_API_KEY:
        headers["Authorization"] = f"Bearer {LLM_API_KEY}"
    payload = {
        "model": LLM_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
    }
    url = f"{LLM_BASE_URL.rstrip('/')}/chat/completions"

    for attempt in range(LLM_RETRIES + 1):
        try:
            response = httpx.post(url, json=payload, headers=headers, timeout=LLM_TIMEOUT)
            response.raise_for_status()
            content = response.json()["choices"][0]["message"]["content"]
            return content.strip() if isinstance(content, str) else None
        except httpx.HTTPStatusError as exc:  # log the provider's message (quota/billing/throttle) to help debugging
            status = exc.response.status_code
            if status in (429, 503) and attempt < LLM_RETRIES:
                time.sleep(LLM_RETRY_DELAY)
                continue  # transient throttle on a shared/free model — retry shortly
            body = " ".join(exc.response.text.split())[:200]
            logger.info("LLM HTTP %s: %s — using template fallback.", status, body)
            return None
        except Exception as exc:  # noqa: BLE001 - any other failure → None so caller falls back to template
            logger.info("LLM unavailable (%s); using template fallback.", type(exc).__name__)
            return None
    return None
