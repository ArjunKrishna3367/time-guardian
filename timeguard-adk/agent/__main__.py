"""Entry point for the TimeGuard agent using Google's Agent Development Kit.

This agent is responsible for deciding whether a given website is time-wasting
(e.g., social media, gaming, or entertainment) and returning a structured
response that can be used by a browser extension or backend service to block
or allow the site.

You will need to set the GOOGLE_API_KEY environment variable with a valid
key for the Google AI/Vertex model you want to use.
"""

import os
import sys
import json
from typing import Any, Dict
import traceback
import asyncio

from google.adk.agents import Agent, LlmAgent
from google.adk.apps.app import App, EventsCompactionConfig
from google.adk.models.google_llm import Gemini
from google.adk.sessions import DatabaseSessionService
from google.adk.sessions import InMemorySessionService
from google.adk.runners import InMemoryRunner
from google.adk.tools.tool_context import ToolContext
from google.genai import types
from dotenv import load_dotenv

load_dotenv()


TIME_WASTING_KEYWORDS = [
    "instagram.com",
    "facebook.com",
    "tiktok.com",
    "reddit.com",
    "youtube.com",
    "x.com",
    "twitter.com",
    "twitch.tv",
    "roblox.com",
    "steampowered.com",
    "epicgames.com",
]


LLM_MODEL_NAME = os.getenv("TIMEGUARD_MODEL_NAME", "gemini-2.0-flash")
_llm_agent: Agent | None = None

# Simple in-memory runner setup for local classification.
APP_NAME = "timeguard_app"
USER_ID = "timeguard_user"
SESSION_ID = "timeguard_session"
_runner: InMemoryRunner | None = None


def get_llm_agent() -> Agent:
    """Build (once) and return a Gemini-backed LlmAgent for URL classification.

    This uses Google ADK's LlmAgent under the hood. If anything goes wrong
    while constructing the agent (e.g., missing dependencies or misconfigured
    environment), callers should catch exceptions and fall back to a
    heuristic-only classification.
    """

    global _llm_agent
    if _llm_agent is not None:
        return _llm_agent

    # Simple single-agent setup. For this use case we don't need multi-agent
    # orchestration or complex session persistence; we just use one LLM agent.
    _llm_agent = LlmAgent(
        name="timeguard_url_classifier",
        model=LLM_MODEL_NAME,
        instruction="""You are a productivity assistant that decides whether a URL is a
            time-wasting site (like social media, entertainment, or gaming) or a
            productive/neutral site.

            Return ONLY a compact JSON object with the following keys:
            - is_time_wasting: true or false
            - reason: short string explanation

            Examples of time-wasting sites: instagram.com, tiktok.com, reddit.com,
            youtube.com, twitch.tv, roblox.com, steampowered.com, epicgames.com.
            """,
    )

    return _llm_agent


def get_runner() -> InMemoryRunner:
    """Build (once) and return an InMemoryRunner for the LLM agent.

    This is intended for local use/testing. It creates a single in-memory
    session that we reuse for all classifications.
    """

    global _runner
    if _runner is not None:
        return _runner

    agent = get_llm_agent()
    _runner = InMemoryRunner(agent=agent, app_name=APP_NAME)

    async def _ensure_session() -> None:
        session = await _runner.session_service.get_session(  # type: ignore[func-returns-value]
            app_name=APP_NAME,
            user_id=USER_ID,
            session_id=SESSION_ID,
        )
        if not session:
            await _runner.session_service.create_session(  # type: ignore[func-returns-value]
                app_name=APP_NAME,
                user_id=USER_ID,
                session_id=SESSION_ID,
            )

    # Create the in-memory session once.
    asyncio.run(_ensure_session())

    return _runner


def _heuristic_classification(url: str, context: str | None = None) -> Dict[str, Any]:
    """Fallback heuristic classification based on known domains."""

    url_lower = url.lower()
    is_time_wasting = any(keyword in url_lower for keyword in TIME_WASTING_KEYWORDS)

    reason = (
        "URL matches known time-wasting/social-media/gaming domains."
        if is_time_wasting
        else "URL does not match the list of known time-wasting domains."
    )

    return {
        "url": url,
        "is_time_wasting": is_time_wasting,
        "action": "block" if is_time_wasting else "allow",
        "reason": reason,
        "context": context or "",
    }


def classify_website(url: str, context: str | None = None) -> Dict[str, Any]:
    """Classify a website as time-wasting or not using Google ADK.

    This first attempts to use a Gemini-backed LlmAgent from the Agent
    Development Kit. If any error occurs (e.g., ADK not installed, model
    misconfigured, or unexpected response format), it falls back to a
    deterministic keyword-based heuristic.
    """

    # Try LLM-based classification via ADK Runner (sync run for local testing)
    try:
        runner = get_runner()
        prompt = json.dumps({
            "url": url,
            "context": context or "",
        })

        # Prepare Content for ADK
        message = types.UserContent(parts=[types.Part(text=prompt)])

        final_text: str | None = None

        # Runner.run yields Event objects; we take the last event with
        # non-empty text content from the agent/model.
        for event in runner.run(
            user_id=USER_ID,
            session_id=SESSION_ID,
            new_message=message,
        ):
            if not event.content or not event.content.parts:
                continue
            part = event.content.parts[0]
            if not getattr(part, "text", None):
                continue
            # Prefer responses authored by the agent/model, but in practice
            # taking the last text content is sufficient for this use case.
            final_text = part.text

        if final_text:
            text = final_text.strip()
            parsed: Dict[str, Any] | None = None

            # First, try to parse the entire string as JSON.
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError:
                # If the model wrapped JSON in extra text, try to extract the
                # JSON object between the first '{' and the last '}'.
                start = text.find("{")
                end = text.rfind("}")
                if start != -1 and end != -1 and end > start:
                    try:
                        parsed = json.loads(text[start : end + 1])
                    except json.JSONDecodeError:
                        parsed = None
                else:
                    parsed = None

            if isinstance(parsed, dict) and "is_time_wasting" in parsed:
                is_time_wasting = bool(parsed.get("is_time_wasting"))
                reason = str(parsed.get("reason", "")) or (
                    "Model indicated time-wasting site."
                    if is_time_wasting
                    else "Model indicated non-time-wasting site."
                )
                return {
                    "url": url,
                    "is_time_wasting": is_time_wasting,
                    "action": "block" if is_time_wasting else "allow",
                    "reason": reason,
                    "context": context or "",
                }
    except Exception as exc:  # pragma: no cover
        # Log and fall back to heuristic so callers still get a valid answer.
        print(f"TimeGuard LLM agent error, using heuristic instead: {exc}")
        traceback.print_exc()

    # Fallback path: deterministic heuristic.
    return _heuristic_classification(url, context=context)


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint for the TimeGuard agent.

    Usage (example):

        python -m agent --url https://www.instagram.com
    """

    argv = list(sys.argv[1:] if argv is None else argv)

    if "--help" in argv or "-h" in argv:
        print("Usage: python -m agent --url <URL> [--context <TEXT>]")
        return 0

    url = None
    context = None

    i = 0
    while i < len(argv):
        if argv[i] == "--url" and i + 1 < len(argv):
            url = argv[i + 1]
            i += 2
        elif argv[i] == "--context" and i + 1 < len(argv):
            context = argv[i + 1]
            i += 2
        else:
            i += 1

    if not url:
        print("Error: --url is required. Use --help for usage.", file=sys.stderr)
        return 1

    # For CLI usage, simply run the classifier once and print the result.

    result = classify_website(url, context=context)
    print(result)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
