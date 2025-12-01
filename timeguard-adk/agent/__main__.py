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
from datetime import datetime

from google.adk.agents import Agent, LlmAgent, SequentialAgent
from google.adk.apps.app import App, EventsCompactionConfig
from google.adk.models.google_llm import Gemini
from google.adk.sessions import DatabaseSessionService
from google.adk.sessions import InMemorySessionService
from google.adk.runners import Runner
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

# Simple runner setup for local classification, backed by a SQLite DB so
# sessions persist across runs.
APP_NAME = "timeguard_app"
USER_ID = "timeguard_user"
SESSION_ID = "timeguard_session"
_runner: Runner | None = None

# Separate configuration for the history reporting pipeline, which uses a
# SequentialAgent composed of a summarizer and a coaching agent.
HISTORY_APP_NAME = "timeguard_history_app"
HISTORY_SESSION_ID = "timeguard_history_session"
HISTORY_MODEL_NAME = os.getenv("TIMEGUARD_HISTORY_MODEL_NAME", LLM_MODEL_NAME)
_history_summarizer: Agent | None = None
_history_coach: Agent | None = None
_history_seq_agent: Agent | None = None
_history_runner: Runner | None = None


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


def get_runner() -> Runner:
    """Build (once) and return an InMemoryRunner for the LLM agent.

    This is intended for local use/testing. It creates a single in-memory
    session that we reuse for all classifications.
    """

    global _runner
    if _runner is not None:
        return _runner

    agent = get_llm_agent()

    # Use a local SQLite database for ADK sessions. The aiosqlite driver is
    # used by SQLAlchemy's async engine under the hood.
    db_url = "sqlite+aiosqlite:///timeguard_sessions.db"
    session_service = DatabaseSessionService(db_url=db_url)

    _runner = Runner(
        app_name=APP_NAME,
        agent=agent,
        session_service=session_service,
    )

    async def _ensure_session() -> None:
        session = await session_service.get_session(
            app_name=APP_NAME,
            user_id=USER_ID,
            session_id=SESSION_ID,
        )
        if not session:
            await session_service.create_session(
                app_name=APP_NAME,
                user_id=USER_ID,
                session_id=SESSION_ID,
            )

    # Lazily create the initial session once so subsequent calls can reuse it.
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


def get_history_events_from_db() -> list[Dict[str, Any]]:
    """Load URL classification history from the ADK SQLite session DB.

    This inspects the stored session events for the single configured
    (APP_NAME, USER_ID, SESSION_ID) and attempts to reconstruct a list of
    simplified history entries with shape:

        {"timestamp": str | None, "url": str, "blocked": bool, "reason": str}

    It pairs user messages (which carry the URL) with model responses
    (which carry is_time_wasting / reason) by shared invocation_id.
    """

    db_url = "sqlite+aiosqlite:///timeguard_sessions.db"
    session_service = DatabaseSessionService(db_url=db_url)

    async def _collect() -> list[Dict[str, Any]]:
        session = await session_service.get_session(
            app_name=APP_NAME,
            user_id=USER_ID,
            session_id=SESSION_ID,
        )
        if not session or not getattr(session, "events", None):
            return []

        history: list[Dict[str, Any]] = []
        agent_name = getattr(get_llm_agent(), "name", "")

        # Walk events in chronological order, pairing each user URL message
        # with the next model response that returns a JSON object containing
        # is_time_wasting / reason.
        pending_user: Dict[str, Any] | None = None
        # print(session.events)

        for ev in session.events:
            author = getattr(ev, "author", "") or ""
            content = getattr(ev, "content", None)

            if author == "user":
                if content is None or not getattr(content, "parts", None):
                    continue
                part = content.parts[0]
                text = getattr(part, "text", None)
                if not text:
                    continue
                try:
                    payload = json.loads(text)
                except Exception:
                    continue
                if not isinstance(payload, dict):
                    continue
                url = payload.get("url")
                if not isinstance(url, str):
                    continue
                ts = getattr(ev, "create_time", None)
                pending_user = {
                    "url": url,
                    "timestamp": str(ts) if ts is not None else None,
                }
                continue

            if author != agent_name:
                continue
            if not pending_user:
                continue
            if content is None or not getattr(content, "parts", None):
                continue
            part = content.parts[0]
            raw_text = getattr(part, "text", None)
            if not raw_text:
                continue

            text = raw_text.strip()

            # Model responses are currently wrapped in ```json code fences; try
            # to robustly extract the JSON object.
            if text.startswith("```"):
                # Drop leading/backtick language marker and trailing fence.
                # e.g. ```json\n{...}\n``` -> {...}
                # Remove leading ```json or ``` and trailing ```.
                text_no_fence = text.strip("`")
                # Fallback to bracket extraction below.
                text = text_no_fence

            # Try to locate a JSON object between the first '{' and last '}'.
            start = text.find("{")
            end = text.rfind("}")
            if start != -1 and end != -1 and end > start:
                candidate = text[start : end + 1]
            else:
                candidate = text

            try:
                payload = json.loads(candidate)
            except Exception:
                continue
            if not isinstance(payload, dict):
                continue
            if "is_time_wasting" not in payload:
                continue

            is_tw = bool(payload.get("is_time_wasting"))
            reason = str(payload.get("reason", ""))

            # Prefer the model event's timestamp if available; fall back to
            # the user event timestamp that we captured earlier.
            model_ts = getattr(ev, "timestamp", None)
            ts_str = (
                str(model_ts)
                if model_ts is not None
                else pending_user.get("timestamp")
            )

            history.append(
                {
                    "timestamp": ts_str,
                    "url": pending_user["url"],
                    "blocked": is_tw,
                    "reason": reason,
                }
            )

            pending_user = None

        return history

    try:
        return asyncio.run(_collect())
    except RuntimeError:
        # If we're already in an event loop (e.g. called from async context),
        # fall back to an empty list rather than crashing.
        return []


def unix_to_local_time(unix_timestamp: float) -> str:
    """Convert a UNIX timestamp (seconds since epoch) to a local human-readable string.

    Example output: "11:59 p.m. on Nov. 30". Falls back to the numeric value if
    parsing fails.
    """

    try:
        dt = datetime.fromtimestamp(unix_timestamp)
    except (ValueError, OSError, TypeError):
        return str(unix_timestamp)

    time_str = dt.strftime("%I:%M %p")  # e.g. "07:05 PM"
    time_str = time_str.lstrip("0")
    time_str = (
        time_str.replace("AM", "a.m.")
        .replace("PM", "p.m.")
    )

    month_day = dt.strftime("%b %d")  # e.g. "Nov 30"
    if " " in month_day:
        month, day = month_day.split(" ", 1)
        date_str = f"{month}. {day}"
    else:
        date_str = month_day

    return f"{time_str} on {date_str}"


def _get_history_summarizer() -> Agent:
    """Build (once) and return the history summarizer LlmAgent.

    This agent takes the raw browsing events JSON and produces a structured
    JSON summary. Its textual output is placed into the shared state under
    the key "history_summary_json" so that downstream agents can reference it
    in their instructions via {history_summary_json}.
    """

    global _history_summarizer
    if _history_summarizer is not None:
        return _history_summarizer

    _history_summarizer = LlmAgent(
        name="timeguard_history_summarizer",
        model=HISTORY_MODEL_NAME,
        instruction="""\
            You are a data analyst. You are given a JSON array of browsing events.
            Each event has: timestamp (ISO string, may be null), url, blocked (boolean), and reason (string).

            Analyze this history and return ONLY a single JSON object (no extra text) with fields such as:
            - top_sites: list of {"domain": string, "blocks": integer}
            - peak_hours: list of {"range": string, "blocks": integer}
            - totals: {"total_events": int, "total_blocks": int}
            - any other fields you find useful for downstream coaching.

            The output must be valid JSON and must NOT be wrapped in markdown code fences.
        """,
        description="Summarizes raw browsing events into structured JSON.",
        output_key="history_summary_json",
    )

    return _history_summarizer


def _get_history_coach() -> Agent:
    """Build (once) and return the history coaching LlmAgent.

    This agent reads the JSON summary from the shared state key
    "history_summary_json" (produced by the summarizer) and turns it into a
    plain-text productivity report.
    """

    global _history_coach
    if _history_coach is not None:
        return _history_coach

    _history_coach = LlmAgent(
        name="timeguard_history_coach",
        model=HISTORY_MODEL_NAME,
        instruction="""\
            You are a productivity coach. You are given a JSON summary of a user's browsing history,
            provided in the placeholder {history_summary_json}.

            Using ONLY the information in this summary JSON, produce a concise plain-text report
            for the user that:
            - Describes the top time-wasting sites and how often they were blocked.
            - Describes any noticeable time-of-day patterns where blocking is frequent.
            - Provides 2-3 actionable suggestions to improve focus.

            You also have access to a tool called unix_to_local_time which takes a UNIX
            timestamp (seconds since epoch) and returns a human-readable local time such
            as "11:59 p.m. on Nov. 30". When explaining time-of-day patterns or giving
            examples of when blocking is frequent, you must call this tool to turn raw
            timestamps into clearer descriptions of local times.

            IMPORTANT:
            - Do NOT use any markdown formatting.
            - Do NOT use headings, bullet points, numbered lists, or symbols like #, *, -, or •.
            - Write as one or more simple paragraphs of plain text only.
            - If any UNIX timestamps (10 digit numbers such as 1764565000) appear, use the 
            unix_to_local_time tool to rewrite them into a human-readable version.

            Keep the report under 250 words.
        """,
        description="Turns a history summary JSON into a user-facing productivity report.",
        output_key="history_report_text",
        tools=[unix_to_local_time],
    )

    return _history_coach


def _get_history_sequential_agent() -> Agent:
    """Build (once) and return the SequentialAgent for history reporting.

    The pipeline is:
      1) Summarizer -> produces history_summary_json
      2) Coach      -> produces history_report_text
    """

    global _history_seq_agent
    if _history_seq_agent is not None:
        return _history_seq_agent

    summarizer = _get_history_summarizer()
    coach = _get_history_coach()

    _history_seq_agent = SequentialAgent(
        name="timeguard_history_pipeline",
        sub_agents=[summarizer, coach],
        description=(
            "Executes a sequence of history summarization and productivity coaching "
            "over the user's browsing events."
        ),
    )

    return _history_seq_agent


def _get_history_runner() -> Runner:
    """Return a Runner configured for the history SequentialAgent."""

    global _history_runner
    if _history_runner is not None:
        return _history_runner

    db_url = "sqlite+aiosqlite:///timeguard_sessions.db"
    session_service = DatabaseSessionService(db_url=db_url)

    seq_agent = _get_history_sequential_agent()

    _history_runner = Runner(
        app_name=HISTORY_APP_NAME,
        agent=seq_agent,
        session_service=session_service,
    )

    async def _ensure_session() -> None:
        session = await session_service.get_session(
            app_name=HISTORY_APP_NAME,
            user_id=USER_ID,
            session_id=HISTORY_SESSION_ID,
        )
        if not session:
            await session_service.create_session(
                app_name=HISTORY_APP_NAME,
                user_id=USER_ID,
                session_id=HISTORY_SESSION_ID,
            )

    asyncio.run(_ensure_session())

    return _history_runner


def generate_history_report_with_adk(events: list[Dict[str, Any]]) -> str:
    """Run the history SequentialAgent over the given events and return a report.

    The raw events JSON is provided as the initial user message. The
    SequentialAgent first summarizes the events, then produces a coaching
    report. We return the final text generated by the coach.
    """

    runner = _get_history_runner()

    payload = json.dumps(events, indent=2)
    message = types.UserContent(
        parts=[types.Part(text="Browsing events JSON:\n" + payload)]
    )

    final_text: str | None = None

    for event in runner.run(
        user_id=USER_ID,
        session_id=HISTORY_SESSION_ID,
        new_message=message,
    ):
        if not event.content or not getattr(event.content, "parts", None):
            continue
        part = event.content.parts[0]
        if not getattr(part, "text", None):
            continue
        final_text = part.text

    return final_text or "No history report could be generated."


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
