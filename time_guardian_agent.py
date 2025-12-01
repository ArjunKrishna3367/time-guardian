import os
from typing import Dict, List
import google.generativeai as genai
from flask import Flask, request, jsonify
from datetime import datetime, timedelta
import json
import importlib.util
from pathlib import Path
from dotenv import load_dotenv

app = Flask(__name__)
load_dotenv()


# Configuration
CONFIG = {
    "time_wasting_sites": [
        "instagram.com",
        "facebook.com",
        "twitter.com",
        "reddit.com",
        "youtube.com",
        "netflix.com",
        "tiktok.com",
        "twitch.tv",
        "pinterest.com",
        "9gag.com"
    ],
    "block_duration_minutes": 30,
    "whitelist": []
}

# In-memory storage for blocked sites and their unblock times
blocked_sites = {}

# Last AI/heuristic reason computed for the most recent URL classification.
_last_ai_reason: str | None = None


def _format_unblock_time_human(unblock_time_iso: str) -> str:
    """Format an ISO timestamp into a human-readable string.

    Example output: "11:59 p.m. on Nov. 30".
    Falls back to the original string if parsing fails.
    """

    try:
        dt = datetime.fromisoformat(unblock_time_iso)
    except ValueError:
        return unblock_time_iso

    # Time like "11:59 p.m."
    time_str = dt.strftime("%I:%M %p")  # e.g. "07:05 PM"
    time_str = time_str.lstrip("0")
    time_str = (
        time_str.replace("AM", "a.m.")
        .replace("PM", "p.m.")
    )

    # Date like "Nov. 30"
    month_day = dt.strftime("%b %d")  # e.g. "Nov 30"
    if " " in month_day:
        month, day = month_day.split(" ", 1)
        date_str = f"{month}. {day}"
    else:
        date_str = month_day

    return f"{time_str} on {date_str}"


_timeguard_agent_module = None


def load_timeguard_agent():
    """Dynamically load the TimeGuard agent module.

    This avoids having to restructure the project into an installable package
    while still letting the backend reuse the agent's classification logic.
    """

    global _timeguard_agent_module
    if _timeguard_agent_module is not None:
        return _timeguard_agent_module

    base_dir = Path(__file__).parent
    agent_path = base_dir / "timeguard-adk" / "agent" / "__main__.py"

    if not agent_path.exists():
        raise RuntimeError(f"TimeGuard agent module not found at {agent_path}")

    spec = importlib.util.spec_from_file_location("timeguard_time_agent", agent_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Unable to create spec for TimeGuard agent module")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    _timeguard_agent_module = module
    return module

# Initialize Google's Generative AI
# Note: You'll need to set the GOOGLE_API_KEY environment variable
api_key = os.getenv('GOOGLE_API_KEY')
print(api_key)
genai.configure(api_key=api_key)
model = genai.GenerativeModel('gemini-pro')

def is_time_wasting_site(url: str) -> bool:
    """Check if a URL is in our list of time-wasting sites."""
    return any(site in url.lower() for site in CONFIG["time_wasting_sites"])

def should_block_site(url: str) -> bool:
    """Determine if a site should be blocked based on various factors."""
    global _last_ai_reason

    # Check if site is whitelisted
    if any(whitelisted in url.lower() for whitelisted in CONFIG["whitelist"]):
        _last_ai_reason = "Site is whitelisted."
        return False

    # Delegate classification to the TimeGuard agent, falling back to the
    # local heuristic if anything goes wrong.
    try:
        agent_module = load_timeguard_agent()
        classify_website = getattr(agent_module, "classify_website", None)
        if classify_website is not None:
            result: Dict = classify_website(url)
            action = result.get("action")
            _last_ai_reason = str(result.get("reason", "")) or None
            if action == "block":
                return True
            if action == "allow":
                return False
    except Exception as exc:  # pragma: no cover
        # If the agent cannot be loaded or executed, fall back to the
        # built-in heuristic so the service continues to function.
        print(f"TimeGuard agent error, falling back to heuristic: {exc}")

    # Fallback: Check if site is in our heuristic list.
    is_tw = is_time_wasting_site(url)
    if is_tw:
        _last_ai_reason = "URL matches known time-wasting/social-media/gaming domains."
    else:
        _last_ai_reason = "URL does not match the list of known time-wasting domains."
    return is_tw

def block_site(url: str):
    """Block a site for the configured duration."""
    unblock_time = datetime.now() + timedelta(minutes=CONFIG["block_duration_minutes"])
    blocked_sites[url] = unblock_time.isoformat()
    # In a real implementation, you would modify the hosts file or use system firewall rules
    print(f"Blocked {url} until {unblock_time}")

def check_blocked_sites():
    """Check and unblock any sites whose block time has expired."""
    current_time = datetime.now()
    to_remove = []
    
    for url, unblock_time_str in blocked_sites.items():
        unblock_time = datetime.fromisoformat(unblock_time_str)
        if current_time >= unblock_time:
            to_remove.append(url)
    
    for url in to_remove:
        del blocked_sites[url]
        print(f"Unblocked {url}")

# API Endpoints
@app.route('/check_url', methods=['POST'])
def check_url():
    """Check if a URL should be blocked."""
    data = request.get_json()
    url = data.get('url', '')
    
    if not url:
        return jsonify({"error": "URL is required"}), 400
    
    check_blocked_sites()
    
    if url in blocked_sites:
        unblock_time_iso = blocked_sites[url]
        return jsonify({
            "blocked": True,
            "reason": "Site is currently blocked",
            "unblock_time": unblock_time_iso,
            "unblock_time_human": _format_unblock_time_human(unblock_time_iso),
        })
    
    if should_block_site(url):
        block_site(url)
        unblock_time_iso = blocked_sites[url]
        return jsonify({
            "blocked": True,
            "reason": _last_ai_reason or "Time-wasting site detected",
            "unblock_time": unblock_time_iso,
            "unblock_time_human": _format_unblock_time_human(unblock_time_iso),
        })
    
    return jsonify({
        "blocked": False,
        "reason": _last_ai_reason or "This site is not considered time-wasting right now.",
    })

@app.route('/blocked_sites', methods=['GET'])
def get_blocked_sites():
    """Get the list of currently blocked sites."""
    check_blocked_sites()
    return jsonify({"blocked_sites": blocked_sites})

@app.route('/whitelist', methods=['POST'])
def add_to_whitelist():
    """Add a site to the whitelist."""
    data = request.get_json()
    site = data.get('site', '').lower()
    
    if not site:
        return jsonify({"error": "Site is required"}), 400
    
    if site not in CONFIG["whitelist"]:
        CONFIG["whitelist"].append(site)
    
    return jsonify({"message": f"Added {site} to whitelist", "whitelist": CONFIG["whitelist"]})


@app.route('/ui', methods=['GET'])
def ui():
    """Serve a simple web UI for manually checking URLs."""

    return """<!DOCTYPE html>
<html lang=\"en\">
<head>
  <meta charset=\"UTF-8\" />
  <title>Time Guardian - URL Checker</title>
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
  <style>
    body { font-family: system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; margin: 2rem; background: #0f172a; color: #e5e7eb; }
    h1 { color: #f97316; }
    .card { max-width: 640px; background: #020617; border-radius: 0.75rem; padding: 1.5rem; box-shadow: 0 10px 40px rgba(15,23,42,0.7); border: 1px solid rgba(148,163,184,0.25); }
    label { display: block; margin-bottom: 0.5rem; font-weight: 600; }
    input[type=text] { width: 100%; padding: 0.6rem 0.75rem; border-radius: 0.5rem; border: 1px solid #334155; background: #020617; color: #e5e7eb; }
    input[type=text]:focus { outline: none; border-color: #f97316; box-shadow: 0 0 0 1px rgba(249,115,22,0.5); }
    button { margin-top: 0.75rem; padding: 0.55rem 1.25rem; border-radius: 999px; border: none; background: linear-gradient(to right, #f97316, #fb7185); color: #0f172a; font-weight: 600; cursor: pointer; }
    button:disabled { opacity: 0.6; cursor: default; }
    .result { margin-top: 1.25rem; padding: 0.75rem 1rem; border-radius: 0.5rem; background: #020617; border: 1px solid #1f2937; font-size: 0.95rem; }
    .blocked { border-color: #f97316; color: #fed7aa; }
    .allowed { border-color: #22c55e; color: #bbf7d0; }
    .error { border-color: #ef4444; color: #fecaca; }
    small { color: #9ca3af; }
  </style>
</head>
<body>
  <div class=\"card\">
    <h1>Time Guardian URL Checker</h1>
    <p><small>Enter a full URL (e.g. https://www.instagram.com) to see if Time Guardian would block it.</small></p>
    <label for=\"urlInput\">Website URL</label>
    <input id=\"urlInput\" type=\"text\" placeholder=\"https://www.example.com\" />
    <button id=\"checkBtn\">Check URL</button>
    <div id=\"result\" class=\"result\" style=\"display:none;\"></div>
  </div>

  <script>
    const input = document.getElementById('urlInput');
    const button = document.getElementById('checkBtn');
    const resultEl = document.getElementById('result');

    async function checkUrl() {
      const url = input.value.trim();
      if (!url) {
        showResult('Please enter a URL to check.', 'error');
        return;
      }

      button.disabled = true;
      showResult('Checking...', '');

      try {
        const resp = await fetch('/check_url', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ url })
        });

        const data = await resp.json();

        if (!resp.ok) {
          const message = data.error || 'Request failed.';
          showResult('Error: ' + message, 'error');
          return;
        }

        if (data.blocked) {
          const reason = data.reason || 'Time-wasting site detected.';
          const human = data.unblock_time_human || data.unblock_time;
          const until = human ? ' (Blocked until ' + human + ')' : '';
          showResult('BLOCKED: ' + reason + until, 'blocked');
        } else {
          showResult('ALLOWED: This site is not currently blocked.', 'allowed');
        }
      } catch (err) {
        showResult('Error calling backend: ' + err, 'error');
      } finally {
        button.disabled = false;
      }
    }

    function showResult(text, kind) {
      resultEl.style.display = 'block';
      resultEl.textContent = text;
      resultEl.className = 'result';
      if (kind) {
        resultEl.classList.add(kind);
      }
    }

    button.addEventListener('click', (e) => {
      e.preventDefault();
      checkUrl();
    });

    input.addEventListener('keydown', (e) => {
      if (e.key === 'Enter') {
        e.preventDefault();
        checkUrl();
      }
    });
  </script>
</body>
</html>
"""

if __name__ == '__main__':
    # Load configuration from file if it exists
    try:
        with open('config.json', 'r') as f:
            CONFIG.update(json.load(f))
    except FileNotFoundError:
        # Save default config
        with open('config.json', 'w') as f:
            json.dump(CONFIG, f, indent=2)
    
    print("Starting Time Guardian Agent...")
    print(f"Blocking time-wasting sites: {', '.join(CONFIG['time_wasting_sites'])}")
    app.run(host='0.0.0.0', port=5000, debug=True)
