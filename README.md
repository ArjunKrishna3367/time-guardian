# Time Guardian

Time Guardian is a local AI-powered productivity assistant that monitors your browsing and blocks time-wasting sites through a Chrome extension, while providing periodic productivity reports based on your history.

---

## 1. Prerequisites

- **Python** 3.10+ (3.11/3.12 are fine)
- **Google API key** for Gemini models
- **Google Chrome** (or Chromium-based browser) with extension developer mode

### Python dependencies

It is recommended to use a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

If you don't have a `requirements.txt`, install at least:

```bash
pip install flask python-dotenv google-generativeai google-adk
```

---

## 2. Environment variables

Create a `.env` file in the project root (same folder as `time_guardian_agent.py`) and set:

```env
GOOGLE_API_KEY=your_gemini_api_key_here
TIMEGUARD_MODEL_NAME=gemini-2.0-flash          # optional, default is used if absent
TIMEGUARD_HISTORY_MODEL_NAME=gemini-2.0-flash  # optional, for history pipeline
```

Make sure the key has access to Gemini models.

---

## 3. Running the backend locally

From the project root (where `time_guardian_agent.py` lives):

```bash
# Activate your venv if not already
# On Windows PowerShell:
#   .venv\Scripts\Activate.ps1
# On cmd:
#   .venv\Scripts\activate

python time_guardian_agent.py
```

You should see output similar to:

- `Starting Time Guardian Agent...`
- Flask serving on `http://127.0.0.1:5000`

Endpoints used by the extension:

- `POST /check_url` – classify & possibly block a URL
- `GET /blocked_sites` – list currently blocked URLs
- `GET /history_report` – run the sequential history summarizer + coach and return a plain-text report

You can also open `http://localhost:5000/ui` in a browser to manually test URL checks.

---

## 4. Chrome extension setup

The extension lives in the `extension/` folder.

### 4.1 Load the extension in Chrome

1. Open `chrome://extensions` in Chrome.
2. Enable **Developer mode** (toggle in the top-right).
3. Click **Load unpacked**.
4. Select the `extension` directory in this project.

The Time Guardian extension should now appear in your extensions list.

> **Important:** The backend must be running on `http://localhost:5000` before the extension can function.

---

## 5. Testing the flow end-to-end

1. **Start the backend**
   - Run `python time_guardian_agent.py` from the project root.

2. **Load the extension**
   - Via `chrome://extensions` → Developer mode → Load unpacked → select `extension/`.

3. **Trigger a block**
   - Navigate to a known time-wasting site (e.g., `https://www.youtube.com/`, `https://www.reddit.com/`).
   - The extension should:
     - Call `POST /check_url`.
     - If blocked, redirect to `blocked.html` with query parameters describing the URL, reason, and unblock time.

4. **View the AI reason and unblock time**
   - On the blocked page, confirm:
     - The blocked URL is displayed.
     - The AI reason is shown.
     - An approximate unblock time is shown.

5. **View the productivity report**
   - On the blocked page, click **"View Productivity Report"**.
   - This navigates to `report.html` inside the extension.
   - `report.js` calls `GET /history_report` to fetch:
     - A plain-text `report` summarizing your browsing patterns.
     - The raw `events` array (used mainly for debugging/analysis).

If everything is wired correctly, you should see a short, plain-text productivity report without markdown formatting, possibly mentioning peak distraction times and focus suggestions.

---

## 6. Development notes

- The backend uses a small ADK-based agent module in `timeguard-adk/agent/__main__.py` for:
  - URL classification via an LlmAgent + Runner.
  - History reporting via a SequentialAgent that chains a summarizer agent and a coaching agent.
- Browsing history is stored in a SQLite database (`timeguard_sessions.db`) managed by the Google ADK session service.
- The history coach agent has access to a `unix_to_local_time` helper to turn UNIX timestamps into human-readable local times.
