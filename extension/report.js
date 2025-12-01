// report.js - logic for the Time Guardian productivity report page

const API_URL = 'http://localhost:5000';

const statusEl = document.getElementById('status');
const reportEl = document.getElementById('report');
const backLink = document.getElementById('backLink');

async function loadReport() {
  try {
    const resp = await fetch(`${API_URL}/history_report`);
    if (!resp.ok) {
      statusEl.textContent = `Error loading report: HTTP ${resp.status}`;
      statusEl.classList.add('error');
      return;
    }

    const data = await resp.json();
    const text = data.report || 'No browsing history has been recorded yet.';

    statusEl.textContent = 'Report loaded.';
    reportEl.textContent = text;
  } catch (err) {
    statusEl.textContent = 'Error loading report: ' + err;
    statusEl.classList.add('error');
  }
}

if (backLink) {
  backLink.addEventListener('click', (e) => {
    e.preventDefault();
    if (window.history.length > 1) {
      window.history.back();
    } else {
      window.close();
    }
  });
}

loadReport();
