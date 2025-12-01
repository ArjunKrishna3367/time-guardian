// blocked.js - logic for the Time Guardian blocked page

// Get URL parameters
const urlParams = new URLSearchParams(window.location.search);
const blockedUrl = urlParams.get('url') || 'this website';
const reason = urlParams.get('reason') || 'Time-wasting site';
const unblockTime = urlParams.get('unblock_time');

// Update the page with the blocked site info
const blockedUrlEl = document.getElementById('blockedUrl');
const blockReasonEl = document.getElementById('blockReason');
const unblockTimeEl = document.getElementById('unblockTime');

if (blockedUrlEl) {
  blockedUrlEl.textContent = blockedUrl;
}
if (blockReasonEl) {
  blockReasonEl.textContent = `AI's reason: ${reason}`;
}
if (unblockTimeEl) {
  if (unblockTime) {
    const unblockDate = new Date(unblockTime);
    unblockTimeEl.textContent = `This site will be unblocked at ${unblockDate.toLocaleTimeString()}`;
  } else {
    unblockTimeEl.textContent = 'This site is permanently blocked.';
  }
}

// Handle Go Back button
const goBackBtn = document.getElementById('goBackBtn');
if (goBackBtn) {
  goBackBtn.addEventListener('click', () => {
    if (window.history.length > 1) {
      window.history.back();
    } else {
      window.close();
    }
  });
}

// Handle whitelist button
const whitelistBtn = document.getElementById('whitelistBtn');
if (whitelistBtn) {
  whitelistBtn.addEventListener('click', () => {
    let domain = '';
    try {
      const url = new URL(blockedUrl.startsWith('http') ? blockedUrl : `https://${blockedUrl}`);
      domain = url.hostname;
    } catch (e) {
      console.error('Error parsing URL:', e);
      domain = blockedUrl;
    }

    chrome.runtime.sendMessage(
      {
        action: 'whitelistSite',
        site: domain,
      },
      (response) => {
        if (response && !response.error) {
          alert(`Successfully whitelisted ${domain}. You can now visit the site.`);
          window.location.href = blockedUrl;
        } else {
          alert(`Failed to whitelist site: ${response?.error || 'Unknown error'}`);
        }
      },
    );
  });
}

// Handle "View Productivity Report" button
const viewReportBtn = document.getElementById('viewReportBtn');
if (viewReportBtn && chrome && chrome.runtime && chrome.runtime.getURL) {
  viewReportBtn.addEventListener('click', () => {
    const reportUrl = chrome.runtime.getURL('report.html');
    window.location.href = reportUrl;
  });
}
