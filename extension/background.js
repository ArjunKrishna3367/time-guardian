// Configuration
const API_URL = 'http://localhost:5000';

// Check if URL should be blocked
async function checkUrl(url) {
  try {
    const response = await fetch(`${API_URL}/check_url`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ url })
    });
    
    const data = await response.json();
    return data;
  } catch (error) {
    console.error('Error checking URL:', error);
    return { blocked: false, error: error.message };
  }
}

// Block navigation to time-wasting sites
chrome.webNavigation.onBeforeNavigate.addListener(async (details) => {
  // Skip if the navigation is in a subframe
  if (details.frameId !== 0) return;
  
  const result = await checkUrl(details.url);
  
  if (result.blocked) {
    // Cancel the navigation
    chrome.tabs.update(details.tabId, {
      url: chrome.runtime.getURL('blocked.html') + 
           `?url=${encodeURIComponent(details.url)}` +
           `&reason=${encodeURIComponent(result.reason || 'Time-wasting site')}` +
           `&unblock_time=${encodeURIComponent(result.unblock_time || '')}`
    });
  }
}, { url: [{ urlMatches: 'http://*/*' }, { urlMatches: 'https://*/*' }] });

// Listen for messages from popup
chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
  if (request.action === 'getBlockedSites') {
    fetch(`${API_URL}/blocked_sites`)
      .then(response => response.json())
      .then(data => sendResponse(data))
      .catch(error => sendResponse({ error: error.message }));
    return true; // Will respond asynchronously
  }
  
  if (request.action === 'whitelistSite') {
    fetch(`${API_URL}/whitelist`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ site: request.site })
    })
      .then(response => response.json())
      .then(data => sendResponse(data))
      .catch(error => sendResponse({ error: error.message }));
    return true; // Will respond asynchronously
  }
});
