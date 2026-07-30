// AI Wisdom Router — frontend.
// The page is served by the same FastAPI app that exposes the API, so the API
// path is relative: it works unchanged on a different host or port.

const API = '/api';
const SESSION_KEY = 'wisdom-router-session';

let sessionId = loadSessionId();
let currentMode = 'adaptive';
let mentorColors = {};
let mentorNames = {};
let isLoading = false;

// ── Session identity ──────────────────────────────────────────────────
// Kept in localStorage so a page reload continues the conversation rather
// than silently starting a new one with reset mentor weights.
function loadSessionId() {
  try {
    const existing = localStorage.getItem(SESSION_KEY);
    if (existing) return existing;
    const created = newId();
    localStorage.setItem(SESSION_KEY, created);
    return created;
  } catch (e) {
    return newId();
  }
}

function newId() {
  if (window.crypto && crypto.randomUUID) return crypto.randomUUID();
  return 'sess-' + Math.random().toString(36).slice(2) + Date.now().toString(36);
}

function rotateSessionId() {
  sessionId = newId();
  try {
    localStorage.setItem(SESSION_KEY, sessionId);
  } catch (e) {
    /* private browsing — an in-memory id still works for this tab */
  }
}

// ── Init ──────────────────────────────────────────────────────────────
async function init() {
  try {
    const res = await fetch(`${API}/mentors`);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const mentors = await res.json();

    mentorColors = {};
    mentorNames = {};
    Object.entries(mentors).forEach(([id, m]) => {
      mentorColors[id] = m.color;
      mentorNames[id] = m.display_name;
    });

    // Show the stored distribution for this session, not a fresh even split.
    const weights = await loadWeights();
    renderWeights(weights, mentorNames);
  } catch (e) {
    showBanner('Could not reach the server. Is it running? (python -m backend.main)');
  }
  checkHealth();
}

async function loadWeights() {
  try {
    const res = await fetch(`${API}/session/${encodeURIComponent(sessionId)}/weights`);
    if (res.ok) return (await res.json()).mentor_weights;
  } catch (e) {
    /* fall through to an even split */
  }
  const ids = Object.keys(mentorNames);
  return Object.fromEntries(ids.map(id => [id, 1 / (ids.length || 1)]));
}

// Surfaces a misconfigured backend up front instead of on the first message.
async function checkHealth() {
  try {
    const res = await fetch(`${API}/health`);
    if (!res.ok) return;
    const health = await res.json();
    if (health.status === 'ok') { hideBanner(); return; }

    const ollama = health.ollama || {};
    const store = health.vector_store || {};
    if (!ollama.reachable) {
      showBanner('Ollama is not reachable — start it with "ollama serve".');
    } else if (!ollama.llm_model_available) {
      showBanner(`Model "${ollama.llm_model}" is not installed — run: ollama pull ${ollama.llm_model}`);
    } else if (!ollama.embedding_model_available) {
      showBanner(`Embedding model "${ollama.embedding_model}" is missing — run: ollama pull ${ollama.embedding_model}`);
    } else if (!store.reachable || !store.points) {
      showBanner('Mentor knowledge is not loaded — run: python -m backend.vector_store.seeder');
    }
  } catch (e) {
    /* health is advisory; never block the UI on it */
  }
}

function showBanner(text) {
  let banner = document.getElementById('status-banner');
  if (!banner) {
    banner = document.createElement('div');
    banner.id = 'status-banner';
    banner.className = 'status-banner';
    document.querySelector('.main').prepend(banner);
  }
  banner.textContent = text;
  banner.style.display = 'block';
}

function hideBanner() {
  const banner = document.getElementById('status-banner');
  if (banner) banner.style.display = 'none';
}

// ── Mode toggle ───────────────────────────────────────────────────────
document.querySelectorAll('.mode-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    if (isLoading) return;
    document.querySelectorAll('.mode-btn').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    currentMode = btn.dataset.mode;
  });
});

// ── Send message ──────────────────────────────────────────────────────
document.getElementById('btn-send').addEventListener('click', sendMessage);
document.getElementById('user-input').addEventListener('keydown', e => {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    sendMessage();
  }
});

async function sendMessage() {
  if (isLoading) return;
  const input = document.getElementById('user-input');
  const text = input.value.trim();
  if (!text) return;

  const mode = currentMode; // pin the mode for this turn
  input.value = '';
  appendUserMessage(text);
  showTyping(mode);
  setLoading(true);

  try {
    const res = await fetch(`${API}/chat`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ session_id: sessionId, message: text, mode }),
    });

    if (!res.ok) throw new Error(await errorDetail(res));
    const data = await res.json();

    removeTyping();
    hideBanner();
    renderWeights(data.mentor_weights, data.mentor_names);
    renderTopics(data.detected_topics || []);

    if (data.mode === 'council') {
      appendCouncilResponse(data.council_responses || []);
    } else {
      appendAssistantMessage(data.response, data.mentor_weights, data.mentor_names);
    }
  } catch (err) {
    removeTyping();
    appendError(err.message);
  } finally {
    setLoading(false);
    input.focus();
  }
}

// FastAPI puts the useful text in `detail`; a bare status code helps nobody.
async function errorDetail(res) {
  try {
    const body = await res.json();
    if (typeof body.detail === 'string') return body.detail;
    if (Array.isArray(body.detail) && body.detail.length) {
      return body.detail.map(d => d.msg || JSON.stringify(d)).join('; ');
    }
  } catch (e) {
    /* not JSON */
  }
  return res.status === 503
    ? 'The model backend is unavailable. Check that Ollama is running.'
    : `Request failed (HTTP ${res.status}).`;
}

// ── Weight display ────────────────────────────────────────────────────
function renderWeights(weights, names) {
  const container = document.getElementById('weight-display');
  const sorted = Object.entries(weights || {}).sort((a, b) => b[1] - a[1]);

  container.innerHTML = sorted.map(([id, w]) => {
    const pct = Math.round(w * 100);
    const color = mentorColors[id] || '#888';
    const name = (names && names[id]) || mentorNames[id] || id;
    return `
      <div class="mentor-row">
        <div class="mentor-row-header">
          <span class="mentor-name">${escHtml(name)}</span>
          <span class="mentor-pct">${pct}%</span>
        </div>
        <div class="weight-track">
          <div class="weight-fill" style="width:${pct}%;background:${color}"></div>
        </div>
      </div>
    `;
  }).join('');
}

// ── Topics ────────────────────────────────────────────────────────────
function renderTopics(topics) {
  document.getElementById('topic-tags').innerHTML = topics
    .map(t => `<span class="topic-tag">${escHtml(t)}</span>`)
    .join('');
}

// ── Chat rendering ────────────────────────────────────────────────────
function appendUserMessage(text) {
  clearWelcome();
  const el = document.createElement('div');
  el.className = 'msg user';
  el.innerHTML = `<div class="msg-bubble">${escHtml(text)}</div>`;
  document.getElementById('chat-window').appendChild(el);
  scrollBottom();
}

function appendAssistantMessage(text, weights, names) {
  const el = document.createElement('div');
  el.className = 'msg assistant';

  const top = topMentorFrom(weights, names);
  el.innerHTML = `
    <div class="msg-label">Wisdom Router</div>
    <div class="msg-bubble">${escHtml(text)}</div>
    <div class="perspective-shift">
      <span class="perspective-dot" style="background:${top.color}"></span>
      ${escHtml(top.name)} perspective dominant
    </div>
  `;
  document.getElementById('chat-window').appendChild(el);
  scrollBottom();
}

function appendCouncilResponse(members) {
  const board = document.createElement('div');
  board.className = 'council-board';
  board.innerHTML = `<div class="council-label">Council response</div>`;

  members.forEach((m, i) => {
    const card = document.createElement('div');
    card.className = 'council-card';
    card.innerHTML = `
      <div class="council-mentor-name">
        <span class="mentor-badge" style="background:${m.color}"></span>
        ${escHtml(m.mentor_name)}
      </div>
      <p>${escHtml(m.response)}</p>
    `;
    board.appendChild(card);
    // Stagger the reveal so the council reads as arriving, not as a wall of text.
    setTimeout(() => card.classList.add('visible'), i * 250);
  });

  document.getElementById('chat-window').appendChild(board);
  scrollBottom();
}

// ── Typing indicator ──────────────────────────────────────────────────
function showTyping(mode) {
  const el = document.createElement('div');
  el.id = 'typing-indicator';
  el.className = 'msg assistant';
  const label = mode === 'council' ? 'The council is deliberating' : 'Thinking';
  el.innerHTML = `
    <div class="msg-label">${label}</div>
    <div class="msg-bubble">
      <div class="typing-dots"><span></span><span></span><span></span></div>
    </div>
  `;
  document.getElementById('chat-window').appendChild(el);
  scrollBottom();
}

function removeTyping() {
  const el = document.getElementById('typing-indicator');
  if (el) el.remove();
}

// ── Error ─────────────────────────────────────────────────────────────
function appendError(msg) {
  const el = document.createElement('div');
  el.className = 'msg assistant';
  el.innerHTML = `<div class="msg-bubble msg-error">${escHtml(msg)}</div>`;
  document.getElementById('chat-window').appendChild(el);
  scrollBottom();
}

// ── Reset ─────────────────────────────────────────────────────────────
document.getElementById('btn-reset').addEventListener('click', async () => {
  try {
    await fetch(`${API}/session/${encodeURIComponent(sessionId)}`, { method: 'DELETE' });
  } catch (e) {
    /* the server may already have forgotten it */
  }
  rotateSessionId();
  document.getElementById('chat-window').innerHTML = `
    <div class="welcome">
      <p>Ask anything. The right mind will answer.</p>
      <p class="welcome-sub">Switch to <strong>Council Mode</strong> to hear multiple perspectives at once.</p>
    </div>
  `;
  document.getElementById('topic-tags').innerHTML = '';
  init();
});

// ── Helpers ───────────────────────────────────────────────────────────
function clearWelcome() {
  const w = document.querySelector('.welcome');
  if (w) w.remove();
}

function scrollBottom() {
  const win = document.getElementById('chat-window');
  requestAnimationFrame(() => { win.scrollTop = win.scrollHeight; });
}

function setLoading(state) {
  isLoading = state;
  document.getElementById('btn-send').disabled = state;
  document.getElementById('user-input').disabled = state;
  document.querySelectorAll('.mode-btn').forEach(b => { b.disabled = state; });
}

function topMentorFrom(weights, names) {
  const entries = Object.entries(weights || {});
  if (!entries.length) return { name: 'Advisor', color: '#888' };
  const [id] = entries.sort((a, b) => b[1] - a[1])[0];
  return {
    name: (names && names[id]) || mentorNames[id] || id,
    color: mentorColors[id] || '#888',
  };
}

function escHtml(str) {
  return String(str)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/\n/g, '<br>');
}

// ── Start ─────────────────────────────────────────────────────────────
init();
