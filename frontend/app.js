const API = 'http://localhost:8000/api';
let sessionId = crypto.randomUUID();
let currentMode = 'adaptive';
let mentorColors = {};
let isLoading = false;

// ── Init ──────────────────────────────────────────────────────────────
async function init() {
  try {
    const res = await fetch(`${API}/mentors`);
    const mentors = await res.json();
    mentorColors = Object.fromEntries(
      Object.entries(mentors).map(([id, m]) => [id, m.color])
    );
    renderWeights(
      Object.fromEntries(Object.keys(mentors).map(id => [id, 1 / Object.keys(mentors).length])),
      Object.fromEntries(Object.entries(mentors).map(([id, m]) => [id, m.display_name]))
    );
  } catch (e) {
    console.error('Failed to load mentors:', e);
  }
}

// ── Mode toggle ───────────────────────────────────────────────────────
document.querySelectorAll('.mode-btn').forEach(btn => {
  btn.addEventListener('click', () => {
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

  input.value = '';
  appendUserMessage(text);
  showTyping();
  setLoading(true);

  try {
    const res = await fetch(`${API}/chat`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ session_id: sessionId, message: text, mode: currentMode }),
    });

    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();

    removeTyping();
    renderWeights(data.mentor_weights, data.mentor_names);
    renderTopics(data.detected_topics || []);

    if (currentMode === 'council') {
      appendCouncilResponse(data.council_responses || []);
    } else {
      appendAssistantMessage(data.response, data.mentor_weights, data.mentor_names);
    }
  } catch (err) {
    removeTyping();
    appendError(err.message);
  } finally {
    setLoading(false);
  }
}

// ── Weight display ────────────────────────────────────────────────────
function renderWeights(weights, names) {
  const container = document.getElementById('weight-display');
  const sorted = Object.entries(weights).sort((a, b) => b[1] - a[1]);

  container.innerHTML = sorted.map(([id, w]) => {
    const pct = Math.round(w * 100);
    const color = mentorColors[id] || '#888';
    const name = (names && names[id]) || id;
    return `
      <div class="mentor-row">
        <div class="mentor-row-header">
          <span class="mentor-name">${name}</span>
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
  const container = document.getElementById('topic-tags');
  container.innerHTML = topics.map(t =>
    `<span class="topic-tag">${t}</span>`
  ).join('');
}

// ── Chat rendering ────────────────────────────────────────────────────
function appendUserMessage(text) {
  clearWelcome();
  const win = document.getElementById('chat-window');
  const el = document.createElement('div');
  el.className = 'msg user';
  el.innerHTML = `<div class="msg-bubble">${escHtml(text)}</div>`;
  win.appendChild(el);
  scrollBottom();
}

function appendAssistantMessage(text, weights, names) {
  const win = document.getElementById('chat-window');
  const el = document.createElement('div');
  el.className = 'msg assistant';

  const topMentor = topMentorFrom(weights, names);
  el.innerHTML = `
    <div class="msg-label">Wisdom Router</div>
    <div class="msg-bubble">${escHtml(text)}</div>
    <div class="perspective-shift">
      <span class="perspective-dot" style="background:${topMentor.color}"></span>
      ${topMentor.name} perspective dominant
    </div>
  `;
  win.appendChild(el);
  scrollBottom();
}

function appendCouncilResponse(members) {
  const win = document.getElementById('chat-window');
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
    setTimeout(() => card.classList.add('visible'), i * 250);
  });

  win.appendChild(board);
  scrollBottom();
}

// ── Typing indicator ──────────────────────────────────────────────────
function showTyping() {
  const win = document.getElementById('chat-window');
  const el = document.createElement('div');
  el.id = 'typing-indicator';
  el.className = 'msg assistant';
  el.innerHTML = `
    <div class="msg-bubble">
      <div class="typing-dots">
        <span></span><span></span><span></span>
      </div>
    </div>
  `;
  win.appendChild(el);
  scrollBottom();
}

function removeTyping() {
  const el = document.getElementById('typing-indicator');
  if (el) el.remove();
}

// ── Error ─────────────────────────────────────────────────────────────
function appendError(msg) {
  const win = document.getElementById('chat-window');
  const el = document.createElement('div');
  el.className = 'msg assistant';
  el.innerHTML = `<div class="msg-bubble" style="border-color:#E74C3C;color:#E74C3C">Error: ${escHtml(msg)}</div>`;
  win.appendChild(el);
  scrollBottom();
}

// ── Reset ─────────────────────────────────────────────────────────────
document.getElementById('btn-reset').addEventListener('click', async () => {
  await fetch(`${API}/session/${sessionId}`, { method: 'DELETE' });
  sessionId = crypto.randomUUID();
  const win = document.getElementById('chat-window');
  win.innerHTML = `
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
}

function topMentorFrom(weights, names) {
  if (!weights) return { name: 'Advisor', color: '#888' };
  const top = Object.entries(weights).sort((a, b) => b[1] - a[1])[0];
  return {
    name: (names && names[top[0]]) || top[0],
    color: mentorColors[top[0]] || '#888',
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
