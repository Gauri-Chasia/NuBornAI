const chatScroll = document.getElementById("chatScroll");
const input = document.getElementById("composerInput");
const sendBtn = document.getElementById("sendBtn");
const resetBtn = document.getElementById("resetBtn");
const sourceList = document.getElementById("sourceList");
const modelValue = document.getElementById("modelValue");
const rerankerLine = document.getElementById("rerankerLine");
const rerankerValue = document.getElementById("rerankerValue");

let ready = false;

function escapeHtml(str) {
  const div = document.createElement("div");
  div.textContent = str;
  return div.innerHTML;
}

function scrollToBottom() {
  chatScroll.scrollTop = chatScroll.scrollHeight;
}

function tagIconSvg() {
  return `<svg class="tag-icon" width="10" height="10" viewBox="0 0 12 12" fill="none">
    <path d="M1.5 6.2V2a.5.5 0 0 1 .5-.5h4.2a.5.5 0 0 1 .35.15l4.3 4.3a.5.5 0 0 1 0 .7L6.65 10.85a.5.5 0 0 1-.7 0l-4.3-4.3a.5.5 0 0 1-.15-.35Z" stroke="currentColor" stroke-width="1"/>
    <circle cx="4" cy="4" r=".6" fill="currentColor"/>
  </svg>`;
}

function addUserMessage(text) {
  const wrap = document.createElement("div");
  wrap.className = "msg msg-user";
  wrap.innerHTML = `<div class="msg-bubble">${escapeHtml(text)}</div>`;
  chatScroll.appendChild(wrap);
  scrollToBottom();
}

function addThinkingMessage() {
  const wrap = document.createElement("div");
  wrap.className = "msg msg-bot thinking";
  wrap.id = "thinkingMsg";
  wrap.innerHTML = `<div class="msg-bubble">Thinking<span class="thinking-dots"><span>.</span><span>.</span><span>.</span></span></div>`;
  chatScroll.appendChild(wrap);
  scrollToBottom();
}

function removeThinkingMessage() {
  const el = document.getElementById("thinkingMsg");
  if (el) el.remove();
}

function addSetupMessage() {
  const wrap = document.createElement("div");
  wrap.className = "msg msg-bot thinking";
  wrap.id = "setupMsg";
  wrap.innerHTML = `<div class="msg-bubble">Setting up your documents<span class="thinking-dots"><span>.</span><span>.</span><span>.</span></span></div>`;
  chatScroll.appendChild(wrap);
  scrollToBottom();
}

function removeSetupMessage() {
  const el = document.getElementById("setupMsg");
  if (el) el.remove();
}

function addBotMessage(answer, sources, isError) {
  const group = document.createElement("div");
  group.className = "msg-group msg-bot";

  const bubbleWrap = document.createElement("div");
  bubbleWrap.className = "msg msg-bot" + (isError ? " error-bubble" : "");
  bubbleWrap.innerHTML = `<div class="msg-bubble">${escapeHtml(answer)}</div>`;
  group.appendChild(bubbleWrap);

  if (sources && sources.length) {
    const chipsWrap = document.createElement("div");
    chipsWrap.className = "citations";
    chipsWrap.innerHTML = sources
      .map(s => `<span class="citation-chip">${tagIconSvg()}${escapeHtml(s.name)}</span>`)
      .join("");
    group.appendChild(chipsWrap);
  }

  chatScroll.appendChild(group);
  scrollToBottom();
}

async function refreshStatus() {
  try {
    const res = await fetch("/api/status");
    const data = await res.json();

    if (data.error) {
      removeSetupMessage();
      addBotMessage(`Setup error: ${data.error}`, null, true);
      return true; // stop polling
    }

    if (data.ready) {
      ready = true;
      removeSetupMessage();
      input.disabled = false;
      sendBtn.disabled = false;
      modelValue.textContent = data.ollama_model || "—";
      if (data.reranker_model) {
        rerankerLine.style.display = "flex";
        rerankerValue.textContent = data.reranker_model.split("/").pop();
      }
      renderSources(data.sources);
      return true; // stop polling
    } else {
      return false; // still building — keep polling
    }
  } catch (e) {
    removeSetupMessage();
    addBotMessage("Couldn't reach the server. Is app.py still running?", null, true);
    return true; // stop polling
  }
}

function renderSources(sources) {
  if (!sources || !sources.length) {
    sourceList.innerHTML = `<li class="source-empty">No documents indexed yet — add file paths or URLs to <code>SOURCES</code> in <code>config.py</code>.</li>`;
    return;
  }
  sourceList.innerHTML = sources
    .map(s => `<li><span><span class="src-name">${escapeHtml(s.name)}</span><span class="src-count">${s.chunks} chunks</span></span></li>`)
    .join("");
}

async function pollUntilReady() {
  const done = await refreshStatus();
  if (!done) {
    setTimeout(pollUntilReady, 1500);
  }
}

async function sendMessage() {
  const text = input.value.trim();
  if (!text || !ready) return;

  input.value = "";
  autoResize();
  sendBtn.disabled = true;

  addUserMessage(text);
  addThinkingMessage();

  try {
    const res = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message: text }),
    });
    const data = await res.json();
    removeThinkingMessage();

    if (data.error) {
      addBotMessage(data.error, null, true);
    } else {
      addBotMessage(data.answer, data.sources);
    }
  } catch (e) {
    removeThinkingMessage();
    addBotMessage("Couldn't reach the server. Is app.py still running?", null, true);
  } finally {
    sendBtn.disabled = false;
    input.focus();
  }
}

async function resetConversation() {
  await fetch("/api/reset", { method: "POST" });
  chatScroll.innerHTML = "";
  const wrap = document.createElement("div");
  wrap.className = "msg msg-bot";
  wrap.innerHTML = `<div class="msg-bubble">Conversation reset. What would you like to know?</div>`;
  chatScroll.appendChild(wrap);
}

function autoResize() {
  input.style.height = "auto";
  input.style.height = Math.min(input.scrollHeight, 140) + "px";
}

input.addEventListener("input", autoResize);
input.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    sendMessage();
  }
});
sendBtn.addEventListener("click", sendMessage);
resetBtn.addEventListener("click", resetConversation);

input.disabled = true;
sendBtn.disabled = true;
addSetupMessage();
pollUntilReady();