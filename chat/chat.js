const $ = (selector) => document.querySelector(selector);
const storageKey = "agathe-chat-v1";
const input = $("#message");
const transcript = $("#transcript");
const empty = transcript.firstElementChild;
const sidebar = $("#history");
const narrow = matchMedia("(max-width: 760px)");
const metricsPanel = $("#metrics");
const medium = matchMedia("(max-width: 1099px)");
const format = new Intl.NumberFormat("pt-BR", { maximumFractionDigits: 2 });
let model;
let busy = false;

function readHistory() {
  try {
    const saved = JSON.parse(localStorage.getItem(storageKey));
    if (Array.isArray(saved?.threads)) return saved;
  } catch { /* A browser can disable local storage. */ }
  return { threads: [], activeId: "" };
}

let history = readHistory();
const active = () => history.threads.find((chat) => chat.id === history.activeId);

function save() {
  try {
    localStorage.setItem(storageKey, JSON.stringify(history));
  } catch {
    $("#status").textContent = "Não foi possível salvar o histórico neste navegador.";
  }
}

function controls() {
  $("#new-chat").disabled = busy;
  $("#delete-chat").disabled = busy || !active();
  $("button[type=submit]").disabled = busy || !model || !input.value.trim();
  input.readOnly = busy;
  for (const button of document.querySelectorAll("nav button")) button.disabled = busy;
}

function showMessage(role, text) {
  const article = document.createElement("article");
  article.className = "message " + role;
  article.setAttribute("aria-label", role === "user" ? "Você" : "Agathe");
  article.textContent = text.trimStart();
  transcript.append(article);
  return article;
}

function render() {
  const chat = active();
  $("#chat-title").textContent = chat?.title || "Nova conversa";
  $("#delete-chat").hidden = !chat;
  $("nav").replaceChildren(...history.threads.map((item) => {
    const button = document.createElement("button");
    button.textContent = item.title;
    button.title = item.title;
    button.setAttribute("aria-current", String(item === chat));
    button.onclick = () => select(item.id);
    return button;
  }));
  transcript.replaceChildren();
  if (!chat?.messages.length) transcript.append(empty);
  for (const message of chat?.messages || []) {
    showMessage(message.role, message.text || "Fim do texto.");
  }
  const metrics = chat?.metrics;
  $("#elapsed").textContent = metrics ? format.format(metrics.seconds) + " s" : "—";
  $("#count").textContent = metrics ? format.format(metrics.characters) : "—";
  $("#rate").textContent = metrics?.seconds
    ? format.format(metrics.characters / metrics.seconds) : "—";
  $("#context").textContent = metrics ? metrics.used + " de " + metrics.capacity : "—";
  transcript.scrollTop = transcript.scrollHeight;
  controls();
}

function select(id) {
  if (sidebar.matches(":popover-open")) sidebar.hidePopover();
  history.activeId = id;
  input.value = "";
  $("#status").textContent = "";
  save();
  render();
  input.focus();
}

async function generate(prefix, append) {
  const response = await fetch("/api/continue", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ prefix }),
  });
  if (!response.ok) throw new Error("Falha na geração.");
  let pending = "";
  const reader = response.body.pipeThrough(new TextDecoderStream()).getReader();
  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      const lines = (pending + value).split("\n");
      pending = lines.pop();
      for (const line of lines) {
        const event = JSON.parse(line);
        if (event.metrics) return event.metrics;
        if (typeof event.text !== "string") throw new Error("Resposta inválida.");
        append(event.text);
      }
    }
    throw new Error("Geração interrompida.");
  } finally {
    reader.releaseLock();
  }
}

$("form").onsubmit = async (event) => {
  event.preventDefault();
  const text = input.value;
  if (!text.trim() || !model || busy) return;
  busy = true;
  const previous = active();
  const chat = previous || {
    id: crypto.randomUUID(), title: text.trim().slice(0, 42), messages: [],
  };
  if (!previous) history.threads.unshift(chat);
  history.activeId = chat.id;
  chat.messages.push({ role: "user", text });
  input.value = "";
  $("#status").textContent = "";
  save();
  render();
  const body = showMessage("assistant", "Continuando…");
  let answer = "";
  try {
    const full = chat.messages.map((message) => message.text).join("\n");
    // The server keeps the last `context` tokens; a token rarely spans more than 8 characters.
    const prefix = Array.from(full).slice(-model.context * 8).join("");
    chat.metrics = await generate(prefix, (piece) => {
      answer += piece;
      body.textContent = answer.trimStart();
      transcript.scrollTop = transcript.scrollHeight;
    });
    chat.messages.push({ role: "assistant", text: answer });
  } catch {
    chat.messages.pop();
    if (!previous) {
      history.threads = history.threads.filter((item) => item !== chat);
      history.activeId = "";
    }
    input.value = text;
    $("#status").textContent = "A geração foi interrompida. Envie novamente para tentar.";
  } finally {
    busy = false;
    save();
    render();
    input.focus();
  }
};

$("#new-chat").onclick = () => select("");
$("#delete-chat").onclick = () => {
  history.threads = history.threads.filter((chat) => chat.id !== history.activeId);
  select(history.threads[0]?.id || "");
};
input.oninput = controls;
input.onkeydown = (event) => {
  if (event.key === "Enter" && !event.shiftKey && !event.isComposing) {
    event.preventDefault();
    $("form").requestSubmit();
  }
};

function sizePanels() {
  for (const [panel, query] of [[sidebar, narrow], [metricsPanel, medium]]) {
    if (query.matches) panel.setAttribute("popover", "auto");
    else panel.removeAttribute("popover");
  }
}
sizePanels();
narrow.addEventListener("change", sizePanels);
medium.addEventListener("change", sizePanels);
render();
try {
  const response = await fetch("/api/model");
  if (!response.ok) throw new Error();
  model = await response.json();
  $("#checkpoint").textContent = model.name;
  $("#parameters").textContent = format.format(model.parameters);
  controls();
} catch {
  $("#status").textContent =
    "Não foi possível carregar o modelo. Recarregue a página para tentar.";
}
