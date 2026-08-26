/* AgriGPT frontend logic
   - JWT stored in localStorage, sent as Bearer header on every /api call
   - Text chat, image upload (multipart), voice input (Web Speech API),
     voice output (browser SpeechSynthesis)
*/
const $ = (id) => document.getElementById(id);
const VOICE_LANG = { en:"en-IN", hi:"hi-IN", ta:"ta-IN", te:"te-IN", ml:"ml-IN", kn:"kn-IN", bn:"bn-IN", pa:"pa-IN" };

let token = localStorage.getItem("agrigpt_token");
let currentUser = null;
let pendingImage = null;
let recognition = null;
let listening = false;

/* ---------------- screens ---------------- */
function showAuth() {
  $("auth-screen").style.display = "flex";
  $("chat-screen").style.display = "none";
}
function showChat() {
  $("auth-screen").style.display = "none";
  $("chat-screen").style.display = "flex";
}
function switchTab(mode) {
  const isSignup = mode === "signup";
  $("tab-login").classList.toggle("active", !isSignup);
  $("tab-signup").classList.toggle("active", isSignup);
  $("signup-fields").style.display = isSignup ? "flex" : "none";
  $("auth-submit").textContent = isSignup ? "Create account" : "Login";
}
function logout() {
  localStorage.removeItem("agrigpt_token");
  token = null;
  currentUser = null;
  $("chat").innerHTML = "";
  showAuth();
}

/* ---------------- API helper ---------------- */
async function api(path, options = {}) {
  options.headers = options.headers || {};
  if (token) options.headers["Authorization"] = "Bearer " + token;
  const res = await fetch(path, options);
  if (res.status === 401) { logout(); throw new Error("Please login again"); }
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.detail || "Error " + res.status);
  }
  return res.json();
}

/* ---------------- auth ---------------- */
$("auth-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  $("auth-error").textContent = "";
  const isSignup = $("tab-signup").classList.contains("active");
  const payload = { email: $("email").value.trim(), password: $("password").value };
  if (isSignup) {
    payload.language = $("language").value;
  }
  const btn = $("auth-submit");
  btn.disabled = true;
  try {
    const data = await api(isSignup ? "/api/signup" : "/api/login", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    token = data.access_token;
    localStorage.setItem("agrigpt_token", token);
    currentUser = data.user;
    enterChat();
  } catch (err) {
    $("auth-error").textContent = err.message;
  } finally {
    btn.disabled = false;
  }
});

function enterChat() {
  $("user-label").textContent = currentUser.email;
  $("lang-switch").value = currentUser.language || "en";
  showChat();
  loadHistory();
}

$("logout-btn").onclick = logout;

$("lang-switch").onchange = async (e) => {
  if (!currentUser) return;
  try {
    currentUser = await api("/api/profile", {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ language: e.target.value }),
    });
  } catch (err) { console.warn(err); }
};

/* ---------------- chat UI ---------------- */
function addBubble(role, text, meta) {
  const div = document.createElement("div");
  div.className = "bubble " + (role === "user" ? "user" : "bot");
  if (meta && meta.disease) {
    const chip = document.createElement("div");
    chip.className = "disease-chip";
    chip.textContent = `🔬 ${meta.disease} — ${meta.confidence}%`;
    div.appendChild(chip);
  }
  const span = document.createElement("span");
  span.textContent = text;
  div.appendChild(span);
  $("chat").appendChild(div);
  $("chat").scrollTop = $("chat").scrollHeight;
  return div;
}

function typingIndicator() {
  const div = document.createElement("div");
  div.className = "bubble bot typing";
  div.id = "typing";
  div.textContent = "…";
  $("chat").appendChild(div);
  $("chat").scrollTop = $("chat").scrollHeight;
}
function removeTyping() {
  const t = $("typing");
  if (t) t.remove();
}

async function loadHistory() {
  $("chat").innerHTML = "";
  if (!token) return;
  try {
    const data = await api("/api/history");
    data.messages.forEach((m) => addBubble(m.role, m.content));
  } catch (e) { console.warn(e); }
}

/* ---------------- send ---------------- */
$("send").onclick = send;
$("message").addEventListener("keydown", (e) => { if (e.key === "Enter") send(); });

async function send() {
  const text = $("message").value.trim();
  if ((!text && !pendingImage) || !token) return;
  $("message").value = "";
  addBubble("user", text || "📷 " + (pendingImage.name || "crop photo"));
  typingIndicator();
  try {
    // always send multipart form-data (text and/or image)
    const fd = new FormData();
    fd.append("message", text);
    if (pendingImage) fd.append("image", pendingImage);
    const data = await api("/api/chat", { method: "POST", body: fd });
    removeTyping();
    addBubble("assistant", data.advice, data);
    speak(data.advice);
  } catch (err) {
    removeTyping();
    addBubble("assistant", "⚠️ " + err.message);
  } finally {
    clearImage();
  }
}

/* ---------------- image upload ---------------- */
$("file").addEventListener("change", (e) => {
  const f = e.target.files[0];
  if (!f) return;
  pendingImage = f;
  $("preview-img").src = URL.createObjectURL(f);
  $("image-preview").style.display = "flex";
});
function clearImage() {
  pendingImage = null;
  $("file").value = "";
  $("image-preview").style.display = "none";
}
$("remove-image").onclick = clearImage;

/* ---------------- voice input (Web Speech API) ---------------- */
$("mic").onclick = () => {
  const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
  if (!SR) { alert("Voice input needs Chrome or Edge browser"); return; }
  if (listening) { recognition.stop(); return; }
  recognition = new SR();
  const lang = (currentUser && currentUser.language) || "en";
  recognition.lang = VOICE_LANG[lang] || "en-IN";
  recognition.interimResults = false;
  recognition.onresult = (e) => {
    $("message").value += e.results[0][0].transcript;
    $("message").focus();
  };
  recognition.onend = () => { listening = false; $("mic").classList.remove("on"); };
  recognition.onerror = () => { listening = false; $("mic").classList.remove("on"); };
  listening = true;
  $("mic").classList.add("on");
  recognition.start();
};

/* ---------------- voice output (browser TTS) ---------------- */
$("stop-voice").onclick = () => { try { speechSynthesis.cancel(); } catch (e) {} };

function speak(text) {
  try {
    const u = new SpeechSynthesisUtterance(text.replace(/[#*_]/g, ""));
    u.lang = VOICE_LANG[(currentUser && currentUser.language) || "en"] || "en-IN";
    speechSynthesis.cancel();
    speechSynthesis.speak(u);
  } catch (e) { /* ignore */ }
}

/* ---------------- boot ---------------- */
(async function boot() {
  if (!token) { showAuth(); return; }
  try {
    currentUser = await api("/api/me");
    enterChat();
  } catch (e) {
    showAuth();
  }
})();
