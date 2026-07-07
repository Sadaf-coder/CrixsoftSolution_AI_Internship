"""
app.py
------
TakeHelp — a single-file NLP customer support chatbot.

Everything lives in this one file:
  - Training data (intents, example phrases, responses)
  - The NLP model (TF-IDF + Logistic Regression, trained at startup)
  - The Flask web server (chat API)
  - The chat UI (HTML/CSS/JS, inlined as a template string)

Run with:
    pip install -r requirements.txt
    python app.py
Then open http://localhost:5000
"""

import random
import re
from flask import Flask, request, jsonify, render_template_string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

# ---------------------------------------------------------------------------
# 1. Training data — add more tags/patterns/responses here to extend the bot
# ---------------------------------------------------------------------------
INTENTS = [
    {
        "tag": "greeting",
        "patterns": ["hi", "hello", "hey", "good morning", "good evening",
                     "is anyone there", "hey there", "what's up", "yo", "hiya"],
        "responses": ["Hey there! I'm TakeHelp, your support assistant. What can I help you with today?",
                      "Hello! How can I help you today?"]
    },
    {
        "tag": "goodbye",
        "patterns": ["bye", "goodbye", "see you later", "talk to you later",
                     "i have to go", "that's all for now"],
        "responses": ["Goodbye! Reach out anytime you need help.",
                      "Take care! Come back if you have more questions."]
    },
    {
        "tag": "thanks",
        "patterns": ["thanks", "thank you", "that helped", "appreciate it", "thanks a lot"],
        "responses": ["You're welcome! Anything else I can help with?",
                      "Glad I could help!"]
    },
    {
        "tag": "order_status",
        "patterns": ["where is my order", "track my order", "order status",
                     "has my order shipped", "when will my package arrive",
                     "track my package", "where is my order number 123",
                     "my delivery hasn't arrived"],
        "responses": ["I can help track your order. Could you share your order ID? It usually starts with #ORD.",
                      "Sure, please provide your order number and I'll check the status."]
    },
    {
        "tag": "refund",
        "patterns": ["i want a refund", "how do i get a refund", "cancel my order and refund",
                     "refund policy", "money back", "i want to cancel and get my money back",
                     "can i return this item for a refund", "what is your return policy",
                     "return policy", "what is the returns policy", "how do returns work",
                     "can i return this product", "i want to return an item"],
        "responses": ["Refunds are processed within 5-7 business days once the return is received. Would you like me to start a refund request?",
                      "I can help with that. Could you share the order ID you'd like refunded?"]
    },
    {
        "tag": "product_info",
        "patterns": ["tell me about your products", "what do you sell", "product details",
                     "do you have this in stock", "what sizes are available",
                     "what about products", "tell me about products", "show me your products",
                     "what products do you have", "product information", "products available"],
        "responses": ["We offer a range of products across categories. Could you tell me which product you're interested in?",
                      "Happy to help! Which product would you like details on?"]
    },
    {
        "tag": "human_agent",
        "patterns": ["talk to a human", "connect me to an agent", "i want a real person",
                     "customer service representative", "speak to support staff",
                     "can i speak to a real person", "get me a human agent"],
        "responses": ["I'll connect you with a human agent right away. Please hold on.",
                      "Sure, transferring you to a live support agent now."]
    },
    {
        "tag": "hours",
        "patterns": ["what are your working hours", "when are you open",
                     "support hours", "are you open now"],
        "responses": ["Our support team is available Monday to Saturday, 9 AM to 9 PM.",
                      "We're open 9 AM - 9 PM, Monday through Saturday."]
    },
    {
        "tag": "complaint",
        "patterns": ["i have a complaint", "i am not satisfied", "this is a bad experience",
                     "i want to complain", "your service is bad"],
        "responses": ["I'm sorry to hear that. Could you describe the issue in detail so I can help resolve it?",
                      "I apologize for the trouble. Please share more details and I'll escalate this for you."]
    },
    {
        "tag": "acknowledgment",
        "patterns": ["ok", "okay", "alright", "sure", "got it", "cool", "fine",
                     "yeah", "yep", "sounds good", "understood", "noted"],
        "responses": ["Got it! Let me know if there's anything else I can help with.",
                      "Sounds good — I'm here if you need anything else."]
    },
    {
        "tag": "fallback",
        "patterns": [],
        "responses": ["I'm not fully sure I understood that. Could you rephrase, or ask me about orders, refunds, products, or support hours?",
                      "Sorry, I didn't quite get that. Try asking about your order, a refund, or our products."]
    },
]

CONFIDENCE_THRESHOLD = 0.20

# A lightweight keyword layer that runs BEFORE the ML model. Small training
# datasets like this one can make the ML model's confidence spread thin
# across classes for short or unusual phrasing (e.g. "what about products",
# "okay"). Catching obvious topic words directly makes the bot noticeably
# less rigid, while the ML model still handles longer/less obvious phrasing.
# Order matters: more specific tags are checked before generic ones.
KEYWORDS = [
    ("human_agent", ["human", "agent", "representative", "real person"]),
    ("refund", ["refund", "return", "returns", "money back"]),
    ("order_status", ["order", "package", "shipment", "delivery", "shipped", "track"]),
    ("product_info", ["product", "products", "in stock", "sizes"]),
    ("hours", ["hours", "open", "working hours"]),
    ("complaint", ["complaint", "complain", "not satisfied", "bad experience", "bad service"]),
    ("thanks", ["thanks", "thank you", "appreciate"]),
    ("goodbye", ["bye", "goodbye", "see you"]),
    ("greeting", ["hi", "hello", "hey", "hiya", "yo"]),
    ("acknowledgment", ["ok", "okay", "alright", "sure", "got it", "cool", "fine", "yep", "yeah"]),
]

# ---------------------------------------------------------------------------
# 2. Train the NLP model in memory at startup
# ---------------------------------------------------------------------------
_texts, _labels = [], []
RESPONSES = {}
for intent in INTENTS:
    RESPONSES[intent["tag"]] = intent["responses"]
    for pattern in intent["patterns"]:
        _texts.append(pattern.lower())
        _labels.append(intent["tag"])

model = Pipeline([
    ("tfidf", TfidfVectorizer(ngram_range=(1, 2))),
    ("clf", LogisticRegression(max_iter=1000)),
])
model.fit(_texts, _labels)


def get_reply(user_message: str) -> str:
    if not user_message.strip():
        return "Could you type a message so I can help?"

    lowered = user_message.lower()

    # 1. Keyword layer — catches obvious topic words reliably regardless of
    #    exact phrasing or sentence length.
    for tag, keywords in KEYWORDS:
        for kw in keywords:
            if re.search(r"\b" + re.escape(kw) + r"\b", lowered):
                return random.choice(RESPONSES[tag])

    # 2. ML model — handles longer or less obvious phrasing the keyword
    #    layer doesn't catch.
    probs = model.predict_proba([lowered])[0]
    classes = model.classes_
    best_idx = probs.argmax()
    best_tag = classes[best_idx]
    best_prob = probs[best_idx]

    if best_prob < CONFIDENCE_THRESHOLD:
        best_tag = "fallback"

    return random.choice(RESPONSES.get(best_tag, RESPONSES["fallback"]))


# ---------------------------------------------------------------------------
# 3. Flask app + routes
# ---------------------------------------------------------------------------
app = Flask(__name__)


@app.route("/")
def home():
    return render_template_string(PAGE_HTML)


@app.route("/api/chat", methods=["POST"])
def chat():
    payload = request.get_json(silent=True) or {}
    user_message = payload.get("message", "")
    reply = get_reply(user_message)
    return jsonify({"reply": reply})


# ---------------------------------------------------------------------------
# 4. Chat UI — HTML, CSS and JS all inlined in one template string
# ---------------------------------------------------------------------------
PAGE_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>TakeHelp — Support Assistant</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,500;9..144,600&family=Inter:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>
:root {
  --bg: #F4F6F3;
  --panel: #FFFFFF;
  --ink: #16241F;
  --ink-soft: #4B5A54;
  --teal: #0B6E4F;
  --teal-dark: #084C37;
  --gold: #E8A33D;
  --border: #D8DED8;
  --user-bubble: #16241F;
  --bot-bubble: #FFFFFF;
}
* { box-sizing: border-box; }
body {
  margin: 0;
  font-family: 'Inter', sans-serif;
  background: var(--bg);
  color: var(--ink);
  height: 100vh;
  overflow: hidden;
}
.app { display: flex; height: 100vh; }

.brand-panel {
  width: 320px;
  min-width: 260px;
  background: var(--teal-dark);
  color: #EAF3EE;
  padding: 40px 28px;
  display: flex;
  flex-direction: column;
  justify-content: space-between;
}
.brand-mark .brand-stub {
  display: inline-block;
  font-family: 'IBM Plex Mono', monospace;
  font-size: 12px;
  letter-spacing: 0.06em;
  color: var(--gold);
  border: 1px dashed rgba(232, 163, 61, 0.5);
  border-radius: 20px;
  padding: 4px 12px;
  margin-bottom: 18px;
}
.brand-mark h1 {
  font-family: 'Fraunces', serif;
  font-size: 34px;
  font-weight: 600;
  margin: 0 0 12px 0;
  letter-spacing: -0.01em;
}
.tagline { font-size: 14.5px; line-height: 1.5; color: #BFD8CC; margin: 0 0 32px 0; }
.topics { list-style: none; padding: 0; margin: 0; font-size: 14px; color: #DDEBE3; }
.topics li {
  display: flex; align-items: center; gap: 10px; padding: 9px 0;
  border-bottom: 1px solid rgba(255,255,255,0.08);
}
.dot { width: 6px; height: 6px; border-radius: 50%; background: var(--gold); flex-shrink: 0; }
.status-card {
  display: flex; align-items: center; gap: 10px;
  background: rgba(255,255,255,0.06);
  border: 1px solid rgba(255,255,255,0.1);
  border-radius: 10px; padding: 12px 14px; margin-top: 32px;
}
.status-dot {
  width: 9px; height: 9px; border-radius: 50%; background: #4ADE80;
  box-shadow: 0 0 0 3px rgba(74, 222, 128, 0.2); flex-shrink: 0;
}
.status-title { margin: 0; font-size: 13px; font-weight: 600; color: #fff; }
.status-sub { margin: 2px 0 0 0; font-size: 12px; color: #9FC2AF; }

.chat-panel { flex: 1; display: flex; flex-direction: column; background: var(--bg); }
.chat-header {
  display: flex; align-items: baseline; justify-content: space-between;
  padding: 26px 40px 14px 40px; border-bottom: 1px solid var(--border);
}
.chat-header h2 { font-family: 'Fraunces', serif; font-size: 20px; font-weight: 600; margin: 0; }
.ticket-id { font-family: 'IBM Plex Mono', monospace; font-size: 12px; color: var(--ink-soft); }

.messages {
  flex: 1; overflow-y: auto; padding: 28px 40px;
  display: flex; flex-direction: column; gap: 18px;
}
.msg-row { display: flex; flex-direction: column; max-width: 62%; }
.msg-row.user { align-self: flex-end; align-items: flex-end; }
.msg-row.bot { align-self: flex-start; align-items: flex-start; }
.msg-label {
  font-family: 'IBM Plex Mono', monospace; font-size: 10.5px; letter-spacing: 0.08em;
  color: var(--ink-soft); margin-bottom: 6px;
}
.bubble { padding: 13px 16px; font-size: 14.5px; line-height: 1.5; border-radius: 4px; position: relative; }
.msg-row.bot .bubble {
  background: var(--bot-bubble); border: 1px solid var(--border);
  border-left: 3px dashed var(--teal); color: var(--ink);
}
.msg-row.user .bubble {
  background: var(--user-bubble); color: #F4F6F3; border-right: 3px dashed var(--gold);
}
.typing .bubble, .bubble.typing { display: flex; gap: 4px; padding: 15px 18px; }
.bubble.typing span {
  width: 6px; height: 6px; border-radius: 50%; background: var(--teal);
  opacity: 0.5; animation: pulse 1s infinite ease-in-out;
}
.bubble.typing span:nth-child(2) { animation-delay: 0.15s; }
.bubble.typing span:nth-child(3) { animation-delay: 0.3s; }
@keyframes pulse {
  0%, 80%, 100% { opacity: 0.3; transform: scale(0.85); }
  40% { opacity: 1; transform: scale(1); }
}

.composer { display: flex; gap: 12px; padding: 20px 40px 28px 40px; border-top: 1px solid var(--border); }
.composer input {
  flex: 1; padding: 13px 16px; border: 1px solid var(--border); border-radius: 8px;
  font-family: 'Inter', sans-serif; font-size: 14.5px; background: var(--panel); color: var(--ink);
}
.composer input:focus { outline: 2px solid var(--teal); outline-offset: 1px; }
.composer button {
  background: var(--teal); color: #fff; border: none; border-radius: 8px;
  padding: 0 24px; font-family: 'Inter', sans-serif; font-weight: 600; font-size: 14px;
  cursor: pointer; transition: background 0.15s ease;
}
.composer button:hover { background: var(--teal-dark); }
.composer button:focus-visible { outline: 2px solid var(--gold); outline-offset: 2px; }

@media (max-width: 760px) {
  .app { flex-direction: column; }
  .brand-panel { width: 100%; padding: 24px; flex-direction: row; align-items: center; justify-content: space-between; }
  .brand-mark p.tagline, .topics, .status-card { display: none; }
  .chat-header, .messages, .composer { padding-left: 20px; padding-right: 20px; }
  .msg-row { max-width: 85%; }
}
</style>
</head>
<body>
  <div class="app">
    <aside class="brand-panel">
      <div class="brand-mark">
        <span class="brand-stub">#TH-001</span>
        <h1>TakeHelp</h1>
        <p class="tagline">A ticket for every question.<br>Answered on the spot.</p>
      </div>

      <ul class="topics">
        <li><span class="dot"></span>Order status &amp; tracking</li>
        <li><span class="dot"></span>Refunds &amp; returns</li>
        <li><span class="dot"></span>Product questions</li>
        <li><span class="dot"></span>Talk to a human agent</li>
      </ul>

      <div class="status-card">
        <span class="status-dot"></span>
        <div>
          <p class="status-title">Assistant online</p>
          <p class="status-sub">Mon–Sat · 9 AM–9 PM</p>
        </div>
      </div>
    </aside>

    <main class="chat-panel">
      <header class="chat-header">
        <h2>Support conversation</h2>
        <span class="ticket-id" id="ticket-id">Ticket #-----</span>
      </header>

      <div class="messages" id="messages"></div>

      <form class="composer" id="composer">
        <input type="text" id="user-input" placeholder="Describe your issue or ask a question…" autocomplete="off" required>
        <button type="submit" aria-label="Send message">Send</button>
      </form>
    </main>
  </div>

<script>
const messagesEl = document.getElementById('messages');
const form = document.getElementById('composer');
const input = document.getElementById('user-input');
const ticketIdEl = document.getElementById('ticket-id');

ticketIdEl.textContent = 'Ticket #' + Math.floor(10000 + Math.random() * 89999);

function addMessage(text, sender) {
  const row = document.createElement('div');
  row.className = 'msg-row ' + sender;
  const bubble = document.createElement('div');
  bubble.className = 'bubble';
  bubble.textContent = text;
  const label = document.createElement('span');
  label.className = 'msg-label';
  label.textContent = sender === 'user' ? 'YOU' : 'TAKEHELP';
  row.appendChild(label);
  row.appendChild(bubble);
  messagesEl.appendChild(row);
  messagesEl.scrollTop = messagesEl.scrollHeight;
}

function addTyping() {
  const row = document.createElement('div');
  row.className = 'msg-row bot typing-row';
  row.id = 'typing-row';
  row.innerHTML = '<span class="msg-label">TAKEHELP</span><div class="bubble typing"><span></span><span></span><span></span></div>';
  messagesEl.appendChild(row);
  messagesEl.scrollTop = messagesEl.scrollHeight;
}

function removeTyping() {
  const row = document.getElementById('typing-row');
  if (row) row.remove();
}

addMessage("Hi, I'm TakeHelp. Ask me about an order, a refund, or anything else — I'm here to help.", 'bot');

form.addEventListener('submit', async (e) => {
  e.preventDefault();
  const text = input.value.trim();
  if (!text) return;
  addMessage(text, 'user');
  input.value = '';
  addTyping();
  try {
    const res = await fetch('/api/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message: text })
    });
    const data = await res.json();
    removeTyping();
    addMessage(data.reply, 'bot');
  } catch (err) {
    removeTyping();
    addMessage("I couldn't reach the server. Please check the connection and try again.", 'bot');
  }
});
</script>
</body>
</html>
"""

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)