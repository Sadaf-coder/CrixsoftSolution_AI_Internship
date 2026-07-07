# TakeHelp — AI Chatbot (single-file version)

An NLP-powered customer support chatbot, built as an AI internship project.
Everything — training data, the machine learning model, the Flask server,
and the chat UI (HTML/CSS/JS) — lives in one file: `app.py`.

## How it works

1. **Training data** — a Python list called `INTENTS` near the top of
   `app.py`. Each intent has a `tag`, example `patterns` (things a user
   might type), and possible `responses`.
2. **NLP model** — at startup, the app converts the example patterns into
   TF-IDF vectors (a way of numerically weighing words by importance) and
   trains a Logistic Regression classifier. This lets the bot recognize
   messages it has never seen before — e.g. "when will my package
   arrive?" is understood as an order-status question even though that
   exact sentence isn't in the training data.
3. **Flask server** — serves the chat page and an `/api/chat` endpoint.
   Each incoming message is classified, and the bot replies with a
   matching response. If it isn't confident about any intent, it asks a
   clarifying question instead of guessing.
4. **Chat UI** — inlined directly in `app.py` as an HTML template string,
   styled like a support-ticket desk.

## Setup & running

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the app (trains the model automatically on startup)
python app.py

# 3. Open in your browser
http://localhost:5000
```

No separate training step needed — the model trains in memory each time
the app starts.

## Extending the chatbot

- **Add new topics**: open `app.py`, find the `INTENTS` list, and add a
  new dictionary with a `tag`, 5–10 example `patterns`, and a list of
  `responses`. Restart the app to retrain.
- **Connect to a real backend**: inside `get_reply()`, branch on
  `best_tag` to call a real order-tracking API, database, or ticketing
  system instead of returning a canned response.
- **Deploy**: this Flask app can be deployed to any platform that runs
  Python (Render, Railway, PythonAnywhere, a VPS, etc.) and embedded on a
  website, or its `/api/chat` endpoint can be called from a widget on a
  messaging app or social media integration.

## Tech stack

- **Python** — core logic
- **scikit-learn** — TF-IDF vectorization + Logistic Regression for intent
  classification (lightweight, fast, explainable — no GPU or large model
  downloads required)
- **Flask** — web server and REST API
- **HTML/CSS/JS** — inlined chat widget front-end