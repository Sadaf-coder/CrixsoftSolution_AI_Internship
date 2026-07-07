# TakeHelp — AI Chatbot

NLP customer support chatbot in one file (`app.py`): training data, ML
model, Flask server, and chat UI (HTML/CSS/JS) all included.

## How it works

1. **Training data** — `INTENTS` list in `app.py` (tag + example patterns + responses).
2. **Keyword layer** — checks the message for obvious topic words (order, refund, product, human, hours...) first, so short/casual replies aren't misread.
3. **NLP model** — TF-IDF + Logistic Regression, trained at startup, handles phrasing the keyword layer misses.
4. **Flask server** — `/api/chat` classifies each message and replies.
5. **Chat UI** — inlined in `app.py`, ticket-desk themed.

## Run it

```bash
pip install -r requirements.txt
python app.py
```
Open `http://localhost:5000`.

## Extend it

- New topic: add a dict to `INTENTS` (tag, patterns, responses) + a few keywords to `KEYWORDS`. Restart to retrain.
- Real backend: branch on the resolved tag inside `get_reply()`.
- Deploy: any Python host (Render, Railway, PythonAnywhere, VPS) — embed on a site or call `/api/chat` from a widget.

## Stack

Python · scikit-learn (TF-IDF + Logistic Regression) · Flask · HTML/CSS/JS
