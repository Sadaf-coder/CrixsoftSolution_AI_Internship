# Personality Prediction System Through CV Analysis

Predicts a **Big Five (OCEAN)** personality profile from a resume/CV using NLP + machine learning — built as a single-file Streamlit app.

## Features
- Upload a resume as **PDF, DOCX, or TXT** — or paste text directly
- Predicts 5 traits: Openness, Conscientiousness, Extraversion, Agreeableness, Emotional Stability
- Interactive radar chart + per-trait score cards with plain-English interpretations
- Rule-based **Suggested Role Fit**
- Model transparency panel: R² / MAE per trait, and the top words driving each prediction
- Downloadable `.txt` report

## Tech Stack
| Purpose | Library |
|---|---|
| App / UI | Streamlit |
| ML model | scikit-learn (TF-IDF + Ridge regression, one model per trait) |
| Resume parsing | pypdf, python-docx |
| Visualization | Plotly |

## Project Structure
```
.
├── app.py             # entire app: data generation, training, UI
├── requirements.txt
└── README.md
```

## Setup (VS Code)

1. Open this folder in VS Code.
2. Create and activate a virtual environment (recommended):
   ```bash
   # Windows
   python -m venv venv
   venv\Scripts\activate

   # macOS / Linux
   python3 -m venv venv
   source venv/bin/activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Run the app:
   ```bash
   streamlit run app.py
   ```
5. It opens automatically at `http://localhost:8501`. First load trains the models (well under a second) and caches them for the rest of the session.

## How It Works

1. **Synthetic training data** — there is no public, ethically-sourced dataset pairing real resumes with verified personality labels. This project generates ~1,500 synthetic resume-style texts from psychology-informed keyword/phrase banks (e.g. *"led a cross-functional team"* → Extraversion signal), and derives each trait label from which phrases were included, plus random noise. This is a standard technique called **weak / heuristic supervision**, used when ground-truth labels aren't available.
2. **Feature extraction** — TF-IDF (unigrams + bigrams, top 400 features).
3. **Modeling** — 5 independent Ridge regression models, one per trait.
4. **Prediction** — new resume text goes through the same TF-IDF transform, then each trait model scores it 0–100.

Typical held-out R² on the synthetic validation split is ~0.77–0.88 per trait (visible live in the app's "Model performance" panel).

## ⚠️ Limitations (worth keeping for your report / viva)

- Predicting personality from resume text is scientifically contested — resumes are curated, persuasive documents, not validated psychometric instruments. Real personality assessment normally relies on validated questionnaires (e.g. NEO-PI-R) or structured interviews.
- Training data here is **synthetic**, built from hand-designed keyword associations. It encodes the assumption "these words correlate with these traits" rather than learning that association from real psychological ground truth.
- Predictions therefore reflect **language patterns in the text**, not verified personality — two people who write similarly will score similarly regardless of their actual personalities.
- **This is an educational/demo project, not a hiring tool.** Personality-from-text systems raise real fairness and bias concerns in actual recruiting, and are not considered scientifically validated for hiring decisions by most I/O psychology research.

## Possible Extensions
- Swap in a real Big-Five-labeled text corpus (e.g. the Pennebaker & King essays dataset) for a more defensible training signal
- Add SHAP/LIME-based explanations
- Multi-resume batch comparison view
- Fine-tune a small transformer model instead of TF-IDF + Ridge