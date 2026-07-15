import sys
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import random
import re

import numpy as np
import pandas as pd
from pypdf import PdfReader
import docx
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split

# Matplotlib integration for Tkinter
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Config / constants
# ---------------------------------------------------------------------------
TRAITS = ["Openness", "Conscientiousness", "Extraversion", "Agreeableness", "Emotional Stability"]

TRAIT_COLORS = {
    "Openness": "#7C6FDB",
    "Conscientiousness": "#2E9C8F",
    "Extraversion": "#E8834A",
    "Agreeableness": "#D8607A",
    "Emotional Stability": "#4A7FC9",
}

RANDOM_SEED = 42
N_TRAINING_SAMPLES = 1500

# ---------------------------------------------------------------------------
# Synthetic training data generation
# ---------------------------------------------------------------------------
KEYWORD_BANK = {
    "Openness": [
        "pioneered an innovative approach to {domain}",
        "explored creative solutions for {domain} challenges",
        "designed a novel {domain} concept from scratch",
        "conducted independent research into emerging {domain} trends",
        "experimented with unconventional methods to improve {domain}",
        "conceptualized an original {domain} strategy",
        "curious self-starter who enjoys tackling ambiguous {domain} problems",
        "developed an imaginative redesign of the {domain} workflow",
        "pursued interdisciplinary projects spanning {domain} and design",
        "introduced a fresh perspective to the {domain} team",
    ],
    "Conscientiousness": [
        "meticulously planned and delivered {domain} projects on schedule",
        "maintained a detail-oriented, organized approach to {domain} tasks",
        "consistently met tight deadlines across multiple {domain} initiatives",
        "developed systematic processes to streamline {domain} operations",
        "managed {domain} budgets with thorough, disciplined tracking",
        "known for reliable, high-quality {domain} deliverables",
        "optimized {domain} workflows for maximum efficiency",
        "ensured rigorous quality control throughout the {domain} process",
        "kept meticulous records of all {domain} activities",
        "structured and prioritized {domain} tasks to hit every milestone",
    ],
    "Extraversion": [
        "led a cross-functional team of engineers on a {domain} initiative",
        "presented {domain} results to senior stakeholders and clients",
        "spearheaded company-wide {domain} events and outreach",
        "actively networked with {domain} partners at industry conferences",
        "motivated and energized the {domain} team to exceed targets",
        "initiated and drove a new {domain} partnership",
        "confident public speaker at {domain} conferences",
        "built strong client relationships through {domain} negotiations",
        "organized and hosted {domain} workshops for large audiences",
        "thrives in fast-paced, people-facing {domain} environments",
    ],
    "Agreeableness": [
        "mentored junior colleagues on {domain} best practices",
        "volunteered extensively in community {domain} programs",
        "known as a supportive, collaborative {domain} team player",
        "coordinated closely with peers to resolve {domain} conflicts amicably",
        "assisted teammates generously with {domain} workload during crunch periods",
        "built trust across departments during the {domain} rollout",
        "patient and empathetic when training new {domain} hires",
        "facilitated consensus among stakeholders on {domain} decisions",
        "contributed to a positive, cooperative {domain} team culture",
        "consistently prioritized team success over individual {domain} credit",
    ],
    "Emotional Stability": [
        "remained calm and composed while managing a {domain} crisis",
        "adapted smoothly to sudden changes in {domain} priorities",
        "handled high-pressure {domain} deadlines without losing focus",
        "demonstrated resilience through a challenging {domain} restructuring",
        "maintained steady, consistent performance during {domain} audits",
        "stayed level-headed when resolving urgent {domain} escalations",
        "managed stress effectively while juggling several {domain} projects",
        "provided a stabilizing presence for the {domain} team under pressure",
        "recovered quickly from {domain} setbacks and kept the team focused",
        "balanced competing {domain} demands with a composed, steady approach",
    ],
}

NEUTRAL_FILLER = [
    "developed and implemented {domain} solutions using industry-standard tools",
    "responsible for day-to-day {domain} operations",
    "bachelor's degree with a focus on {domain}",
    "proficient in relevant {domain} software and platforms",
    "utilized data analysis to support {domain} decisions",
    "completed a {duration}-month internship in {domain}",
    "contributed to key {domain} deliverables as part of a larger team",
    "certified in {domain}-related tools and methodologies",
    "gained hands-on experience with {domain} systems",
    "supported the {domain} function across multiple projects",
]

DOMAINS = [
    "marketing", "software engineering", "finance", "operations", "sales",
    "customer support", "product design", "human resources", "data analysis",
    "supply chain", "research", "quality assurance", "business development",
]


def _make_resume_and_scores(rnd: random.Random, rng: np.random.Generator):
    counts = {t: rnd.randint(0, 4) for t in TRAITS}
    bullets = []
    for trait, n in counts.items():
        chosen = rnd.sample(KEYWORD_BANK[trait], k=min(n, len(KEYWORD_BANK[trait])))
        bullets.extend(c.format(domain=rnd.choice(DOMAINS)) for c in chosen)

    for c in rnd.sample(NEUTRAL_FILLER, k=rnd.randint(2, 4)):
        bullets.append(c.format(domain=rnd.choice(DOMAINS), duration=rnd.choice([3, 6, 12])))

    rnd.shuffle(bullets)
    text = ". ".join(b[0].upper() + b[1:] for b in bullets) + "."

    max_count = 4
    scores = {}
    for t in TRAITS:
        base = 35 + (counts[t] / max_count) * 55
        noise = rng.normal(0, 5)
        scores[t] = float(np.clip(base + noise, 0, 100))
    return text, scores


def generate_training_data(n_samples: int = N_TRAINING_SAMPLES, seed: int = RANDOM_SEED) -> pd.DataFrame:
    rnd = random.Random(seed)
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(n_samples):
        text, scores = _make_resume_and_scores(rnd, rng)
        row = {"text": text}
        row.update(scores)
        rows.append(row)
    return pd.DataFrame(rows)


def train_models():
    df = generate_training_data()
    vectorizer = TfidfVectorizer(max_features=400, ngram_range=(1, 2), stop_words="english")
    X = vectorizer.fit_transform(df["text"])
    X_train, _, y_train, _ = train_test_split(X, df[TRAITS], test_size=0.2, random_state=RANDOM_SEED)

    models = {}
    for t in TRAITS:
        model = Ridge(alpha=3.0, random_state=RANDOM_SEED)
        model.fit(X_train, y_train[t])
        models[t] = model

    return vectorizer, models


# ---------------------------------------------------------------------------
# Text Extraction & Logic
# ---------------------------------------------------------------------------
def extract_text_from_pdf(filepath) -> str:
    reader = PdfReader(filepath)
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def extract_text_from_docx(filepath) -> str:
    document = docx.Document(filepath)
    return "\n".join(p.text for p in document.paragraphs)


def extract_resume_text(filepath) -> str:
    name = filepath.lower()
    if name.endswith(".pdf"):
        return extract_text_from_pdf(filepath)
    if name.endswith(".docx"):
        return extract_text_from_docx(filepath)
    if name.endswith(".txt"):
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()
    raise ValueError("Unsupported file type.")


def clean_text(text: str) -> str:
    text = re.sub(r"[^A-Za-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text.lower()


def predict_personality(text: str, vectorizer, models) -> dict:
    vec = vectorizer.transform([clean_text(text)])
    return {t: float(np.clip(models[t].predict(vec)[0], 0, 100)) for t in TRAITS}


def score_band(score: float) -> str:
    if score >= 65: return "High"
    if score <= 35: return "Low"
    return "Moderate"


TRAIT_INTERPRETATIONS = {
    "Openness": {
        "High": "Shows strong curiosity and creativity — likely drawn to new ideas, varied projects, and unconventional approaches.",
        "Moderate": "Shows a balanced mix of creativity and practicality, comfortable with both new ideas and established methods.",
        "Low": "Leans toward proven, familiar approaches over experimentation — likely values consistency over novelty.",
    },
    "Conscientiousness": {
        "High": "Highly organized and detail-oriented — resume reflects strong planning, reliability, and follow-through.",
        "Moderate": "Reasonably organized with a workable balance of structure and flexibility.",
        "Low": "Resume shows less emphasis on structure or planning — may thrive in loosely-defined, flexible roles.",
    },
    "Extraversion": {
        "High": "Comes across as sociable and assertive — comfortable leading, presenting, and engaging with others.",
        "Moderate": "Shows a moderate mix of independent and people-facing work.",
        "Low": "Leans toward independent, focused work over highly social or leadership-facing roles.",
    },
    "Agreeableness": {
        "High": "Strong emphasis on collaboration, empathy, and supporting others — likely a strong team player.",
        "Moderate": "Shows a reasonable balance of teamwork and independent contribution.",
        "Low": "Resume emphasizes individual achievement more than collaborative or supportive work.",
    },
    "Emotional Stability": {
        "High": "Resume suggests composure under pressure and resilience through change.",
        "Moderate": "Shows a typical mix of steady performance with some challenges under pressure.",
        "Low": "Fewer explicit signals of steady performance under high-pressure situations in this resume.",
    },
}

ROLE_RULES = [
    (lambda s: s["Extraversion"] >= 60 and s["Agreeableness"] >= 55, "Sales, Client Relations, or Business Development"),
    (lambda s: s["Conscientiousness"] >= 60 and s["Emotional Stability"] >= 55, "Project Management, Operations, or Quality Assurance"),
    (lambda s: s["Openness"] >= 60, "Product Design, R&D, or Innovation-focused roles"),
    (lambda s: s["Agreeableness"] >= 60 and s["Conscientiousness"] >= 50, "HR, Customer Success, or Team Coordination"),
    (lambda s: s["Extraversion"] >= 55 and s["Openness"] >= 55, "Marketing, Public Relations, or Community roles"),
]


def suggest_roles(scores: dict) -> list:
    suggestions = [label for cond, label in ROLE_RULES if cond(scores)]
    if not suggestions:
        suggestions = ["Roles requiring steady, independent, detail-focused work (e.g., Analysis, Research, Technical Writing)"]
    return list(dict.fromkeys(suggestions))[:3]


def build_report_text(scores: dict, roles: list) -> str:
    lines = ["PERSONALITY PREDICTION REPORT", "=" * 55, ""]
    for t in TRAITS:
        lines.append(f"{t}: {scores[t]:.0f}/100 ({score_band(scores[t])})")
        lines.append(f"  {TRAIT_INTERPRETATIONS[t][score_band(scores[t])]}\n")
    lines.append("Suggested Role Fit: " + ", ".join(roles))
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Tkinter GUI App Class
# ---------------------------------------------------------------------------
class PersonalityApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Personality Prediction via CV Analysis")
        self.geometry("1240x760")
        
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TNotebook", background="#F4F6F9", borderwidth=0)
        style.configure("TNotebook.Tab", font=("Segoe UI", 10, "bold"), padding=[15, 6], background="#E0E4E8")
        style.map("TNotebook.Tab", background=[("selected", "#FFFFFF")])

        self.vectorizer, self.models = train_models()
        self.current_scores = None
        self.current_roles = None
        
        self.setup_ui()

    def setup_ui(self):
        header_frame = tk.Frame(self, bg="#1E88E5", height=80)
        header_frame.pack(fill=tk.X)
        header_frame.pack_propagate(False)
        
        lbl_title = tk.Label(header_frame, text="🧭 Personality Prediction via CV Analysis", font=("Segoe UI", 20, "bold"), fg="#FFFFFF", bg="#1E88E5")
        lbl_title.pack(anchor="w", padx=20, pady=(10, 0))
        lbl_sub = tk.Label(header_frame, text="AI-Powered Big Five (OCEAN) Profile Inferences from Resume Semantics", font=("Segoe UI", 9), fg="#E3F2FD", bg="#1E88E5")
        lbl_sub.pack(anchor="w", padx=22, pady=(0, 10))

        main_container = tk.Frame(self, bg="#F4F6F9")
        main_container.pack(fill=tk.BOTH, expand=True, padx=15, pady=15)

        left_card = tk.Frame(main_container, bg="#FFFFFF", relief=tk.SOLID, bd=0, highlightbackground="#D1D5DB", highlightthickness=1)
        left_card.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
        
        tk.Label(left_card, text="Resume Source Input", font=("Segoe UI", 13, "bold"), fg="#232B3A", bg="#FFFFFF").pack(anchor="w", padx=15, pady=(15, 5))
        
        self.notebook = ttk.Notebook(left_card)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=15, pady=5)
        
        tab_upload = tk.Frame(self.notebook, bg="#FFFFFF")
        self.notebook.add(tab_upload, text="  📄 Upload Document  ")
        
        btn_upload = tk.Button(tab_upload, text="Browse System Files (PDF, DOCX, TXT)", command=self.upload_file, 
                               bg="#F1F5F9", fg="#1E88E5", font=("Segoe UI", 10, "bold"), activebackground="#E2E8F0",
                               bd=1, relief=tk.GROOVE, cursor="hand2", padx=20, pady=12)
        btn_upload.pack(pady=50)
        self.lbl_file_status = tk.Label(tab_upload, text="No file attached yet", font=("Segoe UI", 9, "italic"), bg="#FFFFFF", fg="#6B7280")
        self.lbl_file_status.pack()

        tab_paste = tk.Frame(self.notebook, bg="#FFFFFF")
        self.notebook.add(tab_paste, text="  ✍️ Direct Text Entry  ")
        
        self.txt_area = scrolledtext.ScrolledText(tab_paste, wrap=tk.WORD, font=("Consolas", 10), bg="#F8FAFC", fg="#334155", bd=1, relief=tk.SOLID)
        self.txt_area.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        btn_frame = tk.Frame(left_card, bg="#FFFFFF")
        btn_frame.pack(fill=tk.X, padx=15, pady=15)
        
        btn_analyze = tk.Button(btn_frame, text="🔍 Run Assessment", command=self.run_analysis, bg="#1E88E5", fg="white", 
                                 activebackground="#1565C0", activeforeground="white", font=("Segoe UI", 11, "bold"), bd=0, cursor="hand2", padx=15, pady=8)
        btn_analyze.pack(side=tk.LEFT, padx=(0, 10))
        
        self.btn_download = tk.Button(btn_frame, text="⬇️ Export Text Report", command=self.download_report, state=tk.DISABLED, 
                                       bg="#00897B", fg="white", activebackground="#00695C", activeforeground="white", font=("Segoe UI", 11, "bold"), bd=0, cursor="hand2", padx=15, pady=8)
        self.btn_download.pack(side=tk.LEFT)

        self.right_card = tk.Frame(main_container, bg="#FFFFFF", relief=tk.SOLID, bd=0, highlightbackground="#D1D5DB", highlightthickness=1)
        self.right_card.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(10, 0))
        
        self.lbl_results_placeholder = tk.Label(self.right_card, text="📊 Evaluation Metrics Viewport\n\nRun the assessment on a candidate profile to parse visualization assets.", 
                                                font=("Segoe UI", 11, "italic"), fg="#6B7280", bg="#FFFFFF", justify=tk.CENTER)
        self.lbl_results_placeholder.pack(expand=True, padx=20, pady=20)

    def upload_file(self):
        filepath = filedialog.askopenfilename(filetypes=[("Resume Formats", "*.pdf *.docx *.txt")])
        if filepath:
            try:
                text = extract_resume_text(filepath)
                self.txt_area.delete("1.0", tk.END)
                self.txt_area.insert(tk.END, text)
                self.lbl_file_status.configure(text=f"✔ Attached: {filepath.split('/')[-1]}", fg="#00897B", font=("Segoe UI", 9, "bold"))
                self.notebook.select(1)
            except Exception as e:
                messagebox.showerror("IO Parsing Error", f"Could not extract structure: {e}")

    def run_analysis(self):
        text = self.txt_area.get("1.0", tk.END).strip()
        if not text:
            messagebox.showwarning("Empty Context", "Submission context empty.")
            return
        
        self.current_scores = predict_personality(text, self.vectorizer, self.models)
        self.current_roles = suggest_roles(self.current_scores)
        
        for widget in self.right_card.winfo_children():
            widget.destroy()
            
        self.display_results()
        self.btn_download.configure(state=tk.NORMAL)

    def display_results(self):
        tk.Label(self.right_card, text="Psychometric Analysis Output", font=("Segoe UI", 14, "bold"), fg="#1F2937", bg="#FFFFFF").pack(anchor="w", padx=15, pady=(15, 0))
        
        role_card = tk.Frame(self.right_card, bg="#F0FDF4", highlightbackground="#DCFCE7", highlightthickness=1, padx=12, pady=8)
        role_card.pack(fill=tk.X, padx=15, pady=10)
        tk.Label(role_card, text="RECOMMENDED ROLE VECTORS:", font=("Segoe UI", 9, "bold"), fg="#166534", bg="#F0FDF4").pack(anchor="w")
        tk.Label(role_card, text="  •  ".join(self.current_roles), font=("Segoe UI", 11, "bold"), fg="#15803D", bg="#F0FDF4", justify=tk.LEFT, wraplength=520).pack(anchor="w", pady=(2, 0))
        
        visual_split = tk.Frame(self.right_card, bg="#FFFFFF")
        visual_split.pack(fill=tk.BOTH, expand=True, padx=15, pady=5)
        
        chart_frame = tk.Frame(visual_split, bg="#FFFFFF")
        chart_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        fig = plt.figure(figsize=(3.6, 3.4), dpi=100, facecolor='none')
        ax = fig.add_subplot(111, polar=True, facecolor='#FAFAFA')
        
        # Safe shortened categories naming strings to prevent horizontal screen clip
        categories = ["Openness", "Conscientious.", "Extraversion", "Agreeable.", "Emotional St."]
        categories = categories + [categories[0]]
        values = [self.current_scores[t] for t in TRAITS] + [self.current_scores[TRAITS[0]]]
        
        ax.plot(categories, values, color="#1E88E5", linewidth=2.5, linestyle='solid')
        ax.fill(categories, values, color="#1E88E5", alpha=0.15)
        ax.set_ylim(0, 100)
        ax.grid(color='#E5E7EB', linestyle='--', linewidth=0.6)
        ax.tick_params(colors='#4B5563', labelsize=8, pad=8)
        
        # Safe padding balance config so left/right labels won't hit boundaries
        fig.subplots_adjust(left=0.20, right=0.85, top=0.85, bottom=0.15)
        
        canvas = FigureCanvasTkAgg(fig, master=chart_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        breakdown_scroll = scrolledtext.ScrolledText(visual_split, wrap=tk.WORD, font=("Segoe UI", 9), bg="#F8FAFC", fg="#334155", bd=0, highlightthickness=1, highlightbackground="#E2E8F0", width=34)
        breakdown_scroll.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(10, 0), pady=5)
        
        for t in TRAITS:
            score = self.current_scores[t]
            band = score_band(score)
            desc = TRAIT_INTERPRETATIONS[t][band]
            
            breakdown_scroll.insert(tk.END, f"■ {t.upper()}  |  {score:.0f}/100 [{band.upper()}]\n", "title")
            breakdown_scroll.insert(tk.END, f"{desc}\n\n", "body")
            
        breakdown_scroll.tag_config("title", font=("Segoe UI", 9, "bold"), fg="#1E88E5")
        breakdown_scroll.tag_config("body", fg="#475569")
        breakdown_scroll.configure(state=tk.DISABLED)

    def download_report(self):
        if not self.current_scores:
            return
        filepath = filedialog.asksaveasfilename(defaultextension=".txt", filetypes=[("Text Files", "*.txt")])
        if filepath:
            report_text = build_report_text(self.current_scores, self.current_roles)
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(report_text)
            messagebox.showinfo("Pipeline Status", "Assessment exported flawlessly.")


if __name__ == "__main__":
    app = PersonalityApp()
    app.mainloop()