import re
from pathlib import Path

import joblib
import nltk
import pandas as pd
import streamlit as st
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize


# ============================================================
# Page configuration
# ============================================================
st.set_page_config(
    page_title="Sentiment Analysis",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = Path(__file__).resolve().parent


# ============================================================
# NLTK resources
# ============================================================
def ensure_nltk_resources():
    resources = [
        ("punkt_tab", "tokenizers/punkt_tab"),
        ("stopwords", "corpora/stopwords"),
        ("wordnet", "corpora/wordnet"),
    ]
    for package, resource_path in resources:
        try:
            nltk.data.find(resource_path)
        except LookupError:
            nltk.download(package, quiet=True)


ensure_nltk_resources()


# ============================================================
# Custom UI
# ============================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Playfair+Display:wght@500;600&display=swap');

:root {
    --plum:#4C1F39; --violet:#757089; --violet-dark:#5F566F;
    --rose:#934A3F; --paper:#F9F6FA; --white:#FFFFFF;
    --ink:#29232B; --muted:#77707B; --line:#E8E1EA;
}
.stApp { background:var(--paper); color:var(--ink); font-family:'DM Sans',sans-serif; }
.block-container { max-width:1380px; padding:2.2rem 4rem 4rem; }
[data-testid="stSidebar"] { background:#F3EEF5; border-right:1px solid var(--line); }
h1,h2,h3 { font-family:'Playfair Display',serif !important; color:var(--plum) !important; }
h1 { font-size:3.7rem !important; line-height:1.02 !important; }
h2 { font-size:2.25rem !important; }
h3 { font-size:1.35rem !important; }
.hero {
    padding:3.4rem 3.6rem; border-radius:30px;
    background:linear-gradient(135deg,#EEE8F2 0%,#F9F6FA 58%,#E8DFEA 100%);
    border:1px solid #E5DDE8; margin-bottom:1.8rem; position:relative; overflow:hidden;
}
.hero:after {
    content:""; position:absolute; width:330px; height:330px; right:-90px; top:-120px;
    border-radius:50%; background:rgba(117,112,137,.13);
}
.eyebrow { color:var(--rose); font-size:.78rem; font-weight:700; letter-spacing:.18em; text-transform:uppercase; margin-bottom:.8rem; }
.hero-title { max-width:800px; font-family:'Playfair Display',serif; font-size:clamp(2.8rem,6vw,5.4rem); line-height:.98; color:var(--plum); position:relative; z-index:2; }
.hero-sub { max-width:720px; color:#6D6570; font-size:1.08rem; line-height:1.7; margin-top:1.2rem; position:relative; z-index:2; }
.pill { display:inline-block; padding:.45rem .75rem; border:1px solid #DCD3E2; background:rgba(255,255,255,.62); border-radius:999px; margin:.25rem .25rem .1rem 0; font-size:.78rem; color:var(--violet-dark); }
.section-kicker { color:var(--rose); font-weight:700; letter-spacing:.14em; text-transform:uppercase; font-size:.72rem; margin-top:2.2rem; }
.section-title { font-family:'Playfair Display',serif; font-size:2.1rem; color:var(--plum); margin:.2rem 0 1rem; }
.card,.kpi {
    background:var(--white); border:1px solid var(--line);
    border-radius:22px; box-shadow:0 10px 35px rgba(76,31,57,.045);
}
.card { padding:1.35rem 1.45rem; }
.kpi { padding:1.25rem 1.35rem; min-height:116px; }
.kpi-label { color:var(--muted); font-size:.78rem; text-transform:uppercase; letter-spacing:.08em; }
.kpi-value { color:var(--plum); font-family:'Playfair Display',serif; font-size:2rem; margin-top:.25rem; }
.kpi-note { color:var(--violet); font-size:.76rem; margin-top:.2rem; }
.result-positive,.result-negative,.result-neutral { border-radius:24px; padding:2rem; margin:1rem 0; border:1px solid var(--line); }
.result-positive { background:#F0F4F0; } .result-negative { background:#F8EEEE; } .result-neutral { background:#F1EFF5; }
.result-label { font-size:.75rem; text-transform:uppercase; letter-spacing:.15em; font-weight:700; }
.result-sentiment { font-family:'Playfair Display',serif; font-size:2.8rem; color:var(--plum); }

/* Medium prediction result card */
.result-card {
    width: 100%;
    box-sizing: border-box;
    border-radius: 22px;
    padding: 1.8rem 2.2rem;
    margin: 1rem 0;
    min-height: 180px;
    border: 1px solid var(--line);
    display: flex;
    flex-direction: column;
    justify-content: center;
    box-shadow: 0 8px 28px rgba(76,31,57,.045);
}
.result-card.positive { background: #F0F4F0; }
.result-card.negative { background: #F8EEEE; }
.result-card.neutral { background: #F1EFF5; }
.result-card .result-label {
    font-size: .82rem;
    text-transform: uppercase;
    letter-spacing: .17em;
    font-weight: 700;
    color: var(--plum);
    margin-bottom: .4rem;
}
.result-card .result-sub {
    font-family: 'Playfair Display', serif;
    font-size: 2.0rem;
    line-height: 1;
    font-weight: 600;
    color: var(--plum);
}
.result-card .result-sub { white-space: nowrap; letter-spacing: -.025em; }

.stButton > button { border-radius:999px; border:1px solid var(--plum); background:var(--plum); color:white; padding:.65rem 1.2rem; font-weight:600; }
.stButton > button:hover { background:var(--violet-dark); border-color:var(--violet-dark); }
.stTabs [data-baseweb="tab-list"] { gap:2rem; border-bottom:1px solid var(--line); }
.stTabs [data-baseweb="tab"] { color:var(--muted); font-weight:600; padding-left:0; padding-right:0; }
.stTabs [aria-selected="true"] { color:var(--plum) !important; }
[data-testid="stMetric"] { background:var(--white); border:1px solid var(--line); border-radius:18px; padding:1rem; }
div[data-testid="stDataFrame"] { border:1px solid var(--line); border-radius:18px; overflow:hidden; }
hr { border-color:var(--line); }
.small-muted { color:var(--muted); font-size:.86rem; line-height:1.6; }
.footer { text-align:center; color:#938B95; font-size:.75rem; padding:3rem 0 1rem; }

/* Keep Streamlit's native text readable in both Light and Dark mode.
   The existing custom cards/design remain unchanged. */
.stApp [data-testid="stMarkdownContainer"],
.stApp [data-testid="stCaptionContainer"],
.stApp label,
.stApp [data-testid="stText"] {
    color: var(--st-text-color, var(--ink));
}
.stApp input,
.stApp textarea,
.stApp [data-baseweb="select"] * {
    color: var(--st-text-color, var(--ink));
}
@media (max-width:900px) {
    .block-container { padding:1.2rem 1rem 3rem; }
    .hero { padding:2rem 1.4rem; border-radius:22px; }
    .hero-title { font-size:3rem; }
}
</style>
""", unsafe_allow_html=True)


# ============================================================
# Exact preprocessing used by the project notebook
# ============================================================
stop_words = set(stopwords.words("english"))
negation_words = {"no", "not", "never", "neither", "dont", "nor"}
stop_words = stop_words - negation_words
lemmatizer = WordNetLemmatizer()

spec_pattern = re.compile(
    r"^\d+(mAh|hz|mp|gb|tb|w|inch|ghz|nm|fps|ppi|nits)$",
    re.IGNORECASE,
)


def preprocess_text(text):
    tokens = word_tokenize(text)
    cleaned_tokens = []

    for word in tokens:
        word_lower = word.lower()

        if word.isalpha() and word_lower not in stop_words:
            cleaned_tokens.append(lemmatizer.lemmatize(word_lower))
        elif spec_pattern.match(word_lower):
            cleaned_tokens.append(word_lower)
        elif word.isdigit() and 1 <= int(word) <= 10:
            cleaned_tokens.append(word_lower)

    return " ".join(cleaned_tokens)


# ============================================================
# Model metadata and notebook evaluation results
# ============================================================
MODEL_FILES = {
    "Logistic Regression": "logistic_model.pkl",
    "Naive Bayes": "naive_bayes_model.pkl",
    "SVM": "svm_model.pkl",
    "Random Forest": "random_forest_model.pkl",
    "XGBoost": "xgboost_model.pkl",
}

MODEL_RESULTS = pd.DataFrame(
    [
        ["Logistic Regression", 0.8021, 0.7522, 0.8021, 0.7548],
        ["Naive Bayes", 0.7535, 0.6588, 0.7535, 0.6941],
        ["SVM", 0.7847, 0.7655, 0.7847, 0.7622],
        ["Random Forest", 0.7778, 0.6693, 0.7778, 0.7190],
        ["XGBoost", 0.8021, 0.7726, 0.8021, 0.7795],
    ],
    columns=["Model", "Accuracy", "Precision", "Recall", "F1-Score"],
)


@st.cache_data

def load_dataset_info():
    path = BASE_DIR / "cleaned_sentiment_reviews.csv"
    if not path.exists():
        return None, None
    df = pd.read_csv(path)
    if "sentiment" not in df.columns:
        return df, None
    distribution = df["sentiment"].value_counts().reindex(
        ["Negative", "Neutral", "Positive"], fill_value=0
    )
    return df, distribution


@st.cache_resource

def load_models():
    loaded = {}
    missing = []
    for display_name, filename in MODEL_FILES.items():
        path = BASE_DIR / filename
        if path.exists():
            loaded[display_name] = joblib.load(path)
        else:
            missing.append(filename)

    tfidf_path = BASE_DIR / "tfidf_vectorizer.pkl"
    encoder_path = BASE_DIR / "label_encoder.pkl"
    if not tfidf_path.exists() or not encoder_path.exists():
        raise FileNotFoundError("TF-IDF vectorizer or label encoder file is missing.")

    return loaded, joblib.load(tfidf_path), joblib.load(encoder_path), missing


try:
    models, tfidf, label_encoder, missing_models = load_models()
except Exception as exc:
    st.error(f"Unable to load the project artifacts: {exc}")
    st.stop()


dataset, distribution = load_dataset_info()

@st.cache_data
def build_eda_data(df):
    if df is None:
        return {}
    out = {
        "rows": len(df),
        "columns": len(df.columns),
        "duplicates": int(df.duplicated().sum()),
    }

    text_cols = [c for c in df.columns if "review" in c.lower() or "text" in c.lower()]
    out["text_col"] = text_cols[0] if text_cols else None
    if out["text_col"]:
        lengths = df[out["text_col"]].fillna("").astype(str).str.split().str.len()
        lengths = lengths[lengths > 0]
        out["lengths"] = lengths

    if "sentiment" in df.columns:
        out["sentiment"] = df["sentiment"].value_counts().reindex(
            ["Negative", "Neutral", "Positive"], fill_value=0
        )

    rating_cols = [c for c in df.columns if c.lower() in {"rating", "ratings", "score", "stars"}]
    out["rating_col"] = rating_cols[0] if rating_cols else None
    if out["rating_col"]:
        r = pd.to_numeric(df[out["rating_col"]], errors="coerce").dropna()
        out["ratings"] = r.value_counts().sort_index()

    out["missing"] = df.isna().sum()
    return out

eda = build_eda_data(dataset)


# ============================================================
# Session state
# ============================================================
if "history" not in st.session_state:
    st.session_state.history = []


# ============================================================
# Sidebar
# ============================================================
with st.sidebar:
    st.markdown("## Sentiment Analysis")
    st.caption("Sentiment Analysis")
    st.divider()

    st.markdown("### 🔮 Prediction model")
    available_model_names = [name for name in MODEL_FILES if name in models]
    selected_model = st.selectbox(
        "Choose a trained model",
        available_model_names,
        index=available_model_names.index("XGBoost") if "XGBoost" in available_model_names else 0,
        help="All models shown here are trained model artifacts. XGBoost is selected by default because it had the highest weighted F1-score in the project evaluation.",
    )

    st.divider()
    st.markdown("### 🧰 Technology")
    for item in ["Python", "NLTK", "TF-IDF", "Scikit-learn", "XGBoost", "Streamlit"]:
        st.markdown(f"<span class='pill'>{item}</span>", unsafe_allow_html=True)

    st.divider()
    st.markdown("### 📌 Project")
    st.write("**Task:** 3-class sentiment classification")
    st.write("**Domain:** Smartphone/customer reviews")
    st.write("**Classes:** Positive, Neutral, Negative")

    if missing_models:
        st.warning("Missing model files: " + ", ".join(missing_models))


# ============================================================
# Hero
# ============================================================
st.markdown(
    """
    <div class="hero">
        <h1>Sentiment Analysis</h1>
        <p>Analyze customer feedback using NLP, TF-IDF feature extraction and five machine-learning classifiers.</p>
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# Overview metrics
# ============================================================
metric_cols = st.columns(5)
metric_cols[0].metric("🤖 Models", str(len(models)))
metric_cols[1].metric("🎯 Best F1", "77.95%")
metric_cols[2].metric("📈 Best Accuracy", "80.21%")
metric_cols[3].metric("🔤 TF-IDF", "5,000")
metric_cols[4].metric("⭐ Classes", "3")

st.divider()


# ============================================================
# Main tabs
# ============================================================
tab_analyze, tab_compare, tab_eda, tab_explore, tab_method, tab_about = st.tabs(['🔮 Analyze Review', '📊 Model Lab', '📈 EDA Dashboard', '📜 History & Data', '🔬 Methodology', 'ℹ️ About'])


# ============================================================
# Analyze Review
# ============================================================
with tab_analyze:
    left, right = st.columns([1.55, 1], gap="large")

    with left:
        st.subheader("Analyze a customer review")
        st.caption(f"Selected model: **{selected_model}**")

        review = st.text_area(
            "Customer review",
            height=190,
            placeholder=(
                "Example: The battery life is excellent and the camera quality is amazing. "
                "I am very happy with this phone."
            ),
            label_visibility="visible",
        )

        b1, b2 = st.columns([2, 1])
        with b1:
            predict_clicked = st.button(
                "🚀 Analyze Sentiment",
                type="primary",
                use_container_width=True,
            )
        with b2:
            clear_clicked = st.button("Clear", use_container_width=True)

        if clear_clicked:
            st.rerun()

        if predict_clicked:
            if not review.strip():
                st.warning("Please enter a customer review.")
            else:
                cleaned = preprocess_text(review)

                if not cleaned.strip():
                    st.warning("The review became empty after preprocessing. Please enter more descriptive text.")
                else:
                    vector = tfidf.transform([cleaned])
                    model = models[selected_model]
                    encoded_prediction = model.predict(vector)[0]
                    sentiment = label_encoder.inverse_transform([encoded_prediction])[0]

                    confidence = None
                    decision_score = None
                    if selected_model == "SVM":
                        # LinearSVC does not expose probabilities. Use its
                        # decision function and label it as a decision score.
                        decision_scores = model.decision_function(vector)
                        decision_score = float(decision_scores.max())
                    elif hasattr(model, "predict_proba"):
                        probabilities = model.predict_proba(vector)[0]
                        confidence = float(probabilities.max())

                    if sentiment == "Positive":
                        emoji, css, message = "😊", "positive", "Positive customer feedback"
                    elif sentiment == "Negative":
                        emoji, css, message = "😞", "negative", "Negative customer feedback"
                    else:
                        emoji, css, message = "😐", "neutral", "Neutral customer feedback"

                    st.markdown(
                        f"""
                        <div class="result-card {css}">
                            <div class="result-label">{emoji} {sentiment.upper()}</div>
                            <div class="result-sub">{message}</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                    info1, info2 = st.columns(2, gap="medium")
                    if selected_model == "SVM":
                        second_label = "Decision Score"
                        second_value = f"{decision_score:.4f}"
                    else:
                        second_label = "Model Confidence"
                        second_value = f"{confidence * 100:.1f}%"

                    with info1:
                        st.markdown(
                            f"""
                            <div class="model-info-card">
                                <div class="model-info-label">Model Used</div>
                                <div class="model-info-value">{selected_model}</div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
                    with info2:
                        st.markdown(
                            f"""
                            <div class="model-info-card">
                                <div class="model-info-label">{second_label}</div>
                                <div class="model-info-value">{second_value}</div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )


                    st.session_state.history.insert(
                        0,
                        {
                            "Review": review.strip(),
                            "Model": selected_model,
                            "Prediction": sentiment,
                            "Score": (
                                f"{decision_score:.4f}" if selected_model == "SVM"
                                else f"{confidence * 100:.1f}%"
                            ),
                            "Score Type": "Decision Score" if selected_model == "SVM" else "Confidence",
                        },
                    )

                    with st.expander("View preprocessing"):
                        st.code(cleaned)

    with right:
        st.subheader("How it works")
        st.markdown(
            """
            **1. Review input**  
            Customer enters a natural-language review.

            **2. Preprocessing**  
            Tokenization, stopword handling, negation preservation and lemmatization.

            **3. TF-IDF**  
            Converts cleaned text into 5,000 unigram/bigram features.

            **4. Model**  
            The selected trained classifier generates the sentiment.

            **5. Result**  
            Positive, Neutral or Negative.
            """
        )

        st.info(
            "Tip: Change the model in the sidebar and analyze the same review to compare how the five classifiers behave."
        )



# ============================================================
# EDA Dashboard
# ============================================================
with tab_eda:
    st.subheader("📈 EDA Dashboard")
    st.caption("Power BI-style overview of the cleaned customer-review dataset.")

    if dataset is None:
        st.warning("EDA is unavailable because cleaned_sentiment_reviews.csv was not found.")
    else:
        # KPI cards
        c1, c2 = st.columns(2)
        c1.metric("Total Reviews", f"{eda['rows']:,}")
        c2.metric("Features", f"{eda['columns']:,}")

        st.divider()


        # Main visual dashboard
        left, right = st.columns(2, gap="large")

        with left:
            st.markdown("### 🎯 Sentiment Distribution")
            if "sentiment" in eda:
                sent = eda["sentiment"]
                st.bar_chart(sent, height=300)
                pct = (sent / sent.sum() * 100).round(2)
                st.dataframe(
                    pd.DataFrame({"Reviews": sent, "Percentage": pct}),
                    use_container_width=True,
                    hide_index=True,
                )
            else:
                st.info("No sentiment column found.")

        with right:
            st.markdown("### 📋 Dataset Information")
            st.markdown(
                f"""
                <div class="card" style="padding:1.25rem 1.4rem; min-height:300px;">
                    <div style="font-size:.92rem; line-height:2;">
                        <b>Rows:</b> {len(dataset):,}<br>
                        <b>Columns:</b> {len(dataset.columns):,}<br>
                        <b>Review column:</b> {eda.get('text_col') or 'Not detected'}<br>
                        <b>Sentiment column:</b> {'sentiment' if 'sentiment' in dataset.columns else 'Not detected'}<br>
                        <b>Rating column:</b> {eda.get('rating_col') or 'Not detected'}<br>
                        <b>Duplicates:</b> {eda['duplicates']:,}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        if eda.get("rating_col"):
            st.divider()
            st.markdown(f"### ⭐ Rating Distribution — {eda['rating_col']}")
            st.bar_chart(eda["ratings"])

        st.divider()




        with st.expander("🔎 View sample data"):
            # Keep the preview compact and readable.
            preferred_columns = [
                col for col in ["review", "sentiment", "rating"]
                if col in dataset.columns
            ]

            if preferred_columns:
                sample_data = dataset[preferred_columns].head(10).copy()

                column_config = {}
                if "review" in sample_data.columns:
                    column_config["review"] = st.column_config.TextColumn(
                        "Review",
                        width="large",
                        help="Original customer review",
                    )
                if "sentiment" in sample_data.columns:
                    column_config["sentiment"] = st.column_config.TextColumn(
                        "Sentiment",
                        width="medium",
                    )
                if "rating" in sample_data.columns:
                    column_config["rating"] = st.column_config.NumberColumn(
                        "Rating",
                        width="small",
                    )

                st.dataframe(
                    sample_data,
                    width="stretch",
                    height=360,
                    hide_index=True,
                    column_config=column_config,
                )
            else:
                st.dataframe(
                    dataset.head(10),
                    width="stretch",
                    height=360,
                    hide_index=True,
                )


# ============================================================
# Model Lab
# ============================================================
with tab_compare:
    st.subheader("📊 Model Lab")
    st.write("Compare the five models using the evaluation results recorded in your project.")

    display_df = MODEL_RESULTS.copy()
    for col in ["Accuracy", "Precision", "Recall", "F1-Score"]:
        display_df[col] = (display_df[col] * 100).round(2).astype(str) + "%"

    st.dataframe(display_df, use_container_width=True, hide_index=True)

    st.subheader("Performance comparison")
    chart_df = MODEL_RESULTS.set_index("Model")[["Accuracy", "F1-Score"]] * 100
    st.bar_chart(chart_df)

    st.subheader("🧪 Compare all models on one review")
    compare_review = st.text_area(
        "Enter a review to compare model predictions",
        height=130,
        placeholder="Example: The camera is excellent but the battery life is disappointing.",
        key="compare_review",
    )

    if st.button("Compare All 5 Models", use_container_width=True):
        if not compare_review.strip():
            st.warning("Enter a review first.")
        else:
            cleaned = preprocess_text(compare_review)
            if not cleaned:
                st.warning("The review became empty after preprocessing.")
            else:
                vector = tfidf.transform([cleaned])
                comparison_rows = []
                for name in MODEL_FILES:
                    if name not in models:
                        continue
                    m = models[name]
                    pred = m.predict(vector)[0]
                    label = label_encoder.inverse_transform([pred])[0]
                    if name == "SVM":
                        decision_scores = m.decision_function(vector)
                        score_text = f"{float(decision_scores.max()):.4f}"
                        score_type = "Decision Score"
                    elif hasattr(m, "predict_proba"):
                        conf = float(m.predict_proba(vector)[0].max())
                        score_text = f"{conf * 100:.1f}%"
                        score_type = "Confidence"
                    else:
                        score_text = "N/A"
                        score_type = "Score"
                    comparison_rows.append([name, label, score_text, score_type])

                st.dataframe(
                    pd.DataFrame(comparison_rows, columns=["Model", "Prediction", "Score", "Score Type"]),
                    use_container_width=True,
                    hide_index=True,
                )

    st.success("XGBoost achieved the highest weighted F1-score in the project's baseline evaluation: 77.95%.")


# ============================================================
# History & Data
# ============================================================
with tab_explore:
    st.subheader("📜 Current-session prediction history")

    if st.session_state.history:
        history_df = pd.DataFrame(st.session_state.history)
        st.dataframe(history_df, use_container_width=True, hide_index=True)

        if st.button("Clear prediction history"):
            st.session_state.history = []
            st.rerun()
    else:
        st.info("Your prediction history will appear here after you analyze reviews.")

    st.divider()
    st.subheader("📦 Dataset overview")

    if dataset is not None:
        c1, c2, c3 = st.columns(3)
        c1.metric("Reviews", f"{len(dataset):,}")
        c2.metric("Training / Test", "80% / 20%")
        c3.metric("Sentiment Classes", "3")

        if distribution is not None:
            dist_df = distribution.rename("Reviews").to_frame()
            dist_df["Percentage"] = (dist_df["Reviews"] / dist_df["Reviews"].sum() * 100).round(2)
            st.dataframe(dist_df, use_container_width=True)
            st.bar_chart(dist_df[["Percentage"]])
    else:
        st.info("Dataset summary is unavailable because the cleaned CSV was not found.")


# ============================================================
# Methodology
# ============================================================
with tab_method:
    st.subheader("🔬 End-to-end methodology")

    steps = [
        ("01", "Data Collection", "Amazon/customer smartphone reviews"),
        ("02", "Sentiment Labelling", "Rating ≤2: Negative | 3: Neutral | ≥4: Positive"),
        ("03", "Preprocessing", "Tokenization, stopword handling, negation preservation and lemmatization"),
        ("04", "Train-Test Split", "80:20 split with stratification"),
        ("05", "Feature Extraction", "TF-IDF, 5,000 max features, unigram + bigram"),
        ("06", "Model Training", "Logistic Regression, Naive Bayes, SVM, Random Forest, XGBoost"),
        ("07", "Evaluation", "Accuracy, weighted Precision, weighted Recall, weighted F1"),
        ("08", "Deployment", "Interactive Streamlit application"),
    ]

    for number, title, description in steps:
        st.markdown(f"**{number} · {title}**  ")
        st.caption(description)

    st.subheader("Pipeline")
    st.code(
        "Customer Review\n"
        "      ↓\n"
        "Text Preprocessing\n"
        "      ↓\n"
        "TF-IDF (Unigram + Bigram)\n"
        "      ↓\n"
        "5 Machine Learning Models\n"
        "      ↓\n"
        "Evaluation & Comparison\n"
        "      ↓\n"
        "Interactive Model Selection\n"
        "      ↓\n"
        "Sentiment Prediction"
    )


# ============================================================
# About
# ============================================================
with tab_about:
    st.subheader("ℹ️ About this project")
    st.write(
        "Sentiment Analysis is an NLP and machine-learning project designed to "
        "automatically classify customer feedback into Positive, Neutral and Negative sentiment. "
        "The application provides both model comparison and interactive prediction."
    )

    st.subheader("🎯 Objective")
    st.write(
        "Build a practical sentiment classification system that can transform unstructured customer "
        "reviews into an interpretable sentiment signal."
    )

    st.subheader("🧠 Models")
    st.write(", ".join(MODEL_FILES.keys()))

    st.subheader("📌 Deployment note")
    st.write(
        "The deployed prediction path applies the same preprocessing and TF-IDF configuration used "
        "in the project before passing the review to the selected trained model."
    )


st.markdown(
    '<div class="footer">Sentiment Analysis · Data Science Major Project</div>',
    unsafe_allow_html=True,
)
