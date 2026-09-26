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
    page_title="ReviewSense | Customer Sentiment AI",
    page_icon="💬",
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
st.markdown(
    """
    <style>
    .block-container {padding-top: 1.5rem; padding-bottom: 2rem;}

    .hero {
        padding: 2rem 2.2rem;
        border-radius: 22px;
        border: 1px solid rgba(128,128,128,.20);
        background: linear-gradient(135deg, rgba(99,102,241,.13), rgba(14,165,233,.08));
        margin-bottom: 1.2rem;
    }
    .hero h1 {font-size: 2.6rem; margin: 0; font-weight: 800;}
    .hero p {font-size: 1.05rem; margin: .5rem 0 0; opacity: .82;}

    .result-card {
        padding: 1.4rem;
        border-radius: 18px;
        text-align: center;
        border: 1px solid rgba(128,128,128,.22);
        margin: .8rem 0 1rem;
    }
    .positive {border-left: 7px solid #16a34a;}
    .negative {border-left: 7px solid #dc2626;}
    .neutral {border-left: 7px solid #2563eb;}
    .result-label {font-size: 2rem; font-weight: 800;}
    .result-sub {opacity: .78; margin-top: .3rem;}

    .pill {
        display: inline-block;
        padding: .25rem .7rem;
        border-radius: 999px;
        border: 1px solid rgba(128,128,128,.25);
        font-size: .85rem;
        margin-right: .35rem;
    }

    .footer {
        text-align: center;
        opacity: .65;
        padding: 1.5rem 0 .5rem;
        font-size: .85rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


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


# ============================================================
# Session state
# ============================================================
if "history" not in st.session_state:
    st.session_state.history = []


# ============================================================
# Sidebar
# ============================================================
with st.sidebar:
    st.markdown("## 💬 ReviewSense")
    st.caption("Customer Review Sentiment Analysis")
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
        <h1>💬 Customer Review Sentiment Analysis</h1>
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
tab_analyze, tab_compare, tab_explore, tab_method, tab_about = st.tabs(
    ["🔮 Analyze Review", "📊 Model Lab", "📜 History & Data", "🔬 Methodology", "ℹ️ About"]
)


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
                        emoji, css, message = "😐", "neutral", "Neutral or mixed customer feedback"

                    st.markdown(
                        f"""
                        <div class="result-card {css}">
                            <div class="result-label">{emoji} {sentiment.upper()}</div>
                            <div class="result-sub">{message}</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                    info1, info2 = st.columns(2)
                    info1.metric("Model Used", selected_model)
                    if selected_model == "SVM":
                        info2.metric("Decision Score", f"{decision_score:.4f}")
                    else:
                        info2.metric("Model Confidence", f"{confidence * 100:.1f}%")

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
        "Customer Review Sentiment Analysis is an NLP and machine-learning project designed to "
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
    '<div class="footer">Customer Review Sentiment Analysis · Data Science Major Project</div>',
    unsafe_allow_html=True,
)
