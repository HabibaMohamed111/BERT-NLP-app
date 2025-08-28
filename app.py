import streamlit as st
from transformers import pipeline
# from PyPDF2 import PdfReader   # Commented out because PDF upload is disabled

# ----------------------------
# Page setup
# ----------------------------
st.set_page_config(
    page_title="Smart NLP App ✨",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------------------
# Custom CSS
# ----------------------------
CUSTOM_CSS = """
<style>
.main > div {padding-top: 1.2rem;}
.block-container {max-width: 1200px;}
.gradient-title {
  font-weight: 800;
  font-size: 2.2rem;
  background: linear-gradient(90deg, #7c3aed 0%, #22d3ee 50%, #10b981 100%);
  -webkit-background-clip: text; background-clip: text; color: transparent;
}
.card {background: rgba(255,255,255,0.08); border-radius: 18px; padding: 1rem 1.1rem;}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

st.markdown('<div class="gradient-title">Smart NLP App — Powered by BERT</div>', unsafe_allow_html=True)
st.caption("Features: Sentiment Analysis, Text Classification, Named Entities, Question Answering, and Translation.")

# ----------------------------
# Cached pipelines
# ----------------------------
@st.cache_resource(show_spinner=False)
def get_pipeline(task: str, model: str | None = None):
    if model:
        return pipeline(task, model=model)
    return pipeline(task)

# ----------------------------
# Tabs
# ----------------------------
T1, T2, T3, T4, T6 = st.tabs([
    "🧠 Sentiment", "🏷️ Classification", "🧩 NER", "❓ Q&A", "🌐 Translate"
    # Removed Summarize & PDF Upload
])

# ----------------------------
# 1) Sentiment
# ----------------------------
with T1:
    st.subheader("Sentiment Analysis")
    txt = st.text_area("Enter text", key="sentiment_text")

    if st.button("Analyze Sentiment"):
        if txt.strip():
            pipe = get_pipeline("sentiment-analysis")
            st.json(pipe(txt))

# ----------------------------
# 2) Classification
# ----------------------------
with T2:
    st.subheader("Text Classification (Zero-Shot)")
    text = st.text_area("Enter text", key="classification_text")

    if st.button("Classify"):
        if text.strip():
            pipe = get_pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
            labels = ["News", "Politics", "Sports", "Technology", "Health", "Business", "Entertainment"]
            st.json(pipe(text, candidate_labels=labels))

# ----------------------------
# 3) NER
# ----------------------------
with T3:
    st.subheader("Named Entity Recognition (NER)")
    text = st.text_area("Enter text", key="ner_text")
    if st.button("Extract Entities"):
        if text.strip():
            pipe = get_pipeline("ner", model="dslim/bert-base-NER")
            st.json(pipe(text, grouped_entities=True))

# ----------------------------
# 4) Question Answering
# ----------------------------
with T4:
    st.subheader("Question Answering")
    context = st.text_area("Context", key="qa_context")
    question = st.text_area("Question", key="qa_question")

    if st.button("Get Answer"):
        if context.strip() and question.strip():
            pipe = get_pipeline("question-answering")
            st.json(pipe(question=question, context=context))

# ----------------------------
# 5) Translation
# ----------------------------
with T6:
    st.subheader("Translation (English → Arabic)")
    text = st.text_area("Enter English text", key="translation_text")

    if st.button("Translate"):
        if text.strip():
            pipe = get_pipeline("translation_en_to_ar", model="Helsinki-NLP/opus-mt-en-ar")
            st.json(pipe(text, max_length=200))

# ----------------------------
# Footer
# ----------------------------
st.markdown("<div class='card'>Made with ❤️ using Streamlit & Hugging Face Transformers.</div>", unsafe_allow_html=True)
