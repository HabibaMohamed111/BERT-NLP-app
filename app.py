import streamlit as st
from transformers import pipeline

# ---------------- Pipeline Loader ----------------
@st.cache_resource
def get_pipeline(task):
    if task == "sentiment":
        return pipeline("sentiment-analysis")  # صغير نسبيًا
    elif task == "classification":
        return pipeline("zero-shot-classification",
                        model="facebook/bart-large-mnli")  # كبير نسبيًا
    elif task == "ner":
        return pipeline("ner", grouped_entities=True)
    elif task == "qa":
        return pipeline("question-answering")
    elif task == "translation":
        return pipeline(
            "translation_en_to_ar",
            model="Helsinki-NLP/opus-mt-en-ar",
            max_length=100,
            clean_up_tokenization_spaces=True
        )

# ---------------- UI ----------------
st.title("Smart NLP App — Powered by BERT")
st.write("Features: Sentiment Analysis, Classification, NER, Q&A, Translation (English → Arabic)")

option = st.sidebar.radio("Choose Feature:", 
                          ["🧠 Sentiment", "🏷️ Classification", "🧩 NER", "❓ Q&A", "🌐 Translate"])

# ---------------- Sentiment ----------------
if option == "🧠 Sentiment":
    st.subheader("Sentiment Analysis")
    text = st.text_area("Enter text:", key="sentiment")
    if st.button("Analyze", key="sentiment_btn"):
        pipe = get_pipeline("sentiment")
        st.json(pipe(text))

# ---------------- Classification ----------------
elif option == "🏷️ Classification":
    st.subheader("Text Classification (Zero-Shot)")
    text = st.text_area("Enter text:", key="classification")
    labels = st.text_input("Candidate labels (comma separated):", "politics, sports, technology")
    if st.button("Classify", key="classification_btn"):
        pipe = get_pipeline("classification")
        st.json(pipe(text, candidate_labels=[x.strip() for x in labels.split(",")]))

# ---------------- NER ----------------
elif option == "🧩 NER":
    st.subheader("Named Entity Recognition")
    text = st.text_area("Enter text:", key="ner")
    if st.button("Extract Entities", key="ner_btn"):
        pipe = get_pipeline("ner")
        st.json(pipe(text))

# ---------------- Q&A ----------------
elif option == "❓ Q&A":
    st.subheader("Question Answering")
    context = st.text_area("Context passage:", key="qa_context")
    question = st.text_input("Enter your question:", key="qa_question")
    if st.button("Get Answer", key="qa_btn"):
        pipe = get_pipeline("qa")
        st.json(pipe(question=question, context=context))

# ---------------- Translation ----------------
elif option == "🌐 Translate":
    st.subheader("Translation (English → Arabic)")
    text = st.text_area("Enter English text:", key="translate")
    if st.button("Translate", key="translate_btn"):
        pipe = get_pipeline("translation")
        result = pipe(text)
        st.success(result[0]['translation_text'])
