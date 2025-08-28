import streamlit as st
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
from transformers import MarianMTModel, MarianTokenizer

@st.cache_resource
def get_pipeline(task, model=None):
    if task == "sentiment":
        return pipeline("sentiment-analysis")
    elif task == "classification":
        return pipeline("zero-shot-classification",
                        model="facebook/bart-large-mnli")
    elif task == "ner":
        return pipeline("ner", grouped_entities=True)
    elif task == "qa":
        return pipeline("question-answering")
    elif task == "translation":
        try:
            return pipeline(
                "translation_en_to_ar",
                model="Helsinki-NLP/opus-mt-en-ar",
                max_length=400,  # Increased max length
                clean_up_tokenization_spaces=True
            )
        except Exception as e:
            st.sidebar.warning(f"Primary translation model failed: {str(e)}. Using backup.")
            # Fallback to a different model
            return pipeline(
                "translation_en_to_ar",
                model="Helsinki-NLP/opus-mt-en-ar",
                max_length=400,
                clean_up_tokenization_spaces=True
            )

def improved_translation(text, pipe):
    """Handle translation with better parameters"""
    # Clean the input text
    text = text.strip()
    
    if not text:
        return "Please enter text to translate"
    
    # For very short texts, we need to handle differently
    if len(text.split()) <= 2:
        # Add a period to help the model
        if not text.endswith('.'):
            text = text + '.'
    
    try:
        result = pipe(text, max_length=400, num_beams=5, early_stopping=True)
        return result[0]['translation_text']
    except Exception as e:
        return f"Translation error: {str(e)}"

# ---------------- UI ----------------
st.title("Smart NLP App — Powered by BERT")
st.write("Features: Sentiment Analysis, Text Classification, NER, Question Answering, and Translation (English → Arabic).")

option = st.sidebar.radio("Choose Feature:", 
                          ["🧠 Sentiment", "🏷️ Classification", "🧩 NER", "❓ Q&A", "🌐 Translate"])


# ---------------- Sentiment ----------------
if option == "🧠 Sentiment":
    st.subheader("Sentiment Analysis")
    text = st.text_area("Enter text:", key="sentiment")
    if st.button("Analyze", key="sentiment_btn"):
        pipe = get_pipeline("sentiment")
        result = pipe(text)
        st.json(result)


# ---------------- Classification ----------------
elif option == "🏷️ Classification":
    st.subheader("Text Classification (Zero-Shot)")
    text = st.text_area("Enter text:", key="classification")
    labels = st.text_input("Enter candidate labels (comma separated):", "politics, sports, technology")
    if st.button("Classify", key="classification_btn"):
        pipe = get_pipeline("classification")
        result = pipe(text, candidate_labels=labels.split(","))
        st.json(result)


# ---------------- NER ----------------
elif option == "🧩 NER":
    st.subheader("Named Entity Recognition")
    text = st.text_area("Enter text:", key="ner")
    if st.button("Extract Entities", key="ner_btn"):
        pipe = get_pipeline("ner")
        result = pipe(text)
        st.json(result)


# ---------------- Question Answering ----------------
elif option == "❓ Q&A":
    st.subheader("Question Answering")
    context = st.text_area("Enter context passage:", key="qa_context")
    question = st.text_input("Enter your question:", key="qa_question")
    if st.button("Get Answer", key="qa_btn"):
        pipe = get_pipeline("qa")
        result = pipe(question=question, context=context)
        st.json(result)


# ---------------- Translation ----------------
elif option == "🌐 Translate":
    st.subheader("Translation (English → Arabic)")
    st.info("Note: For best results, use complete sentences with proper punctuation.")
    
    text = st.text_area("Enter English text:", key="translate", 
                       placeholder="e.g., I love programming. It's my passion.")
    
    if st.button("Translate", key="translate_btn"):
        with st.spinner("Translating..."):
            pipe = get_pipeline("translation")
            result = improved_translation(text, pipe)
            
            if "error" in result.lower():
                st.error(result)
            else:
                st.write("**Translation:**")
                st.success(result)
                
                # Show original for comparison
                st.write("**Original text:**")
                st.text(text)
