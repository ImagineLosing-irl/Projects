import os
import warnings
import streamlit as st
from transformers import pipeline
from nltk.tokenize import sent_tokenize
import nltk
from concurrent.futures import ThreadPoolExecutor

# -----------------------------
# Suppress warnings and logs
# -----------------------------
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'   # hide TensorFlow info/warnings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # disable oneDNN messages
warnings.filterwarnings("ignore")          # hide deprecation warnings
warnings.filterwarnings("ignore", category=UserWarning)  # suppress HF pipeline warnings

# -----------------------------
# Streamlit Page Setup
# -----------------------------
st.set_page_config(page_title="Fast AI Text Summarizer", layout="centered")
st.title("🧠 Fast AI Text Summarization App")
st.write("📄 Paste long text below and get a concise summary instantly!")

# -----------------------------
# Load Summarization Model
# -----------------------------
@st.cache_resource
def load_model():
    # Use faster T5-small model for CPU
    return pipeline("summarization", model="t5-small", framework="pt")

summarizer = load_model()

# -----------------------------
# Summarization Function
# -----------------------------
def summarize_text_parallel(text, max_chunk_words=300, max_new_tokens_chunk=100):
    """
    Summarizes the input text in parallel using chunks for faster CPU execution.
    """
    sentences = sent_tokenize(text)
    chunks, current_chunk, current_length = [], [], 0

    # Split text into manageable chunks
    for sentence in sentences:
        words = sentence.split()
        current_length += len(words)
        current_chunk.append(sentence)
        if current_length >= max_chunk_words:
            chunks.append(" ".join(current_chunk))
            current_chunk, current_length = [], 0
    if current_chunk:
        chunks.append(" ".join(current_chunk))

    # Summarize chunks in parallel
    with ThreadPoolExecutor() as executor:
        results = list(executor.map(
            lambda c: summarizer(
                c, 
                max_new_tokens=max_new_tokens_chunk, 
                do_sample=False
            )[0]['summary_text'],
            chunks
        ))

    # Combine chunk summaries into final summary
    final_summary = " ".join(results)
    return final_summary

# -----------------------------
# User Input
# -----------------------------
input_text = st.text_area(
    "📝 Enter text to summarize",
    height=300,
    placeholder="Paste articles, essays, or research papers here..."
)

# -----------------------------
# User chooses summary length
# -----------------------------
summary_length = st.selectbox(
    "📏 Choose summary length",
    ["Short", "Medium", "Long"],
    index=1
)

# Map length to chunk and max_new_tokens parameters
length_map = {
    "Long": (250, 80),
    "Medium": (300, 100),
    "Short": (400, 120)
}

max_chunk_words, max_new_tokens_chunk = length_map[summary_length]

# -----------------------------
# Generate Summary
# -----------------------------
if st.button("Summarize"):
    if len(input_text.strip()) < 50:
        st.warning("⚠️ Please enter at least 50 characters.")
    else:
        with st.spinner("Generating summary..."):
            summary = summarize_text_parallel(
                input_text,
                max_chunk_words=max_chunk_words,
                max_new_tokens_chunk=max_new_tokens_chunk
            )
        st.subheader("📌 Summary")
        st.success("✅ Summary generated successfully!")
        st.write(summary)