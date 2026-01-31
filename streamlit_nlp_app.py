"""
Streamlit NLP Toolkit
- Tabs interface (chosen by user)
- Features: Preprocessing, Vocabulary, BOW, TF-IDF, Word2Vec, Naive Bayes

Run:
    pip install -r requirements.txt
    streamlit run streamlit_nlp_app.py

requirements.txt (suggested):
streamlit
nltk
scikit-learn
gensim
pandas
numpy
"""

import streamlit as st
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from gensim.models import Word2Vec
import pandas as pd
import numpy as np
import io
import warnings

warnings.filterwarnings("ignore")

# Ensure required NLTK data is available
nltk_packages = ["punkt", "stopwords", "wordnet", "omw-1.4"]
for pkg in nltk_packages:
    try:
        nltk.data.find(pkg)
    except Exception:
        nltk.download(pkg)

# Globals
STOP_WORDS = set(stopwords.words("english"))
STEMMER = PorterStemmer()
LEMMATIZER = WordNetLemmatizer()

st.set_page_config(page_title="NLP Toolkit — Streamlit", layout="wide")
st.title("NLP Toolkit — Streamlit (Tabbed Interface)")
st.markdown("Choose a tab to perform different NLP operations.\n\nMade interactive with buttons and options.")

# Helper functions
@st.cache(allow_output_mutation=True)
def preprocess(text, remove_stopwords=True, do_stem=False, do_lem=False):
    if text is None:
        return []
    text = text.lower()
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t.isalpha()]
    if remove_stopwords:
        tokens = [t for t in tokens if t not in STOP_WORDS]
    if do_stem:
        tokens = [STEMMER.stem(t) for t in tokens]
    if do_lem:
        tokens = [LEMMATIZER.lemmatize(t) for t in tokens]
    return tokens

@st.cache(allow_output_mutation=True)
def build_vocab(docs):
    vocab = set()
    for d in docs:
        vocab.update(preprocess(d, remove_stopwords=True, do_stem=False, do_lem=False))
    return sorted(vocab)

@st.cache(allow_output_mutation=True)
def get_bow(docs):
    vec = CountVectorizer()
    X = vec.fit_transform(docs)
    df = pd.DataFrame(X.toarray(), columns=vec.get_feature_names_out())
    return vec, df

@st.cache(allow_output_mutation=True)
def get_tfidf(docs):
    vec = TfidfVectorizer()
    X = vec.fit_transform(docs)
    df = pd.DataFrame(X.toarray(), columns=vec.get_feature_names_out())
    return vec, df

@st.cache(allow_output_mutation=True)
def train_word2vec(docs, vector_size=50, window=5, min_count=1, epochs=50):
    tokenized = [preprocess(d, remove_stopwords=True, do_stem=False, do_lem=False) for d in docs]
    model = Word2Vec(sentences=tokenized, vector_size=vector_size, window=window, min_count=min_count, epochs=epochs)
    return model

@st.cache(allow_output_mutation=True)
def train_naive_bayes(docs, labels):
    tf = TfidfVectorizer()
    X = tf.fit_transform(docs)
    X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42)
    model = MultinomialNB()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    return tf, model, acc

# Tabs
tabs = st.tabs(["Preprocessing", "Vocabulary", "Bag of Words (BOW)", "TF-IDF", "Word2Vec", "Naive Bayes"])

# -------------------- PREPROCESSING TAB --------------------
with tabs[0]:
    st.header("Text Preprocessing")
    st.write("Lower-casing, Tokenization, Stop-word removal, Stemming, Lemmatization")

    input_text = st.text_area("Enter text to preprocess", height=150)
    col1, col2, col3 = st.columns(3)
    with col1:
        remove_sw = st.checkbox("Remove Stopwords", value=True)
        do_stem = st.checkbox("Apply Stemming (Porter)", value=False)
    with col2:
        do_lem = st.checkbox("Apply Lemmatization", value=False)
        show_tokens = st.checkbox("Show tokens table", value=True)
    with col3:
        preview = st.button("Run Preprocessing")

    if preview:
        if input_text.strip() == "":
            st.error("Please enter some text to preprocess.")
        else:
            tokens = preprocess(input_text, remove_stopwords=remove_sw, do_stem=do_stem, do_lem=do_lem)
            st.success("Preprocessing complete — results below")
            if show_tokens:
                df = pd.DataFrame({"token": tokens})
                st.dataframe(df)
            st.write("Re-joined text:", " ".join(tokens))

# -------------------- VOCAB TAB --------------------
with tabs[1]:
    st.header("Vocabulary Builder")
    st.write("Provide multiple documents (one per line) or upload a .txt file with one document per line.")

    docs_input = st.text_area("Enter documents (one doc per line)")
    uploaded_file = st.file_uploader("Or upload a .txt file", type=["txt"])
    build_btn = st.button("Build Vocabulary")

    docs = []
    if uploaded_file is not None:
        try:
            content = uploaded_file.read().decode("utf-8")
            docs = [line.strip() for line in content.splitlines() if line.strip()]
        except Exception:
            st.error("Failed to read uploaded file. Make sure it's a UTF-8 encoded text file.")
    elif docs_input.strip() != "":
        docs = [line.strip() for line in docs_input.splitlines() if line.strip()]

    if build_btn:
        if not docs:
            st.error("No documents provided — type or upload some documents first.")
        else:
            vocab = build_vocab(docs)
            st.write(f"Vocabulary size: {len(vocab)}")
            st.dataframe(pd.DataFrame({"vocab": vocab}))

# -------------------- BOW TAB --------------------
with tabs[2]:
    st.header("Bag of Words (BoW)")
    st.write("Input documents (one per line). The app will display feature names and the BoW matrix.")

    docs_bow = st.text_area("Enter documents (one per line)")
    run_bow = st.button("Generate BoW")

    if run_bow:
        docs = [line.strip() for line in docs_bow.splitlines() if line.strip()]
        if len(docs) == 0:
            st.error("Please enter at least one document.")
        else:
            vec, df_bow = get_bow(docs)
            st.write("Feature names (vocabulary):")
            st.write(vec.get_feature_names_out())
            st.write("BoW matrix:")
            st.dataframe(df_bow)

# -------------------- TF-IDF TAB --------------------
with tabs[3]:
    st.header("TF-IDF Representation")
    st.write("Input documents (one per line). The app will compute TF-IDF and show the matrix.")

    docs_tfidf = st.text_area("Enter documents (one per line)")
    run_tfidf = st.button("Generate TF-IDF")

    if run_tfidf:
        docs = [line.strip() for line in docs_tfidf.splitlines() if line.strip()]
        if len(docs) == 0:
            st.error("Please enter at least one document.")
        else:
            vec, df_tfidf = get_tfidf(docs)
            st.write("TF-IDF feature names:")
            st.write(vec.get_feature_names_out())
            st.write("TF-IDF matrix:")
            st.dataframe(df_tfidf)

# -------------------- WORD2VEC TAB --------------------
with tabs[4]:
    st.header("Word2Vec — Word Embeddings")
    st.write("Train a small Word2Vec model on your documents and inspect word vectors / similar words.")

    docs_w2v = st.text_area("Enter documents (one per line)")
    emb_size = st.number_input("Vector size", min_value=10, max_value=300, value=50)
    w_window = st.number_input("Window size", min_value=2, max_value=10, value=5)
    min_count = st.number_input("min_count", min_value=1, max_value=5, value=1)
    epochs = st.number_input("epochs", min_value=1, max_value=500, value=50)
    run_w2v = st.button("Train Word2Vec")

    if run_w2v:
        docs = [line.strip() for line in docs_w2v.splitlines() if line.strip()]
        if len(docs) == 0:
            st.error("Please enter some documents to train on.")
        else:
            with st.spinner("Training Word2Vec (this may take a few seconds)..."):
                model = train_word2vec(docs, vector_size=emb_size, window=w_window, min_count=min_count, epochs=epochs)
            st.success("Model trained")

            word = st.text_input("Enter a word to inspect vector/similar words")
            if word:
                if word in model.wv:
                    vec = model.wv[word]
                    st.write(f"Vector for '{word}' (first 10 dims):")
                    st.write(vec[:10])
                    st.write("Top similar words:")
                    sims = model.wv.most_similar(word, topn=10)
                    st.table(pd.DataFrame(sims, columns=["word", "similarity"]))
                else:
                    st.warning("Word not in vocabulary — try training on more documents or different words.")

# -------------------- NAIVE BAYES TAB --------------------
with tabs[5]:
    st.header("Naive Bayes Text Classification & Phrase Probability")
    st.write("Train a Multinomial Naive Bayes classifier using TF-IDF features. Then predict class and probability for a given phrase.")

    nb_docs = st.text_area("Training documents (one per line)")
    nb_labels = st.text_area("Labels (one per line, same order as documents)")
    test_phrase = st.text_input("Enter a word/phrase to predict")
    split_train = st.slider("Test size (fraction for testing)", min_value=0.05, max_value=0.5, value=0.2, step=0.05)
    run_nb = st.button("Train & Predict")

    if run_nb:
        docs = [line.strip() for line in nb_docs.splitlines() if line.strip()]
        labels = [line.strip() for line in nb_labels.splitlines() if line.strip()]
        if len(docs) == 0 or len(labels) == 0:
            st.error("Please provide both documents and labels.")
        elif len(docs) != len(labels):
            st.error("Number of documents and labels must match.")
        else:
            # Train using provided test size
            tf = TfidfVectorizer()
            X = tf.fit_transform(docs)
            X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=split_train, random_state=42)
            model = MultinomialNB()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            st.success(f"Model trained — test accuracy: {acc:.3f}")

            if test_phrase.strip() == "":
                st.info("Enter a phrase to predict its class and class probabilities.")
            else:
                vec = tf.transform([test_phrase])
                pred = model.predict(vec)[0]
                probs = model.predict_proba(vec)[0]
                classes = model.classes_
                dfp = pd.DataFrame({"class": classes, "probability": probs})
                st.write(f"Predicted class: {pred}")
                st.dataframe(dfp.sort_values(by="probability", ascending=False).reset_index(drop=True))

# Footer
st.markdown("---")
st.caption("Tip: For larger datasets or production usage, replace the toy training and vectorizers with saved pipelines and more data. This app is for learning and small experiments.")
