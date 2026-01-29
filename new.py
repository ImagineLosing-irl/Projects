import streamlit as st
import nltk
import string
import numpy as np
import pandas as pd
import io
from collections import Counter, defaultdict
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

# Download required NLTK data with error handling
def download_nltk_data():
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')
    
    try:
        nltk.data.find('tokenizers/punkt_tab')
    except LookupError:
        try:
            nltk.download('punkt_tab')
        except:
            nltk.download('punkt')
    
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords')
    
    try:
        nltk.data.find('corpora/wordnet')
    except LookupError:
        nltk.download('wordnet')

# Download NLTK data at the start
download_nltk_data()

from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer

class NLPToolkit:
    def __init__(self):
        self.stemmer = PorterStemmer()
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        self.vocabulary = None
        self.vectorizer = None
        
    def text_preprocessing(self, text, method='lemmatization', remove_stopwords=True):
        """
        Preprocess text with lower-casing, tokenization, stop-word removal, and stemming/lemmatization
        """
        # Lower casing
        text = text.lower()
        
        # Tokenization with error handling
        try:
            tokens = word_tokenize(text)
        except LookupError:
            tokens = text.split()
        
        # Remove punctuation and numbers
        tokens = [token for token in tokens if token not in string.punctuation and not token.isdigit()]
        
        # Stop-word removal
        if remove_stopwords:
            tokens = [token for token in tokens if token not in self.stop_words]
        
        # Stemming or Lemmatization
        if method == 'stemming':
            tokens = [self.stemmer.stem(token) for token in tokens]
        elif method == 'lemmatization':
            tokens = [self.lemmatizer.lemmatize(token) for token in tokens]
        
        return tokens
    
    def create_vocabulary(self, documents, min_freq=1):
        """
        Create vocabulary from a list of documents
        """
        all_tokens = []
        for doc in documents:
            tokens = self.text_preprocessing(doc)
            all_tokens.extend(tokens)
        
        # Count word frequencies
        word_freq = Counter(all_tokens)
        
        # Create vocabulary with words meeting minimum frequency
        self.vocabulary = {word: idx for idx, (word, freq) in enumerate(word_freq.items()) 
                          if freq >= min_freq}
        
        return self.vocabulary
    
    def bow_representation(self, documents):
        """
        Create Bag-of-Words representation
        """
        if self.vectorizer is None:
            self.vectorizer = CountVectorizer(
                preprocessor=lambda x: ' '.join(self.text_preprocessing(x)),
                min_df=1
            )
            bow_matrix = self.vectorizer.fit_transform(documents)
        else:
            bow_matrix = self.vectorizer.transform(documents)
        
        return bow_matrix, self.vectorizer.get_feature_names_out()
    
    def tfidf_representation(self, documents):
        """
        Create TF-IDF representation
        """
        tfidf_vectorizer = TfidfVectorizer(
            preprocessor=lambda x: ' '.join(self.text_preprocessing(x)),
            min_df=1
        )
        tfidf_matrix = tfidf_vectorizer.fit_transform(documents)
        return tfidf_matrix, tfidf_vectorizer.get_feature_names_out()
    
    def analyze_text_statistics(self, text):
        """Analyze basic text statistics"""
        # Original text stats
        try:
            original_words = word_tokenize(text)
            sentence_count = len(sent_tokenize(text))
        except LookupError:
            original_words = text.split()
            sentence_count = len([s for s in text.split('.') if s.strip()])
        
        original_word_count = len(original_words)
        original_char_count = len(text)
        
        # Preprocessed text stats
        preprocessed_tokens = self.text_preprocessing(text)
        preprocessed_word_count = len(preprocessed_tokens)
        unique_words = len(set(preprocessed_tokens))
        
        return {
            'original_word_count': original_word_count,
            'original_char_count': original_char_count,
            'sentence_count': sentence_count,
            'preprocessed_word_count': preprocessed_word_count,
            'unique_words': unique_words,
            'tokens': preprocessed_tokens
        }

def compute_naive_bayes_probabilities(df, target_col, feature_cols, selected_values):
    """
    Compute Naive Bayes probabilities manually: P(class) × Π P(feature=value | class)
    """
    results = {
        "priors": {},
        "likelihoods": {},
        "joint": {},
        "posteriors": {}
    }
    
    # Calculate priors P(class)
    class_counts = df[target_col].value_counts()
    total_instances = len(df)
    
    for cls in class_counts.index:
        results["priors"][str(cls)] = class_counts[cls] / total_instances
    
    # Calculate likelihoods P(feature=value | class)
    for cls in class_counts.index:
        class_data = df[df[target_col] == cls]
        results["likelihoods"][str(cls)] = {}
        
        for feat in feature_cols:
            selected_val = selected_values[feat]
            # Count how many times this feature has the selected value in this class
            feat_count = len(class_data[class_data[feat].astype(str) == selected_val])
            # Use Laplace smoothing to avoid zero probabilities
            likelihood = (feat_count + 1) / (len(class_data) + len(df[feat].unique()))
            results["likelihoods"][str(cls)][feat] = likelihood
    
    # Calculate joint probabilities: P(class) × Π P(feature=value | class)
    for cls in class_counts.index:
        cls_str = str(cls)
        joint_prob = results["priors"][cls_str]
        for feat in feature_cols:
            joint_prob *= results["likelihoods"][cls_str][feat]
        results["joint"][cls_str] = joint_prob
    
    # Calculate posterior probabilities (normalized)
    total_joint = sum(results["joint"].values())
    for cls in class_counts.index:
        cls_str = str(cls)
        if total_joint > 0:
            results["posteriors"][cls_str] = results["joint"][cls_str] / total_joint
        else:
            results["posteriors"][cls_str] = 0.0
    
    return results

def main():
    st.set_page_config(
        page_title="NLP Toolkit with Naive Bayes",
        page_icon="📚",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Initialize session state
    if 'toolkit' not in st.session_state:
        st.session_state.toolkit = NLPToolkit()
    if 'user_text' not in st.session_state:
        st.session_state.user_text = ""
    
    # Sidebar
    st.sidebar.title("📚 Advanced NLP Toolkit")
    st.sidebar.markdown("---") 
        
    # Text input section
    st.header("📝 Input Text for NLP Analysis")
    user_text = st.text_area(
        "Enter your text here:",
        value=st.session_state.user_text,
        height=150,
        placeholder="Type or paste your text here for NLP analysis...",
        key="text_input"
    )
        
    if user_text:
        st.session_state.user_text = user_text
        
    st.markdown("---")
        
    # NLP Operations
    st.header("🔧 NLP Operations")
    
    # Create tabs for different operations (removed Classification tab)
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Text Analysis", 
        "🔤 Text Preprocessing", 
        "📖 Vocabulary", 
        "🎯 Text Representations", 
        "🤖 Naive Bayes"
    ])
        
    with tab1:
        st.subheader("Text Statistics Analysis")
        if st.session_state.user_text:
            stats = st.session_state.toolkit.analyze_text_statistics(st.session_state.user_text)
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Original Words", stats['original_word_count'])
            with col2:
                st.metric("Characters", stats['original_char_count'])
            with col3:
                st.metric("Sentences", stats['sentence_count'])
            with col4:
                st.metric("Unique Words", stats['unique_words'])
            
            st.subheader("Preprocessed Tokens")
            st.write(stats['tokens'])
        else:
            st.warning("Please enter some text to analyze.")
    
    with tab2:
        st.subheader("Text Preprocessing")
        if st.session_state.user_text:
            col1, col2 = st.columns(2)
            with col1:
                method = st.radio(
                    "Preprocessing Method:",
                    ["lemmatization", "stemming"],
                    horizontal=True
                )
            with col2:
                remove_stopwords = st.checkbox("Remove Stopwords", value=True)
            
            if st.button("Preprocess Text"):
                tokens = st.session_state.toolkit.text_preprocessing(
                    st.session_state.user_text, 
                    method=method, 
                    remove_stopwords=remove_stopwords
                )
                
                st.subheader("Preprocessed Result")
                st.write("**Tokens:**", tokens)
                
                # Show before and after
                col1, col2 = st.columns(2)
                with col1:
                    st.text_area("Original Text", st.session_state.user_text, height=150, disabled=True)
                with col2:
                    st.text_area("Processed Text", " ".join(tokens), height=150, disabled=True)
        else:
            st.warning("Please enter some text to preprocess.")
    
    with tab3:
        st.subheader("Vocabulary Creation")
        if st.session_state.user_text:
            min_freq = st.slider("Minimum Frequency", min_value=1, max_value=5, value=1)
            
            if st.button("Create Vocabulary"):
                vocab = st.session_state.toolkit.create_vocabulary([st.session_state.user_text], min_freq=min_freq)
                
                st.subheader("Vocabulary")
                st.write(f"**Total Words:** {len(vocab)}")
                
                # Display vocabulary in a nice format
                words_per_row = 5
                vocab_words = list(vocab.keys())
                
                for i in range(0, len(vocab_words), words_per_row):
                    cols = st.columns(words_per_row)
                    for j, word in enumerate(vocab_words[i:i+words_per_row]):
                        with cols[j]:
                            st.info(word)
        else:
            st.warning("Please enter some text to create vocabulary.")
    
    with tab4:
        st.subheader("Text Representations")
        if st.session_state.user_text:
            representation_type = st.radio(
                "Select Representation:",
                ["Bag-of-Words (BOW)", "TF-IDF"],
                horizontal=True
            )
            
            if st.button("Generate Representation"):
                if representation_type == "Bag-of-Words (BOW)":
                    bow_matrix, feature_names = st.session_state.toolkit.bow_representation([st.session_state.user_text])
                    st.subheader("Bag-of-Words Representation")
                else:
                    bow_matrix, feature_names = st.session_state.toolkit.tfidf_representation([st.session_state.user_text])
                    st.subheader("TF-IDF Representation")
                
                # Create a DataFrame for better visualization
                df = pd.DataFrame(
                    bow_matrix.toarray(),
                    columns=feature_names,
                    index=['Your Text']
                )
                
                st.write("**Feature Matrix:**")
                st.dataframe(df)
                
                st.write("**Shape:**", bow_matrix.shape)
                st.write("**Features:**", list(feature_names))
        else:
            st.warning("Please enter some text to generate representations.")
        
    with tab5:
            st.markdown('<h2 class="card-title">Naive Bayes — Probability Calculator</h2><p class="card-description">Compute P(class) × Π P(feature=value | class) from your dataset.</p>', unsafe_allow_html=True)
            
            uploaded = st.file_uploader("Upload CSV / Excel", type=["csv", "xls", "xlsx"], key="nb_prob_upload")
            df_nb = None
            file_bytes = None
            filename = None
            
            if uploaded:
                try:
                    file_bytes = uploaded.read()
                    uploaded.seek(0)
                    filename = uploaded.name
                    if filename.lower().endswith(".csv"):
                        df_nb = pd.read_csv(io.BytesIO(file_bytes))
                    else:
                        df_nb = pd.read_excel(io.BytesIO(file_bytes))
                    st.success("File loaded.")
                    st.dataframe(df_nb.head(), use_container_width=True)
                except Exception as e:
                    st.error(f"Failed to read file: {e}")

            if df_nb is None:
                st.info("Upload a file to continue.")
            else:
                cols = list(df_nb.columns)
                target_col = st.selectbox("Target column (label)", cols)
                feature_cols = st.multiselect("Feature columns", [c for c in cols if c != target_col])
                
                if not feature_cols:
                    st.warning("Please select at least one feature column.")
                else:
                    st.markdown("### Choose the feature values to evaluate:")
                    selected_values = {}
                    for feat in feature_cols:
                        uniques = df_nb[feat].dropna().unique().tolist()
                        uniques_str = [str(x).strip() for x in uniques]
                        chosen = st.selectbox(f"{feat}", uniques_str, key=f"nb_val_{feat}")
                        selected_values[feat] = chosen

                    if st.button("⚖️ Compute probabilities", key="nb_prob_compute"):
                        with st.spinner("Computing Naive Bayes probabilities..."):
                            try:
                                res = compute_naive_bayes_probabilities(df_nb, target_col, feature_cols, selected_values)
                                
                                # Display unnormalized joint probabilities
                                joint = res["joint"]
                                st.markdown("### Unnormalized Joint Probabilities (Main Output)")
                                joint_df = pd.DataFrame(list(joint.items()), columns=["Class", "Joint (unnormalized)"])
                                st.dataframe(joint_df.sort_values("Joint (unnormalized)", ascending=False).reset_index(drop=True), use_container_width=True)
                                
                                best = joint_df.sort_values("Joint (unnormalized)", ascending=False).iloc[0]
                                st.success(f"Most likely class: **{best['Class']}** (joint = {best['Joint (unnormalized)']:.8f})")
                                
                                # Display posterior probabilities
                                post_df = pd.DataFrame(list(res["posteriors"].items()), columns=["Class", "Posterior (normalized)"])
                                st.markdown("### Posterior Probabilities (Normalized)")
                                st.dataframe(post_df.sort_values("Posterior (normalized)", ascending=False).reset_index(drop=True), use_container_width=True)
                                
                                # Display priors
                                st.markdown("### Priors P(Class)")
                                st.dataframe(pd.DataFrame(list(res["priors"].items()), columns=["Class", "P(Class)"]))
                                
                                # Display likelihoods
                                st.markdown("### Likelihoods P(feature=value | class)")
                                for cls, feats in res["likelihoods"].items():
                                    st.markdown(f"**Class: {cls}**")
                                    rows = [{"Feature": f, "Value": selected_values[f], "Likelihood": p} for f, p in feats.items()]
                                    st.dataframe(pd.DataFrame(rows), use_container_width=True)
                                
                                # Show the actual calculation
                                st.markdown("### 🔍 Calculation Breakdown")
                                for cls in res["priors"].keys():
                                    prior = res["priors"][cls]
                                    likelihoods = [res["likelihoods"][cls][feat] for feat in feature_cols]
                                    joint_prob = res["joint"][cls]
                                    
                                    st.write(f"**For class '{cls}':**")
                                    st.write(f"P({cls}) = {prior:.6f}")
                                    for i, feat in enumerate(feature_cols):
                                        st.write(f"P({feat}={selected_values[feat]} | {cls}) = {likelihoods[i]:.6f}")
                                    
                                    # Show the multiplication
                                    calculation = " × ".join([f"{prob:.6f}" for prob in [prior] + likelihoods])
                                    st.write(f"Joint = {calculation} = {joint_prob:.8f}")
                                    st.markdown("---")
                                    
                            except Exception as e:
                                st.error(f"Error: {e}")

    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center'>
            <p>Built with ❤️ using Streamlit and NLTK | Naive Bayes Probability Calculator</p>
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()