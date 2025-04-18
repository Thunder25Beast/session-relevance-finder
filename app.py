# Import necessary libraries
import streamlit as st
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer # Although not used in the final model, keep imports consistent
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import os # To check for file existence

# --- Configuration ---
DATA_FILE = 'Session-Summary-for-E6-project.xlsx'
BERT_MODEL_NAME = 'all-MiniLM-L6-v2'
N_CLUSTERS = 13 # Chosen based on evaluation
N_TOP_SUMMARIES = 3 # Number of summaries to show in search results

# --- NLTK Downloads (Cached) ---
@st.cache_resource
def download_nltk_resources():
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords')
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')
    return set(stopwords.words('english'))

stop_words = download_nltk_resources()

# --- Data Loading and Preprocessing (Cached) ---
@st.cache_data
def load_data(file_path):
    if not os.path.exists(file_path):
        st.error(f"Data file not found: {file_path}")
        st.stop() # Stop the app if data file is missing
    df = pd.read_excel(file_path)
    return df

@st.cache_data
def clean_text(df, text_col='Session_Summary'):
    # Reuse the cleaning function developed earlier
    def clean_text_simple(text):
        if pd.isna(text): # Handle potential NaN values
            return ''
        text = str(text).lower()
        text = re.sub(r'[^a-z\s]', '', text)
        words = text.split()
        # Filter stopwords and short words using the cached stop_words set
        words = [word for word in words if word not in stop_words and len(word) > 1]
        return ' '.join(words)

    df['Cleaned_Summary'] = df[text_col].apply(clean_text_simple)
    return df

# --- Model Loading and Embedding Generation (Cached) ---
@st.cache_resource # Use cache_resource for the model itself
def load_bert_model(model_name):
    try:
        model = SentenceTransformer(model_name)
        return model
    except Exception as e:
        st.error(f"Error loading BERT model: {e}")
        st.stop()

# Corrected function signature: add leading underscore to 'model'
@st.cache_data
def generate_embeddings(df, _model, text_col='Cleaned_Summary'):
    # Generate embeddings for the cleaned summaries
    try:
        # Use the _model parameter here
        embeddings = _model.encode(df[text_col].tolist(), show_progress_bar=False) # show_progress_bar=False in Streamlit
        return np.array(embeddings)
    except Exception as e:
        st.error(f"Error generating embeddings: {e}")
        st.stop()

# --- Clustering (Cached) ---
@st.cache_data
def perform_clustering(embeddings, n_clusters):
    try:
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10) # Add n_init for KMeans
        labels = kmeans.fit_predict(embeddings)
        return labels
    except Exception as e:
        st.error(f"Error during clustering: {e}")
        st.stop()

# --- Search Function ---
def find_top_n_summaries_bert(query_keywords, embeddings, df, model, n=N_TOP_SUMMARIES):
    # Convert query keywords to a vector using the BERT model
    try:
        query_embedding = model.encode([query_keywords])[0]
    except Exception as e:
        st.error(f"Error encoding query with BERT model: {e}")
        return []

    # Calculate cosine similarity between query embedding and all summary embeddings
    cosine_similarities = cosine_similarity([query_embedding], embeddings)[0]

    # Get the indices of the top N summaries
    top_n_indices = np.argsort(cosine_similarities)[::-1][:n]

    # Return the top N summaries and their info
    top_summaries_info = []
    for i in top_n_indices:
        summary = df.loc[i, 'Session_Summary']  # Get original summary
        serial_no = df.loc[i, 'SerialNo']  # Get the serial number from the file
        cosine_similarity_score = cosine_similarities[i]  # Explicitly name it as cosine similarity
        cluster_label = df.loc[i, 'Cluster_Label']
        top_summaries_info.append({
            'serial_no': serial_no,
            'summary': summary,
            'cosine_similarity': cosine_similarity_score,
            'cluster': cluster_label
        })

    return top_summaries_info

# --- Streamlit App Layout ---
st.markdown(
    """
    <style>
    body {
        background-color: #0d1b2a;
        color: #e0e1dd;
    }
    .stButton>button {
        background-color: #1b263b;
        color: #e0e1dd;
        border: 1px solid #415a77;
        border-radius: 5px;
    }
    .stTextInput>div>div>input {
        background-color: #1b263b;
        color: #e0e1dd;
        border: 1px solid #415a77;
        border-radius: 5px;
    }
    .stExpander {
        background-color: #1b263b;
        color: #e0e1dd;
        border: 1px solid #415a77;
        border-radius: 5px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("Session Summary DS203")

st.markdown("""
Welcome! Use the search box below to find relevant session summaries based on keywords.
""")
# --- Load Data and Perform Initial Processing ---
# These functions are cached, so they only run the first time or when inputs change
df = load_data(DATA_FILE)
df = clean_text(df)
bert_model = load_bert_model(BERT_MODEL_NAME)
# Pass bert_model with underscore to the cached generate_embeddings function
bert_embeddings = generate_embeddings(df, bert_model)
cluster_labels = perform_clustering(bert_embeddings, N_CLUSTERS)

# Add cluster labels to the DataFrame for easy access in search results
# This column name should match what's used in find_top_n_summaries_bert
df['Cluster_Label'] = cluster_labels


# --- Search Interface ---
st.header("Search Summaries")
query = st.text_input("Enter keywords here:", "")

if st.button("Search"):
    if query:
        st.subheader(f"Top {N_TOP_SUMMARIES} Results for '{query}'")
        # Pass the original bert_model object to the search function (which is not cached)
        top_summaries = find_top_n_summaries_bert(query, bert_embeddings, df, bert_model, N_TOP_SUMMARIES)

        if top_summaries:
            # Display results using expanders (simulating openable windows)
            for i, summary_info in enumerate(top_summaries):
                # Use markdown for title to include serial number, cosine similarity, and cluster
                expander_title = f"**Result {i+1}:** Serial No: {summary_info['serial_no']}, Cosine Similarity: {summary_info['cosine_similarity']:.4f}, Cluster: {summary_info['cluster']}"
                with st.expander(expander_title):
                    st.write(summary_info['summary'])  # Show the full summary
        else:
            st.info("No summaries found matching your query.")
    else:
        st.warning("Please enter some keywords to search.")


# --- Optional: Display Cluster Info (for exploration) ---
# You could add a section here to show info about clusters, e.g., top keywords per cluster
# Requires implementing the get_top_frequent_terms_per_cluster function here
# st.sidebar.header("Cluster Information")
# if st.sidebar.checkbox("Show Cluster Details"):
#     st.sidebar.info("Details about each cluster can be displayed here.")


# --- Styling Customization (Hint) ---
# st.sidebar.header("App Styling")
