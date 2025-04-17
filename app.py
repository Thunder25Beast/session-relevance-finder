import streamlit as st
import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity

# -------------------------------
# Helper Functions and Setup
# -------------------------------

# Ensure the stopwords resource is available
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

def clean_text_simple(text):
    """Basic text cleaning: lowercasing, removing punctuation/numbers, tokenizing, and removing stopwords."""
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    words = text.split()
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]
    return ' '.join(words)

# -------------------------------
# Data Loading and Preprocessing
# -------------------------------

# Load the data from your Excel file. Make sure the file is in the same folder as app.py,
# or provide the correct path.
df = pd.read_excel('Session-Summary-for-E6-project.xlsx')

# Clean the session summaries
df['Cleaned_Summary'] = df['Session_Summary'].apply(clean_text_simple)

# -------------------------------
# Text Featurization and Clustering
# -------------------------------

# Create TF-IDF features for the cleaned summaries
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(df['Cleaned_Summary'])

# Choose the number of clusters. In your case, you've chosen 12.
optimal_k = 13
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
df['Cluster'] = kmeans.fit_predict(tfidf_matrix)

# For ranking summaries within a cluster, compute cosine similarity between each summary
# and its cluster's centroid.
centroids = kmeans.cluster_centers_
similarity_scores = []
for idx in range(df.shape[0]):
    cluster_label = df.loc[idx, 'Cluster']
    summary_vec = tfidf_matrix[idx].toarray()  # convert to dense
    centroid_vec = centroids[cluster_label].reshape(1, -1)
    score = cosine_similarity(summary_vec, centroid_vec)[0][0]
    similarity_scores.append(score)
df['Similarity_Score'] = similarity_scores

# Pre-compute each cluster's centroid in TF-IDF space (mean vector)
cluster_centroids = {}
for cluster in df['Cluster'].unique():
    indices = df[df['Cluster'] == cluster].index
    # Note: tfidf_matrix[indices] is a sparse matrix, so mean is computed over the dense array.
    cluster_centroids[cluster] = tfidf_matrix[indices].mean(axis=0)

def find_relevant_session(query):
    """
    Given a query string, find the session (cluster) that is most relevant.
    Returns the best cluster number.
    """
    query_clean = clean_text_simple(query)
    query_vec = tfidf_vectorizer.transform([query_clean])
    scores = {}
    for cluster, centroid in cluster_centroids.items():
        # Convert centroid to a dense numpy array if needed.
        centroid_dense = np.array(centroid)
        score = cosine_similarity(query_vec, centroid_dense.reshape(1, -1))[0][0]
        scores[cluster] = score
    best_cluster = max(scores, key=scores.get)
    return best_cluster, scores[best_cluster]

# -------------------------------
# Streamlit App Interface
# -------------------------------

st.title("Session Relevance Finder")
st.write("Enter a set of keywords or a brief topic description to identify the most relevant session and view its top summaries.")

# User input for query
user_query = st.text_input("Enter keywords or description:")

if user_query:
    best_cluster, sim_score = find_relevant_session(user_query)
    st.write(f"**Most Relevant Session (Cluster):** {best_cluster} (Cosine Similarity: {sim_score:.3f})")
    
    # Filter summaries from the best cluster and rank by similarity score (descending)
    session_summaries = df[df['Cluster'] == best_cluster].sort_values(by='Similarity_Score', ascending=False)
    
    st.write("### Top 3 Summaries for the Selected Session:")
    for idx, row in session_summaries.head(3).iterrows():
        st.write(f"#### Summary {idx + 1}:")
        st.write(f"- {row['Session_Summary']}")
        st.write("---")  # Add a horizontal line for separation

    # Alternatively, if you want to display the top summaries in an expandable 'text window':
    with st.expander("View Detailed Top Summaries in a draggable text window"):
        for idx, row in session_summaries.head(3).iterrows():
            st.text(f"Summary {idx + 1} (Serial No. {row['SerialNo']}):")
            st.text(row['Session_Summary'])
            st.text("-" * 50)  # Add a dashed line for separation

