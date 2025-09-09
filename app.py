import streamlit as st
import pickle
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# --- Caching Functions (no changes needed here) ---
@st.cache_resource
def load_embeddings(path="embeddings.npy"):
    with open(path, 'rb') as f:
        return np.load(f)

@st.cache_resource
def load_chunks(path="chunks.pkl"):
    with open(path, 'rb') as f:
        return pickle.load(f)

@st.cache_resource
def load_model(model_name='all-MiniLM-L6-v2'):
    return SentenceTransformer(model_name)

# --- Load all necessary components ---
embeddings = load_embeddings()
chunks = load_chunks() # This now loads a list of dictionaries
model = load_model()

# --- Streamlit App Interface ---
st.set_page_config(layout="wide")
st.title("AI Policy Semantic Search Engine 📜")
st.write(
    "Ask a question about the EU AI Act, and this tool will find the most relevant clauses. "
    "Results include the source page number and are formatted as excerpts."
)

# User query input box
query = st.text_input("Enter your question or query:", "")

if query:
    with st.spinner("Finding relevant clauses..."):
        query_embedding = model.encode([query])
        similarities = cosine_similarity(query_embedding, embeddings)[0]
        
        k = 5
        top_k_indices = np.argsort(similarities)[-k:][::-1]
        
        # --- NEW: Updated Display Logic ---
        st.subheader("Top Relevant Clauses:")
        for i in top_k_indices:
            # Each 'chunk' is now a dictionary
            chunk_data = chunks[i]
            st.markdown(f"---")
            # Display the source and score
            st.write(
                f"**Source:** {chunk_data['source']} | **Similarity Score:** {similarities[i]:.2f}"
            )
            # Display the text formatted as an excerpt with ellipses
            st.info(f"...{chunk_data['text']}...")