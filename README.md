# AI Policy Semantic Search Engine

**Streamlit Link: eu-ai-act-semantic-search.streamlit.app/**

This project is a semantic search engine for the EU's Artificial Intelligence Act. It allows users to ask questions in natural language and retrieve the most contextually relevant clauses from the legal document.

This project was built to demonstrate a core component of a **Retrieval-Augmented Generation (RAG)** system, showcasing the ability to process, embed, and search qualitative text data for efficient information retrieval.

## Architecture

The system works in two stages: an offline **Indexing Stage** and an online **Query Stage**. This separation ensures that the user-facing application is fast and responsive, as all the heavy data processing is done upfront.

### 1. Indexing Stage (`process_data.py`)
* **Data Ingestion:** The script first ingests the `Artificial Intelligence Act.pdf` using the `pypdf` library.
* **Text Chunking:** The extracted text is cleaned and split into smaller, semantically meaningful paragraphs (chunks). This is a critical step for ensuring the search results are specific and relevant.
* **Embedding Generation:** Each chunk is converted into a 384-dimensional vector embedding using the `all-MiniLM-L6-v2` model from the `sentence-transformers` library. This model is chosen for its excellent balance of speed and performance.
* **Embedding Storage:** The generated vector embeddings are saved directly to a file (`embeddings.npy`) using NumPy. For this prototype, this approach is more stable and simpler than using a specialized indexing library.

### 2. Query Stage (`app.py`)
* **User Interface:** A web interface built with `Streamlit` allows the user to enter a query.
* **Query Embedding:** The user's query is converted into a vector embedding using the same `all-MiniLM-L6-v2` model to ensure it's in the same vector space as the document chunks.
* **Similarity Search:** The application uses `scikit-learn`'s `cosine_similarity` function to efficiently compare the user's query vector against all the chunk vectors loaded from `embeddings.npy`. The similarities are calculated, and the results are sorted to retrieve the top 5 most relevant text chunks.
* **Display Results:** The most relevant clauses, along with their similarity scores, are displayed to the user.

## Technical Stack

* **Python 3.9+**
* **Streamlit:** For the web interface.
* **Sentence-Transformers:** For generating text embeddings.
* **Scikit-learn:** For performing the cosine similarity search.
* **PyPDF:** For PDF text extraction.
* **NumPy:** For numerical operations and data storage.

## How to Run This Project Locally

1.  Clone the repository and place the `Artificial Intelligence Act.pdf` in the project folder.
2.  Create and activate a virtual environment:
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```
3.  Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```
4.  Run the one-time data processing and embedding script:
    ```bash
    python process_data.py
    ```
5.  Run the Streamlit application:
    ```bash
    streamlit run app.py
    ```
