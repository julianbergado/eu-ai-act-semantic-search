import pypdf
import re
import pickle
from sentence_transformers import SentenceTransformer
import numpy as np

def recursive_chunk_text(text, chunk_size=750, chunk_overlap=100):
    # This recursive function remains the same as before
    if len(text) <= chunk_size:
        return [text]
    separators = ["\n\n", "\n", ". ", " ", ""]
    for separator in separators:
        splits = text.split(separator)
        if len(splits) > 1:
            chunks = []
            for split in splits:
                chunks.extend(recursive_chunk_text(split, chunk_size, chunk_overlap))
            final_chunks = []
            current_chunk = ""
            for chunk in chunks:
                if len(current_chunk) + len(chunk) + 1 < chunk_size:
                    current_chunk += chunk + " "
                else:
                    final_chunks.append(current_chunk.strip())
                    overlap_start = max(0, len(current_chunk) - chunk_overlap)
                    current_chunk = current_chunk[overlap_start:] + chunk + " "
            final_chunks.append(current_chunk.strip())
            return [c for c in final_chunks if c]
    return [text[:chunk_size]]

if __name__ == '__main__':
    pdf_path = 'Artificial Intelligence Act.pdf'
    print(f"Reading PDF from: {pdf_path}")
    pdf_reader = pypdf.PdfReader(pdf_path)
    
    # NEW: Process page by page and store chunks as dictionaries
    all_chunks_with_source = []
    for i, page in enumerate(pdf_reader.pages):
        page_number = i + 1
        page_text = page.extract_text()
        if page_text:
            cleaned_text = re.sub(r'\s+', ' ', page_text).strip()
            page_chunks = recursive_chunk_text(cleaned_text, chunk_size=750, chunk_overlap=100)
            
            for chunk_text in page_chunks:
                all_chunks_with_source.append({
                    'text': chunk_text,
                    'source': f"Page {page_number}"
                })

    # Save the structured chunks
    with open('chunks.pkl', 'wb') as f:
        pickle.dump(all_chunks_with_source, f)
    print(f"Created {len(all_chunks_with_source)} chunks with source info. Saved to chunks.pkl")

    # Extract just the text for embedding
    chunk_texts_only = [chunk['text'] for chunk in all_chunks_with_source]

    # Load the model
    print("Loading sentence transformer model...")
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # Generate and save embeddings
    print("Generating embeddings for the new chunks...")
    embeddings = model.encode(chunk_texts_only, show_progress_bar=True)
    with open('embeddings.npy', 'wb') as f:
        np.save(f, embeddings)
        
    print("New embeddings have been saved to embeddings.npy")
    print("\nPreprocessing complete!")