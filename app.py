import streamlit as st
from PyPDF2 import PdfReader
import pdfplumber
from langchain.text_splitter import CharacterTextSplitter
import openai
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


def extract_text_pypdf2(pdf_file):
    """
    Extract text from a PDF file using PyPDF2.
    """
    reader = PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text

def extract_text_pdfplumber(pdf_file):
    """
    Extract text from a PDF file using pdfplumber.
    """
    text = ""
    with pdfplumber.open(pdf_file) as pdf:
        for page in pdf.pages:
            text += page.extract_text() or ""
    return text

def chunk_text(text, chunk_size=1000, chunk_overlap=200):
    """
    Chunk the given text into manageable pieces using LangChain's CharacterTextSplitter.

    Args:
        text (str): The complete text to be chunked.
        chunk_size (int): Maximum size of each chunk.
        chunk_overlap (int): Overlapping size between consecutive chunks.

    Returns:
        list: A list of text chunks.
    """
    text_splitter = CharacterTextSplitter(
        separator="\n",  # Split by newlines or other separators
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )
    chunks = text_splitter.split_text(text)
    return chunks

def initialize_models():
    # OpenAI API Key
    openai.api_key = st.secrets["openai_api_key"]  # Store the key in Streamlit secrets
    # Sentence-Transformers model
    sentence_transformer_model = SentenceTransformer("all-MiniLM-L6-v2")
    return sentence_transformer_model

# Generate embeddings using OpenAI
def generate_openai_embeddings(chunks):
    """
    Generate embeddings for text chunks using OpenAI's text-embedding-ada-002 model.

    Args:
        chunks (list): List of text chunks.

    Returns:
        list: List of embeddings.
    """
    embeddings = []
    for chunk in chunks:
        response = openai.Embedding.create(input=chunk, model="text-embedding-ada-002")
        embeddings.append(response["data"][0]["embedding"])
    return embeddings

# Generate embeddings using Sentence-Transformers
def generate_sentence_transformer_embeddings(chunks, model):
    """
    Generate embeddings for text chunks using Sentence-Transformers.

    Args:
        chunks (list): List of text chunks.
        model: Preloaded Sentence-Transformers model.

    Returns:
        list: List of embeddings.
    """
    embeddings = model.encode(chunks, convert_to_tensor=True)
    return embeddings

def initialize_models():
    # OpenAI API Key
    openai.api_key = st.secrets["openai_api_key"]  # Store the key in Streamlit secrets
    # Sentence-Transformers model
    sentence_transformer_model = SentenceTransformer("all-MiniLM-L6-v2")
    return sentence_transformer_model

# Generate embeddings for a query
def generate_query_embedding(query, model, method="openai"):
    if method == "openai":
        response = openai.Embedding.create(input=query, model="text-embedding-ada-002")
        return response["data"][0]["embedding"]
    elif method == "sentence-transformer":
        return model.encode(query, convert_to_tensor=True)

# Perform similarity search
def find_relevant_chunks(query_embedding, chunk_embeddings, chunks, top_k=3):
    """
    Find the top_k most similar text chunks to the query based on cosine similarity.
    """
    similarities = cosine_similarity([query_embedding], chunk_embeddings).flatten()
    top_indices = np.argsort(similarities)[::-1][:top_k]
    relevant_chunks = [chunks[i] for i in top_indices]
    return relevant_chunks

# Query LLM with context
def query_llm(question, context, model="openai"):
    """
    Query an LLM with the question and context.
    """
    if model == "openai":
        prompt = (
            f"Context: {context}\n\n"
            f"Question: {question}\n"
            f"Answer:"
        )
        response = openai.Completion.create(
            model="text-davinci-003",
            prompt=prompt,
            max_tokens=200,
            temperature=0
        )
        return response["choices"][0]["text"].strip()

def initialize_openai(api_key):
    openai.api_key = api_key

def initialize_sentence_transformer():
    return SentenceTransformer("all-MiniLM-L6-v2")

# Extract text from PDF
def extract_text_from_pdf(uploaded_file):
    pdf_reader = PdfReader(uploaded_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

# Generate embeddings
def generate_embeddings(texts, model, method="sentence-transformer"):
    if method == "openai":
        embeddings = [openai.Embedding.create(input=chunk, model="text-embedding-ada-002")["data"][0]["embedding"] for chunk in texts]
    else:
        embeddings = model.encode(texts, convert_to_tensor=False)
    return embeddings

# Perform similarity search
def find_relevant_chunks(query_embedding, chunk_embeddings, chunks, top_k=3):
    similarities = cosine_similarity([query_embedding], chunk_embeddings).flatten()
    top_indices = np.argsort(similarities)[::-1][:top_k]
    return [chunks[i] for i in top_indices]

# Query LLM
def query_llm(question, context):
    prompt = f"Context: {context}\n\nQuestion: {question}\nAnswer:"
    response = openai.Completion.create(
        model="text-davinci-003",
        prompt=prompt,
        max_tokens=200,
        temperature=0
    )
    return response["choices"][0]["text"].strip()

# Streamlit App
def main():
    st.title("Chat with PDF")
    st.write("Upload a PDF document to extract text and interact with it.")

    # PDF Upload
    uploaded_file = st.file_uploader("Upload PDF", type="pdf")

    if uploaded_file is not None:
        st.write("### Extracted Text")
        
        # Text extraction using PyPDF2
        try:
            st.write("Using PyPDF2:")
            pypdf2_text = extract_text_pypdf2(uploaded_file)
            if pypdf2_text.strip():
                st.text_area("Extracted Text (PyPDF2)", pypdf2_text, height=300)
            else:
                st.warning("PyPDF2 could not extract text. Trying pdfplumber.")
        except Exception as e:
            st.error(f"Error with PyPDF2: {e}")

        # Text extraction using pdfplumber
        try:
            st.write("Using pdfplumber:")
            pdfplumber_text = extract_text_pdfplumber(uploaded_file)
            if pdfplumber_text.strip():
                st.text_area("Extracted Text (pdfplumber)", pdfplumber_text, height=300)
            else:
                st.warning("pdfplumber could not extract text.")
        except Exception as e:
            st.error(f"Error with pdfplumber: {e}")

    st.title("Chat with PDF: Text Chunking")
    st.write("Upload a PDF and extract its text into manageable chunks.")

    # Simulated Text Input (Replace with actual PDF text extraction in practice)
    example_text = (
        "This is a sample text to demonstrate text chunking. "
        "A large PDF would contain multiple pages of content. "
        "By splitting this content into chunks, we make it easier to process. "
        * 10  # Repeating for demonstration
    )
    st.write("### Full Extracted Text")
    st.text_area("Extracted Text", example_text, height=300)

    # Chunk the text
    st.write("### Text Chunking")
    chunk_size = st.slider("Chunk Size (characters)", min_value=200, max_value=2000, value=1000, step=100)
    chunk_overlap = st.slider("Chunk Overlap (characters)", min_value=0, max_value=500, value=200, step=50)

    chunks = chunk_text(example_text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    st.write(f"Number of Chunks: {len(chunks)}")
    for i, chunk in enumerate(chunks):
        st.write(f"**Chunk {i + 1}:**")
        st.text(chunk)
    st.title("Chat with PDF: Embedding Generation")
    st.write("Upload a PDF, extract text chunks, and generate embeddings.")

    # Simulated chunks (replace with actual chunks in practice)
    chunks = [
        "This is the first chunk of text from the PDF.",
        "This is the second chunk of text from the PDF.",
        "This is the third chunk of text from the PDF."
    ]
    st.write("### Text Chunks")
    for i, chunk in enumerate(chunks):
        st.write(f"**Chunk {i + 1}:** {chunk}")

    # Initialize models
    sentence_transformer_model = initialize_models()

    # Embedding options
    embedding_method = st.selectbox("Select Embedding Method", ["OpenAI", "Sentence-Transformers"])

    if st.button("Generate Embeddings"):
        if embedding_method == "OpenAI":
            st.write("Generating embeddings using OpenAI's text-embedding-ada-002...")
            try:
                openai_embeddings = generate_openai_embeddings(chunks)
                st.write("Embeddings generated successfully!")
                st.write(openai_embeddings)
            except Exception as e:
                st.error(f"Error generating OpenAI embeddings: {e}")
        elif embedding_method == "Sentence-Transformers":
            st.write("Generating embeddings using Sentence-Transformers...")
            try:
                st_embeddings = generate_sentence_transformer_embeddings(chunks, sentence_transformer_model)
                st.write("Embeddings generated successfully!")
                st.write(st_embeddings)
            except Exception as e:
                st.error(f"Error generating Sentence-Transformers embeddings: {e}")

    st.title("Chat with PDF: Question Answering")
    st.write("Upload a PDF, extract text chunks, generate embeddings, and ask questions.")

    # Simulated chunks and embeddings (replace with actual processed data)
    chunks = [
        "This is the first chunk of text from the PDF.",
        "This is the second chunk of text from the PDF.",
        "This is the third chunk of text from the PDF."
    ]
    embeddings_method = st.selectbox("Select Embedding Method", ["OpenAI", "Sentence-Transformers"])
    sentence_transformer_model = initialize_models()
    
    # Generate chunk embeddings
    if embeddings_method == "OpenAI":
        chunk_embeddings = [generate_query_embedding(chunk, None, "openai") for chunk in chunks]
    elif embeddings_method == "Sentence-Transformers":
        chunk_embeddings = sentence_transformer_model.encode(chunks, convert_to_tensor=True)

    # User Question
    question = st.text_input("Ask a question about the PDF content:")
    if st.button("Get Answer"):
        if question:
            # Generate query embedding
            query_embedding = generate_query_embedding(question, sentence_transformer_model, embeddings_method.lower())
            
            # Find relevant chunks
            relevant_chunks = find_relevant_chunks(query_embedding, chunk_embeddings, chunks)
            st.write("### Relevant Chunks")
            for chunk in relevant_chunks:
                st.write(f"- {chunk}")

            # Query LLM
            context = "\n".join(relevant_chunks)
            answer = query_llm(question, context)
            st.write("### Answer")
            st.write(answer)
        else:
            st.error("Please enter a question.")

     st.title("Chat with PDF Application")
    st.sidebar.header("Settings")
    
    # Set up OpenAI API Key
    openai_api_key = st.sidebar.text_input("OpenAI API Key", type="password")
    if openai_api_key:
        initialize_openai(openai_api_key)
    else:
        st.warning("Please enter your OpenAI API Key to use the app.")
        return

    # Upload PDF
    uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])
    if uploaded_file is not None:
        with st.spinner("Extracting text from PDF..."):
            text = extract_text_from_pdf(uploaded_file)
            st.success("Text extracted successfully!")
            st.text_area("Extracted Text", text[:1000], height=300)  # Show only the first 1000 characters

        # Chunking (for demonstration purposes, split into sentences or fixed size chunks)
        chunks = [text[i:i+500] for i in range(0, len(text), 500)]
        model = initialize_sentence_transformer()
        chunk_embeddings = generate_embeddings(chunks, model)

        # User Query Section
        st.subheader("Ask Questions About the PDF")
        question = st.text_input("Enter your question:")
        if st.button("Get Answer"):
            if question:
                query_embedding = generate_embeddings([question], model)[0]
                relevant_chunks = find_relevant_chunks(query_embedding, chunk_embeddings, chunks)
                context = "\n".join(relevant_chunks)
                with st.spinner("Generating response..."):
                    answer = query_llm(question, context)
                st.markdown("### Relevant Chunks")
                for chunk in relevant_chunks:
                    st.write(f"- {chunk}")
                st.markdown("### Answer")
                st.success(answer)
            else:
                st.error("Please enter a question.")


# Run the app
if __name__ == "__main__":
    main()
