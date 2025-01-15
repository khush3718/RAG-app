import streamlit as st
import os
import re
from io import BytesIO
import tempfile
# from database import get_db_connection
from sentence_transformers import SentenceTransformer
import PyPDF2
import docx
 

import psycopg2
import os
from dotenv import load_dotenv
 
# Load environment variables
load_dotenv()
 
# PostgreSQL connection parameters
DB_HOST = os.environ['DB_HOST']
DB_PORT = os.environ['DB_PORT']
DB_NAME = os.environ['DB_NAME']
DB_USER = os.environ['DB_USER']
DB_PASSWORD = os.environ['DB_PASSWORD']
 
def get_db_connection():
   
    return psycopg2.connect(
        host=DB_HOST,
        port=DB_PORT,
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD
    ) 
# Initialize SentenceTransformer model
model = SentenceTransformer('all-mpnet-base-v2')
 
# Define functions to process uploaded files
def read_pdf_file(file):
    text = ""
    pdf_reader = PyPDF2.PdfReader(file)
    for page in pdf_reader.pages:
        text += page.extract_text() + "\n"
    return text
 
def extract_questions_from_text(text):
    # Extract questions based on typical question formats

    question_pattern = re.compile(r'^(?:Who|What|When|Where|Why|How|Is|Are|Do|Does|Did|Can|Could|Would|Should|Will|Has|Have|Had)\b.*\?$')
    return [line.strip() for line in text.split('\n') if question_pattern.match(line.strip())]
    # return [line.strip() for line in text.split('\n') if line.strip().endswith('?')]
 
def process_uploaded_file(uploaded_file):
    file_name, file_extension = os.path.splitext(uploaded_file.name)
    if file_extension.lower() == ".pdf":
        return read_pdf_file(uploaded_file)
    elif file_extension.lower() == ".docx":
        doc = docx.Document(uploaded_file)
        return "\n".join([paragraph.text for paragraph in doc.paragraphs])
    else:
        st.error("Unsupported file format. Please upload PDF or DOCX files only.")
        return None
 
def add_to_database(text, file_name):
    conn = get_db_connection()
    if conn:
        cursor = conn.cursor()
        chunks = split_text(text)  # Split text into manageable chunks
        embeddings = [model.encode(chunk).tolist() for chunk in chunks]
       
        # Check if the file has already been indexed
        cursor.execute(
            """
            SELECT COUNT(*) FROM document_embeddings
            WHERE source_file = %s;
            """,
            (file_name,)
        )
        count = cursor.fetchone()[0]
       
        if count > 0:
            st.warning(f"{file_name} has already been indexed. Skipping file.")
            cursor.close()
            conn.close()
            return
    
        # Insert new records
        for i, chunk in enumerate(chunks):
            cursor.execute(
                """
                INSERT INTO document_embeddings (document_text, embedding, source_file, chunk_number)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (document_text, chunk_number, source_file) DO NOTHING;
                """,
                (chunk, embeddings[i], file_name, i)
            )
        conn.commit()
        cursor.close()
        conn.close()
        st.success(f"{file_name} indexed successfully!")
    else:
        st.error("Database connection failed.")
 
# Split text into chunks for embedding
def split_text(text, chunk_size=500):
    sentences = text.replace('\n', ' ').split('. ')
    chunks = []
    current_chunk = []
    current_size = 0
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        if not sentence.endswith('.'):
            sentence += '.'
        sentence_size = len(sentence)
        if current_size + sentence_size > chunk_size and current_chunk:
            chunks.append(' '.join(current_chunk))
            current_chunk = [sentence]
            current_size = sentence_size
        else:
            current_chunk.append(sentence)
            current_size += sentence_size
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    return chunks
 
# Perform semantic search
def semantic_search(query, n_results=3):
    conn = get_db_connection()
    if conn:
        cursor = conn.cursor()
        query_embedding = model.encode(query).tolist()
        cursor.execute(
            """
            SELECT document_text, source_file, chunk_number,
                   (embedding <=> %s::vector(768)) AS similarity
            FROM document_embeddings
            ORDER BY similarity ASC
            LIMIT %s;
            """,
            (query_embedding, n_results)
        )
        results = cursor.fetchall()
        cursor.close()
        conn.close()
        return results
    else:
        st.error("Database connection failed.")
        return []
 
# Streamlit app layout
st.title("Document-Based Question Answering System")
st.sidebar.title("Upload document_embeddings for Indexing")
 
# Upload for indexing
uploaded_files = st.sidebar.file_uploader(
    "Upload PDF/DOCX files for indexing",
    type=["pdf", "docx"],
    accept_multiple_files=True
)
 
if uploaded_files:
    for uploaded_file in uploaded_files:
        file_text = process_uploaded_file(uploaded_file)
        if file_text:
            add_to_database(file_text, uploaded_file.name)
 
# Question Answering from Uploaded PDFs
st.header("Question Answering from Uploaded PDFs")
 
# File uploader for questions
question_file = st.file_uploader("Upload PDF containing questions", type=["pdf","docx"])

# Input field for manual question input
manual_question_input = st.text_area("Or manually enter questions", height=200)
 
if question_file:
    question_text = read_pdf_file(question_file)
    questions = extract_questions_from_text(question_text)
    if questions:
        st.subheader("Extracted Questions")
        for i, question in enumerate(questions, start=1):
            st.write(f"**Q{i}:** {question}")
       
        if st.button("Get Answers"):
            st.subheader("Answers")
            for i, question in enumerate(questions, start=1):
                results = semantic_search(question)
                if results:
                    st.markdown(f"**Q{i}: {question}**")
                    for result in results:
                        st.markdown(f"""
                        **Source:** {result[1]}, **Chunk {result[2]}**
                        > {result[0]}
                        """)
                else:
                    st.warning(f"No results found for Q{i}: {question}")
    else:
        st.warning("No questions detected in the uploaded file.")
 
# Handling manual questions
elif manual_question_input:
    questions = [line.strip() for line in manual_question_input.split('\n') if line.strip()]
    st.subheader("Manually Entered Questions")
    for i, question in enumerate(questions, start=1):
        st.write(f"**Q{i}:** {question}")
   
    if st.button("Get Answers"):
        st.subheader("Answers")
        for i, question in enumerate(questions, start=1):
            results = semantic_search(question)
            if results:
                st.markdown(f"**Q{i}: {question}**")
                for result in results:
                    st.markdown(f"""
                    **Source:** {result[1]}, **Chunk {result[2]}**
                    > {result[0]}
                    """)
            else:
                st.warning(f"No results found for Q{i}: {question}")