import os
from typing import List
import PyPDF2
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeVectorStore
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def convert_pdf_to_vector_db(file_paths: List[str], index_name: str):
    """
    Convert PDF files to vector database using Google Gemini embeddings.
    This is a basic implementation that you can extend with actual vector DB logic.
    """
    print(f"Converting PDFs to vector DB: {file_paths}")
    print(f"Index name: {index_name}")
    
    try:
        for file_path in file_paths:
            print(f"Processing file: {file_path}")
            # Read the PDF file
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                print(f"PDF file read: {file_path}")
                
                # Process all pages, not just the first one
                all_text = ""
                for page_num, page in enumerate(pdf_reader.pages):
                    page_text = page.extract_text()
                    all_text += f"\n--- Page {page_num + 1} ---\n{page_text}"
                
                print(f"Total text extracted from {len(pdf_reader.pages)} pages")
                
                # Split the text into chunks using langchain
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                text_chunks = text_splitter.split_text(all_text)
                print(f"Text split into {len(text_chunks)} chunks")
                # Generate vector embeddings using langchain gemini embeddings
                google_api_key = os.getenv("GOOGLE_API_KEY")
                if not google_api_key:
                    raise ValueError("GOOGLE_API_KEY not found in environment variables. Please create a .env file with your API key.")
                
                embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001",google_api_key=google_api_key)
                vector_embeddings = embeddings.embed_documents(text_chunks)
                print(f"Generated {len(vector_embeddings)} embeddings")
                
                # Store the vector embeddings in the vector database
                # Note: This requires proper Pinecone configuration with API key and environment
                #Todo: Store the vector embeddings in the vector database
                
        
        print(f"All files processed successfully")
        return {"status": "PDFs processed successfully", "files": file_paths, "index": index_name}
        
    except Exception as e:
        print(f"Error processing PDFs: {str(e)}")
        return {"status": "Error", "error": str(e), "files": file_paths, "index": index_name}
                
            
        
        
        
        
        
        
        
        
        