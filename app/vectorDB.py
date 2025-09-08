import os
from typing import List, Dict, Any
from uuid import uuid4
import PyPDF2
from io import BytesIO
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_qdrant import QdrantVectorStore
from langchain_community.document_loaders import PyPDFLoader, PyPDFDirectoryLoader
from langchain_core.documents import Document
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def convert_pdf_to_vector_db_from_memory(file_contents: List[Dict[str, Any]], index_name: str):
    """
    Convert PDF files to vector database using Google Gemini embeddings.
    Processes PDFs directly from memory without storing them on disk.
    
    Args:
        file_contents: List of dictionaries with 'filename' and 'content' (bytes)
        index_name: Name of the Qdrant collection
    """
    print(f"Converting PDFs to vector DB from memory")
    print(f"Index name: {index_name}")
    print(f"Processing {len(file_contents)} files")
    
    try:
        all_documents = []
        
        # Process each PDF file from memory
        for file_info in file_contents:
            filename = file_info['filename']
            file_bytes = file_info['content']
            
            print(f"Processing file: {filename}")
            
            # Create a BytesIO object from the file content
            pdf_stream = BytesIO(file_bytes)
            
            # Use PyPDF2 directly to extract text from PDF bytes
            try:
                pdf_reader = PyPDF2.PdfReader(pdf_stream)
                documents = []
                
                # Extract text from each page
                for page_num, page in enumerate(pdf_reader.pages):
                    try:
                        page_text = page.extract_text()
                        if page_text.strip():  # Only add non-empty pages
                            doc = Document(
                                page_content=page_text,
                                metadata={
                                    'source': filename,
                                    'filename': filename,
                                    'page': page_num + 1,
                                    'total_pages': len(pdf_reader.pages)
                                }
                            )
                            documents.append(doc)
                    except Exception as page_error:
                        print(f"Error extracting text from page {page_num + 1} of {filename}: {str(page_error)}")
                        continue
                
                all_documents.extend(documents)
                print(f"Loaded {len(documents)} pages from {filename}")
                
            except Exception as pdf_error:
                print(f"Error processing PDF {filename}: {str(pdf_error)}")
                continue
        
        print(f"Total documents loaded: {len(all_documents)}")
        
        # Split the documents into chunks using langchain
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        text_chunks = text_splitter.split_documents(all_documents)
        print(f"Text split into {len(text_chunks)} chunks")
            # Generate vector embeddings using langchain gemini embeddings
        google_api_key = os.getenv("GOOGLE_API_KEY")
        if not google_api_key:
            raise ValueError("GOOGLE_API_KEY not found in environment variables. Please create a .env file with your API key.")
        
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", api_key=google_api_key)
        print(f"Embeddings model initialized: {embeddings}")
        
        # vector_embeddings = embeddings.embed_documents(text_chunks)
        # print(f"Generated {len(vector_embeddings)} embeddings")
            # Store the vector embeddings in the vector database
        # Note: This requires proper Pinecone configuration with API key and environment
        # Store the vector embeddings in Qdrant (Optional)
        # Note: You need to set QDRANT_URL and QDRANT_API_KEY environment variables
        qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
        qdrant_api_key = os.getenv("QDRANT_API_KEY")
        
        if qdrant_url:
            try:
                # Initialize Qdrant client
                if qdrant_api_key:
                    client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
                else:
                    client = QdrantClient(url=qdrant_url)
                
                print(f"Qdrant client initialized successfully")
                
                # Create or get the collection
                try:
                    client.get_collection(index_name)
                    print(f"Qdrant collection already exists: {index_name}")
                except Exception:
                    # Collection doesn't exist, create it
                    client.create_collection(
                        collection_name=index_name,
                        vectors_config=VectorParams(size=768, distance=Distance.COSINE)
                    )
                    print(f"Qdrant collection created: {index_name}")
                
                # Store documents in Qdrant using the Document-based approach with UUIDs
                print(f"Storing {len(text_chunks)} document chunks in Qdrant")
                
                # Create vector store
                vector_store = QdrantVectorStore(
                    client=client,
                    collection_name=index_name,
                    embedding=embeddings
                )
                print(f"Vector store created successfully")
                
                # Generate UUIDs for each document chunk
                uuids = [str(uuid4()) for _ in range(len(text_chunks))]
                print(f"Generated {len(uuids)} UUIDs for document chunks")
                
                # Add documents with UUIDs
                vector_store.add_documents(documents=text_chunks, ids=uuids)
                print(f"Vector embeddings stored in Qdrant collection: {index_name}")
            except Exception as qdrant_error:
                print(f"Qdrant storage failed: {str(qdrant_error)}")
                print("Continuing without vector storage...")
        else:
            print("Qdrant URL not set. Skipping vector storage.")
            print("Set QDRANT_URL environment variable to enable storage.")
            
        
        print(f"All files processed successfully")
        return {"status": "PDFs processed successfully", "files_processed": len(file_contents), "index": index_name}
        
    except Exception as e:
        print(f"Error processing PDFs: {str(e)}")
        return {"status": "Error", "error": str(e), "files_processed": len(file_contents), "index": index_name}

def convert_pdf_to_vector_db(file_path: str, index_name: str):
    """
    Legacy function for backward compatibility.
    This function is deprecated - use convert_pdf_to_vector_db_from_memory instead.
    """
    print("WARNING: Using deprecated file-based processing. Consider using convert_pdf_to_vector_db_from_memory instead.")
    
    try:
        print(f"Processing file: {file_path}")
        # Read the PDF file using PyPDFDirectoryLoader
        file_loader = PyPDFDirectoryLoader(file_path)
        print(f"File loader: {file_loader}")
        documents = file_loader.load()
        print(f"Loaded {len(documents)} documents from PDF")
        
        # Split the documents into chunks using langchain
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        text_chunks = text_splitter.split_documents(documents)
        print(f"Text split into {len(text_chunks)} chunks")
        
        # Generate vector embeddings using langchain gemini embeddings
        google_api_key = os.getenv("GOOGLE_API_KEY")
        if not google_api_key:
            raise ValueError("GOOGLE_API_KEY not found in environment variables. Please create a .env file with your API key.")
        
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", api_key=google_api_key)
        print(f"Embeddings model initialized: {embeddings}")
        
        # Store the vector embeddings in Qdrant (Optional)
        qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
        qdrant_api_key = os.getenv("QDRANT_API_KEY")
        
        if qdrant_url:
            try:
                # Initialize Qdrant client
                if qdrant_api_key:
                    client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
                else:
                    client = QdrantClient(url=qdrant_url)
                
                print(f"Qdrant client initialized successfully")
                
                # Create or get the collection
                try:
                    client.get_collection(index_name)
                    print(f"Qdrant collection already exists: {index_name}")
                except Exception:
                    # Collection doesn't exist, create it
                    client.create_collection(
                        collection_name=index_name,
                        vectors_config=VectorParams(size=768, distance=Distance.COSINE)
                    )
                    print(f"Qdrant collection created: {index_name}")
                
                # Store documents in Qdrant using the Document-based approach with UUIDs
                print(f"Storing {len(text_chunks)} document chunks in Qdrant")
                
                # Create vector store
                vector_store = QdrantVectorStore(
                    client=client,
                    collection_name=index_name,
                    embedding=embeddings
                )
                print(f"Vector store created successfully")
                
                # Generate UUIDs for each document chunk
                uuids = [str(uuid4()) for _ in range(len(text_chunks))]
                print(f"Generated {len(uuids)} UUIDs for document chunks")
                
                # Add documents with UUIDs
                vector_store.add_documents(documents=text_chunks, ids=uuids)
                print(f"Vector embeddings stored in Qdrant collection: {index_name}")
            except Exception as qdrant_error:
                print(f"Qdrant storage failed: {str(qdrant_error)}")
                print("Continuing without vector storage...")
        else:
            print("Qdrant URL not set. Skipping vector storage.")
            print("Set QDRANT_URL environment variable to enable storage.")
            
        print(f"All files processed successfully")
        return {"status": "PDFs processed successfully", "files": file_path, "index": index_name}
        
    except Exception as e:
        print(f"Error processing PDFs: {str(e)}")
        return {"status": "Error", "error": str(e), "files": file_path, "index": index_name}

def query_pdf_documents(question: str, index_name: str = "pdf-chatbot-index"):
    """
    Query the vector database to find relevant documents and generate an answer.
    """
    try:
        print(f"Querying documents for question: {question}")
        
        # Get Google API key
        google_api_key = os.getenv("GOOGLE_API_KEY")
        if not google_api_key:
            raise ValueError("GOOGLE_API_KEY not found in environment variables.")
        
        # Initialize embeddings
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", api_key=google_api_key)
        print("Embeddings model initialized for querying",embeddings)
        # Get Qdrant configuration
        qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
        qdrant_api_key = os.getenv("QDRANT_API_KEY")
        
        if not qdrant_url:
            return {
                "answer": "Qdrant is not configured. Please set QDRANT_URL to enable document querying.",
                "sources": []
            }
        
        # Initialize Qdrant client
        if qdrant_api_key:
            client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
        else:
            client = QdrantClient(url=qdrant_url)
        
        print(client, 'Qdrant client initialized for querying')
        
        # Create vector store
        vector_store = QdrantVectorStore(
            client=client,
            collection_name=index_name,
            embedding=embeddings
        )
        print(vector_store, "Vector store initialized for querying")
        # Search for relevant documents
        docs = vector_store.similarity_search(question, k=3)
        print(f"Found {len(docs)} relevant documents")
        
        if not docs:
            return {
                "answer": "I couldn't find any relevant information in the uploaded documents to answer your question.",
                "sources": []
            }
        
        # Combine the relevant document content
        context = "\n\n".join([doc.page_content for doc in docs])
        print("Context for LLM:",context)
        # Generate answer using Gemini LLM
        # Initialize Gemini chat model (using free model)
        llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            api_key=google_api_key,
            temperature=0.1
        )
        print("LLM model initialized:",llm)
        # Create a prompt for the LLM
        prompt = f"""
        Based on the following context from uploaded documents, please answer the user's question.
        
        User Question: {question}
        
        Context from documents:
        {context}
        
        Please provide a clear, accurate, and helpful answer based on the context provided. 
        If the context doesn't contain enough information to answer the question, please say so.
        Keep your answer concise but informative.
        """
        
        # Generate response using Gemini
        response = llm.invoke(prompt,max_tokens=500)
        print(response,"LLM response")
        answer = response.content
        print(f"Generated answer: {answer}")
        # Extract sources
        sources = []
        for doc in docs:
            if hasattr(doc, 'metadata') and doc.metadata:
                source = doc.metadata.get('source', 'Unknown source')
                sources.append(source)
        
        return {
            "answer": answer,
            "sources": list(set(sources))  # Remove duplicates
        }
        
    except Exception as e:
        print(f"Error querying documents: {str(e)}")
        return {
            "answer": f"Sorry, I encountered an error while processing your question: {str(e)}",
            "sources": []
        }
                
            
        
        
        
        
        
        
        
        
        