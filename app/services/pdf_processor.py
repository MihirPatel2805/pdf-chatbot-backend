"""
PDF processing service for extracting text from PDF files.
"""

import PyPDF2
from io import BytesIO
from typing import List, Dict, Any, Optional
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from ..config import settings
from ..utils import (
    measure_time, 
    create_document_metadata, 
    clean_text,
    log_processing_info,
    handle_processing_error
)
import logging

logger = logging.getLogger(__name__)


class PDFProcessor:
    """Service for processing PDF files and extracting text."""
    
    def __init__(self):
        """Initialize the PDF processor."""
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap
        )
    
    @measure_time
    def extract_text_from_pdf(self, file_content: bytes, filename: str) -> List[Document]:
        """
        Extract text from PDF file content.
        
        Args:
            file_content: PDF file content as bytes
            filename: Name of the PDF file
            
        Returns:
            List of Document objects with extracted text
        """
        documents = []
        
        try:
            # Create BytesIO stream from file content
            pdf_stream = BytesIO(file_content)
            
            # Read PDF using PyPDF2
            pdf_reader = PyPDF2.PdfReader(pdf_stream)
            total_pages = len(pdf_reader.pages)
            
            log_processing_info("PDF extraction started", {
                "filename": filename,
                "total_pages": total_pages,
                "file_size": len(file_content)
            })
            
            # Extract text from each page
            for page_num, page in enumerate(pdf_reader.pages):
                try:
                    page_text = page.extract_text()
                    
                    if page_text and page_text.strip():
                        # Clean the text
                        cleaned_text = clean_text(page_text)
                        
                        if cleaned_text:
                            # Create document with metadata
                            metadata = create_document_metadata(
                                filename=filename,
                                page_num=page_num + 1,
                                total_pages=total_pages
                            )
                            
                            doc = Document(
                                page_content=cleaned_text,
                                metadata=metadata
                            )
                            documents.append(doc)
                    
                except Exception as page_error:
                    error_info = handle_processing_error(
                        "page_extraction",
                        page_error,
                        {"filename": filename, "page": page_num + 1}
                    )
                    logger.warning(f"Skipping page {page_num + 1}: {error_info}")
                    continue
            
            log_processing_info("PDF extraction completed", {
                "filename": filename,
                "documents_created": len(documents),
                "total_pages": total_pages
            })
            
            return documents
            
        except Exception as e:
            error_info = handle_processing_error(
                "pdf_extraction",
                e,
                {"filename": filename, "file_size": len(file_content)}
            )
            raise Exception(f"Failed to extract text from PDF {filename}: {error_info}")
    
    def process_multiple_pdfs(self, file_contents: List[Dict[str, Any]]) -> List[Document]:
        """
        Process multiple PDF files and extract text.
        
        Args:
            file_contents: List of dictionaries with 'filename' and 'content' keys
            
        Returns:
            List of all Document objects from all PDFs
        """
        all_documents = []
        
        for file_info in file_contents:
            filename = file_info['filename']
            content = file_info['content']
            
            try:
                documents = self.extract_text_from_pdf(content, filename)
                all_documents.extend(documents)
                
            except Exception as e:
                logger.error(f"Failed to process PDF {filename}: {str(e)}")
                # Continue processing other files
                continue
        
        log_processing_info("Multiple PDF processing completed", {
            "total_files": len(file_contents),
            "total_documents": len(all_documents)
        })
        
        return all_documents
    
    def split_documents_into_chunks(self, documents: List[Document]) -> List[Document]:
        """
        Split documents into smaller chunks for better vector search.
        
        Args:
            documents: List of Document objects
            
        Returns:
            List of chunked Document objects
        """
        try:
            chunks = self.text_splitter.split_documents(documents)
            
            # Add chunk-specific metadata
            for i, chunk in enumerate(chunks):
                chunk.metadata.update({
                    'chunk_id': f"chunk_{i}",
                    'chunk_index': i,
                    'total_chunks': len(chunks)
                })
            
            log_processing_info("Document chunking completed", {
                "original_documents": len(documents),
                "chunks_created": len(chunks)
            })
            
            return chunks
            
        except Exception as e:
            error_info = handle_processing_error(
                "document_chunking",
                e,
                {"document_count": len(documents)}
            )
            raise Exception(f"Failed to split documents into chunks: {error_info}")
    
    def validate_pdf_content(self, file_content: bytes, filename: str) -> bool:
        """
        Validate PDF file content.
        
        Args:
            file_content: PDF file content as bytes
            filename: Name of the file
            
        Returns:
            True if valid, False otherwise
        """
        try:
            # Check if content is not empty
            if not file_content or len(file_content) == 0:
                logger.warning(f"Empty file content for {filename}")
                return False
            
            # Try to read the PDF to validate it
            pdf_stream = BytesIO(file_content)
            pdf_reader = PyPDF2.PdfReader(pdf_stream)
            
            # Check if PDF has pages
            if len(pdf_reader.pages) == 0:
                logger.warning(f"PDF {filename} has no pages")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"PDF validation failed for {filename}: {str(e)}")
            return False
