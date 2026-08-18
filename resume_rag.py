"""
RAG System Setup - Part A
Implements document processing pipeline with ChromaDB vector database
"""

import os
import sys
import json
import time
from typing import List, Dict, Any, Optional
from pathlib import Path
import shutil
import logging

# Install chromadb if not available
try:
    import chromadb
    from chromadb.config import Settings
except ImportError:
    import subprocess
    print("📦 Installing chromadb...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "chromadb"])
    import chromadb
    from chromadb.config import Settings
    print("✅ chromadb installed successfully!")

# Set up the base path
base_path = "/content/drive/MyDrive/RAG Based Profile matching"
utils_path = os.path.join(base_path, "utils")

# Add both base path and utils path to sys.path
if base_path not in sys.path:
    sys.path.insert(0, base_path)
if utils_path not in sys.path:
    sys.path.insert(0, utils_path)

# Also add current directory
current_dir = os.getcwd()
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# Import from utils
try:
    from utils.document_parser import DocumentParser
    from utils.chunker import IntelligentChunker
    from utils.embeddings import EmbeddingGenerator
    from utils.metadata import MetadataExtractor
    print("✅ Successfully imported from utils!")
except ImportError as e:
    print(f"❌ Import error: {e}")
    print(f"   utils_path: {utils_path}")
    print(f"   Files in utils: {os.listdir(utils_path) if os.path.exists(utils_path) else 'Folder not found'}")
    raise

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def clean_metadata_value(value):
    """Clean metadata values to ensure they are valid for ChromaDB"""
    if value is None:
        return ""
    if isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, list):
        return ','.join([str(v) for v in value])
    if isinstance(value, dict):
        return str(value)
    return str(value)


class ResumeRAGSystem:
    """Complete RAG system for resume processing and retrieval"""
    
    def __init__(self, base_path: str = "/content/drive/MyDrive/RAG Based Profile matching"):
        """
        Initialize RAG system
        
        Args:
            base_path: Base directory path for all files
        """
        self.base_path = base_path
        self.resumes_path = os.path.join(base_path, "resumes")
        self.db_path = os.path.join(base_path, "chroma_db")
        self.metadata_path = os.path.join(base_path, "metadata")
        
        # Create directories
        for path in [self.resumes_path, self.db_path, self.metadata_path]:
            os.makedirs(path, exist_ok=True)
        
        # Initialize components
        self.parser = DocumentParser()
        self.chunker = IntelligentChunker(chunk_size=500, overlap=50)
        self.embedder = EmbeddingGenerator()
        self.metadata_extractor = MetadataExtractor()
        
        # Initialize ChromaDB
        self.client = chromadb.PersistentClient(
            path=self.db_path,
            settings=Settings(anonymized_telemetry=False)
        )
        
        # Get or create collection
        self.collection_name = "resume_collection"
        self.collection = self._get_or_create_collection()
        
        logger.info(f"RAG System initialized at {base_path}")
    
    def _get_or_create_collection(self) -> chromadb.Collection:
        """Get existing collection or create new one"""
        try:
            collection = self.client.get_collection(self.collection_name)
            logger.info(f"Loaded existing collection: {self.collection_name}")
            return collection
        except:
            collection = self.client.create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            logger.info(f"Created new collection: {self.collection_name}")
            return collection
    
    def load_resume(self, file_path: str) -> Dict:
        """
        Load and process a single resume with error handling
        """
        try:
            # Check if file exists
            if not os.path.exists(file_path):
                return {
                    'status': 'error',
                    'file': os.path.basename(file_path),
                    'error': 'File not found'
                }
            
            # Parse document
            logger.info(f"Processing: {os.path.basename(file_path)}")
            doc_data = self.parser.parse_document(file_path)
            text = doc_data['text']
            
            # Check if we got meaningful text
            if not text or len(text.strip()) < 50:
                logger.warning(f"Low text extraction ({len(text)} chars) for {file_path}")
                return {
                    'status': 'skipped',
                    'file': os.path.basename(file_path),
                    'error': f'Insufficient text extracted: {len(text)} characters'
                }
            
            # Extract metadata
            metadata = self.metadata_extractor.extract_metadata(text)
            
            # Get file info
            file_info = self.parser.get_file_metadata(file_path)
            
            # Chunk document
            chunks = self.chunker.chunk_document(
                text, 
                metadata={**metadata, **file_info}
            )
            
            if not chunks:
                return {
                    'status': 'error',
                    'file': os.path.basename(file_path),
                    'error': 'No chunks created from document'
                }
            
            # Generate embeddings for all chunks
            chunk_texts = [chunk['text'] for chunk in chunks]
            embeddings = self.embedder.get_embeddings(chunk_texts)
            
            # Prepare data for ChromaDB
            ids = []
            documents = []
            metadatas = []
            
            for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                chunk_id = f"{file_info['filename']}_chunk_{i}"
                ids.append(chunk_id)
                documents.append(chunk['text'])
                
                # Create metadata for this chunk with cleaned values
                chunk_metadata = {
                    'filename': clean_metadata_value(file_info['filename']),
                    'file_path': clean_metadata_value(file_info['file_path']),
                    'file_type': clean_metadata_value(doc_data.get('file_type', 'unknown')),
                    'chunk_index': clean_metadata_value(i),
                    'section': clean_metadata_value(chunk.get('section', 'general')),
                    'candidate_name': clean_metadata_value(metadata.get('name', 'Unknown')),
                    'skills': clean_metadata_value(','.join(metadata.get('skills', []))),
                    'experience_years': clean_metadata_value(str(metadata.get('experience_years', 0))),
                    'education': clean_metadata_value('||'.join(metadata.get('education', []))),
                    'email': clean_metadata_value(metadata.get('email', '')),
                    'phone': clean_metadata_value(metadata.get('phone', '')),
                    'total_chunks': clean_metadata_value(len(chunks))
                }
                metadatas.append(chunk_metadata)
            
            # Store in ChromaDB
            self.collection.add(
                embeddings=embeddings.tolist(),
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )
            
            # Save metadata separately
            metadata_file = os.path.join(
                self.metadata_path,
                f"{file_info['filename']}_metadata.json"
            )
            with open(metadata_file, 'w') as f:
                json.dump({
                    'file_info': file_info,
                    'metadata': metadata,
                    'chunks': chunks,
                    'chunk_ids': ids,
                    'extraction_method': doc_data.get('method', 'unknown')
                }, f, indent=2)
            
            logger.info(f"✅ Successfully processed: {file_info['filename']}")
            return {
                'status': 'success',
                'file': file_info['filename'],
                'metadata': metadata,
                'chunks': len(chunks),
                'embeddings': len(embeddings),
                'method': doc_data.get('method', 'unknown')
            }
            
        except Exception as e:
            logger.error(f"❌ Error processing resume {file_path}: {str(e)}")
            return {
                'status': 'error',
                'file': os.path.basename(file_path),
                'error': str(e)
            }
    
    def load_all_resumes(self) -> Dict:
        """
        Load and process all resumes in the resumes directory
        
        Returns:
            Dictionary with processing summary
        """
        results = {
            'total_files': 0,
            'processed': 0,
            'failed': 0,
            'skipped': 0,
            'errors': [],
            'details': []
        }
        
        # Get all files in resumes directory
        files = [f for f in os.listdir(self.resumes_path) 
                if os.path.isfile(os.path.join(self.resumes_path, f))]
        
        results['total_files'] = len(files)
        
        for filename in files:
            file_path = os.path.join(self.resumes_path, filename)
            result = self.load_resume(file_path)
            
            if result['status'] == 'success':
                results['processed'] += 1
                results['details'].append(result)
            elif result['status'] == 'skipped':
                results['skipped'] += 1
                results['errors'].append(f"⚠️ Skipped: {filename} - {result.get('error', 'Unknown')}")
            else:
                results['failed'] += 1
                results['errors'].append(f"❌ Failed: {filename} - {result.get('error', 'Unknown error')}")
        
        logger.info(f"✅ Loaded: {results['processed']} resumes")
        logger.info(f"⚠️ Skipped: {results['skipped']} resumes")
        logger.info(f"❌ Failed: {results['failed']} resumes")
        return results
    
    def get_collection_stats(self) -> Dict:
        """Get statistics about the collection"""
        try:
            count = self.collection.count()
            return {
                'total_chunks': count,
                'collection_name': self.collection_name,
                'embedding_dimension': self.embedder.get_embedding_dimension()
            }
        except Exception as e:
            logger.error(f"Error getting collection stats: {str(e)}")
            return {'error': str(e)}
    
    def clear_collection(self):
        """Clear the collection"""
        try:
            self.client.delete_collection(self.collection_name)
            self.collection = self._get_or_create_collection()
            logger.info("Collection cleared")
        except Exception as e:
            logger.error(f"Error clearing collection: {str(e)}")
    
    def save_resume_to_directory(self, file_path: str, target_filename: Optional[str] = None):
        """
        Save a resume file to the resumes directory
        
        Args:
            file_path: Path to source file
            target_filename: Optional target filename
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        if target_filename is None:
            target_filename = os.path.basename(file_path)
        
        target_path = os.path.join(self.resumes_path, target_filename)
        shutil.copy2(file_path, target_path)
        logger.info(f"Saved resume to: {target_path}")


# Test the RAG system
if __name__ == "__main__":
    print("\n" + "="*60)
    print("🚀 Testing ResumeRAGSystem with Robust PDF Support")
    print("="*60)
    
    # Initialize
    rag = ResumeRAGSystem()
    print("✅ RAG System initialized successfully!")
    
    # Check if there are resumes
    resume_files = [f for f in os.listdir(rag.resumes_path) 
                   if os.path.isfile(os.path.join(rag.resumes_path, f))]
    
    if resume_files:
        print(f"\n📄 Found {len(resume_files)} resume files:")
        for f in resume_files[:10]:  # Show first 10
            print(f"   - {f}")
        if len(resume_files) > 10:
            print(f"   ... and {len(resume_files) - 10} more")
        
        print("\n🔄 Loading resumes...")
        results = rag.load_all_resumes()
        print(f"\n✅ Loaded: {results['processed']} resumes")
        print(f"⚠️ Skipped: {results['skipped']} resumes")
        print(f"❌ Failed: {results['failed']} resumes")
        
        if results['errors']:
            print("\n📋 First 5 errors:")
            for error in results['errors'][:5]:
                print(f"   {error}")
    else:
        print("\n⚠️ No resume files found in the resumes directory.")
        print(f"   Please add resumes to: {rag.resumes_path}")
    
    # Get stats
    print("\n📊 Collection Statistics:")
    stats = rag.get_collection_stats()
    for key, value in stats.items():
        print(f"   - {key}: {value}")
    
    print("\n" + "="*60)
    print("✅ ResumeRAGSystem is ready to use!")
    print("="*60)
