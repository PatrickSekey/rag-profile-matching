"""
Quick start script for the RAG-based profile matching system
"""

import os
import sys
import json
from pathlib import Path

# Set up paths (works in both script and notebook)
try:
    current_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    current_dir = os.getcwd()

base_path = "/content/drive/MyDrive/RAG Based Profile matching"
utils_path = os.path.join(base_path, "utils")

# Add paths to sys.path
if base_path not in sys.path:
    sys.path.insert(0, base_path)
if utils_path not in sys.path:
    sys.path.insert(0, utils_path)
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from resume_rag import ResumeRAGSystem
from job_matcher import JobMatcher
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def quick_match(job_description: str, top_k: int = 10):
    """
    Quick function to match a job description against loaded resumes
    
    Args:
        job_description: Text of job description
        top_k: Number of top matches to return
    
    Returns:
        Dictionary with matching results
    """
    base_path = "/content/drive/MyDrive/RAG Based Profile matching"
    
    # Initialize systems
    rag_system = ResumeRAGSystem(base_path)
    
    # Load resumes if not loaded
    stats = rag_system.get_collection_stats()
    if 'total_chunks' not in stats or stats['total_chunks'] == 0:
        logger.info("Loading resumes into RAG system...")
        rag_system.load_all_resumes()
    
    # Create matcher and match
    matcher = JobMatcher(base_path)
    results = matcher.match_job_description(job_description, top_k=top_k)
    
    return results


def add_resume_file(file_path: str, target_filename: str = None):
    """
    Add a resume file to the system
    
    Args:
        file_path: Path to the resume file
        target_filename: Optional target filename in the system
    """
    base_path = "/content/drive/MyDrive/RAG Based Profile matching"
    rag_system = ResumeRAGSystem(base_path)
    
    if target_filename is None:
        target_filename = os.path.basename(file_path)
    
    rag_system.save_resume_to_directory(file_path, target_filename)
    result = rag_system.load_resume(os.path.join(rag_system.resumes_path, target_filename))
    
    return result


if __name__ == "__main__":
    # Example usage
    job_description = """
    We are seeking a Senior ML Engineer with strong Python skills and experience in NLP.
    """
    
    try:
        results = quick_match(job_description, top_k=5)
        print(json.dumps(results, indent=2))
    except Exception as e:
        logger.error(f"Error: {e}")
