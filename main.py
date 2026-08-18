"""
Main execution script for RAG-based profile matching system
"""

import os
import sys
import json
from pathlib import Path
import shutil
import logging

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

# Import modules
try:
    from resume_rag import ResumeRAGSystem
    from job_matcher import JobMatcher
    print("✅ Successfully imported required modules!")
except ImportError as e:
    print(f"❌ Import error: {e}")
    raise

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def setup_directories(base_path: str):
    """Create necessary directories"""
    paths = [
        base_path,
        os.path.join(base_path, "resumes"),
        os.path.join(base_path, "chroma_db"),
        os.path.join(base_path, "metadata"),
        os.path.join(base_path, "utils"),
        os.path.join(base_path, "output")
    ]
    for path in paths:
        os.makedirs(path, exist_ok=True)
    logger.info(f"Created directories at {base_path}")


def main():
    """Command line execution function"""
    base_path = "/content/drive/MyDrive/RAG Based Profile matching"

    # Setup directories
    setup_directories(base_path)

    logger.info("="*60)
    logger.info("RAG-Based Profile Matching System")
    logger.info("="*60)

    # Initialize RAG system
    logger.info("\n1. Initializing RAG System...")
    rag_system = ResumeRAGSystem(base_path)

    # Check if resumes exist
    resume_files = [f for f in os.listdir(rag_system.resumes_path)
                   if os.path.isfile(os.path.join(rag_system.resumes_path, f))]

    if len(resume_files) == 0:
        logger.warning("No resumes found in the resumes directory!")
        logger.info("Please add resume files (PDF, DOCX, or TXT) to:")
        logger.info(f"  {rag_system.resumes_path}")
        return

    # Load all resumes
    logger.info(f"\n2. Loading {len(resume_files)} resumes...")
    load_results = rag_system.load_all_resumes()

    if load_results['processed'] == 0:
        logger.error("No resumes were processed successfully!")
        logger.error("Please check file formats and try again.")
        return

    logger.info(f"Successfully loaded: {load_results['processed']} resumes")
    if load_results['failed'] > 0:
        logger.warning(f"Failed to load: {load_results['failed']} resumes")

    # Get collection stats
    stats = rag_system.get_collection_stats()
    logger.info(f"Collection stats: {stats}")

    # Initialize job matcher
    logger.info("\n3. Initializing Job Matcher...")
    matcher = JobMatcher(base_path)

    # Job Descriptions
    job_descriptions = [
        {
            "title": "Senior Machine Learning Engineer",
            "description": """
    Senior Machine Learning Engineer

    We are looking for an experienced Machine Learning Engineer with:
    - 5+ years of experience in machine learning and data science
    - Strong Python programming skills
    - Experience with deep learning frameworks (TensorFlow, PyTorch)
    - Knowledge of NLP and computer vision
    - Experience with cloud platforms (AWS, GCP)
    - Proficiency in SQL and data processing

    Preferred Skills:
    - MLOps experience
    - Docker and Kubernetes
    - Familiarity with CI/CD pipelines
    """
        },
        {
            "title": "Data Scientist",
            "description": """
    Data Scientist

    We are seeking a Data Scientist to join our analytics team:
    - 3+ years of experience in data science and analytics
    - Strong Python and R programming skills
    - Experience with data visualization tools (Tableau, Power BI)
    - Proficiency in SQL and data processing
    - Knowledge of statistical analysis and machine learning
    - Experience with A/B testing and experimental design

    Preferred Skills:
    - Experience with big data technologies (Spark, Hadoop)
    - Knowledge of cloud platforms (AWS, GCP)
    - Experience with data pipeline tools (Airflow, dbt)
    """
        },
        {
            "title": "Computer Vision Engineer",
            "description": """
    Computer Vision Engineer

    We are looking for a Computer Vision Engineer to develop CV solutions:
    - 3+ years of experience in computer vision and image processing
    - Strong Python programming skills
    - Experience with OpenCV, PyTorch, or TensorFlow
    - Knowledge of object detection, segmentation, and tracking
    - Experience with deep learning architectures (CNN, ResNet, YOLO)
    - Proficiency in image preprocessing and augmentation

    Preferred Skills:
    - Experience with 3D vision and point cloud processing
    - Knowledge of video processing and streaming
    - Familiarity with edge computing and model deployment
    """
        },
        {
            "title": "Full Stack Developer",
            "description": """
    Full Stack Developer

    We are seeking a Full Stack Developer to build web applications:
    - 4+ years of experience in full stack development
    - Strong proficiency in JavaScript, React, and Node.js
    - Experience with REST APIs and microservices
    - Knowledge of database systems (SQL, MongoDB)
    - Experience with version control (Git) and CI/CD
    - Understanding of responsive design and UI/UX principles

    Preferred Skills:
    - Experience with TypeScript and Next.js
    - Knowledge of cloud platforms (AWS, Azure)
    - Experience with Docker and containerization
    """
        },
        {
            "title": "Cybersecurity Analyst",
            "description": """
    Cybersecurity Analyst

    We are looking for a Cybersecurity Analyst to protect our systems:
    - 3+ years of experience in cybersecurity
    - Knowledge of intrusion detection and prevention systems
    - Experience with network security and firewalls
    - Proficiency in security monitoring and incident response
    - Understanding of vulnerability assessment and penetration testing
    - Knowledge of security frameworks (NIST, ISO 27001)

    Preferred Skills:
    - Experience with SIEM tools (Splunk, QRadar)
    - Knowledge of cloud security (AWS, Azure Security)
    - Certifications (CISSP, CEH, Security+)
    """
        },
        {
            "title": "DevOps Engineer",
            "description": """
    DevOps Engineer

    We are seeking a DevOps Engineer to manage our infrastructure:
    - 4+ years of experience in DevOps and cloud infrastructure
    - Strong experience with AWS, GCP, or Azure
    - Proficiency in Docker and Kubernetes
    - Experience with CI/CD pipelines (Jenkins, GitLab CI)
    - Knowledge of infrastructure as code (Terraform, CloudFormation)
    - Understanding of monitoring and logging tools (Prometheus, ELK)

    Preferred Skills:
    - Experience with microservices architecture
    - Knowledge of security best practices
    - Experience with automation and scripting (Python, Bash)
    """
        }
    ]

    logger.info("\n4. Matching 6 job descriptions against resumes...")
    
    all_results = {}
    
    for job in job_descriptions:
        title = job["title"]
        description = job["description"]
        
        logger.info(f"\n{'='*60}")
        logger.info(f"📋 Processing: {title}")
        logger.info(f"{'='*60}")
        
        # Perform matching
        results = matcher.match_job_description(description, top_k=10)

        if 'error' in results:
            logger.error(f"Error in matching: {results['error']}")
            continue

        # Display results
        logger.info(f"Top {len(results['top_matches'])} Matches:")
        logger.info("-"*40)

        for i, match in enumerate(results['top_matches'], 1):
            logger.info(f"\nRank {i}: {match['candidate_name']}")
            logger.info(f"  Match Score: {match['match_score']}/100")
            logger.info(f"  Matched Skills: {', '.join(match['matched_skills'])}")
            logger.info(f"  Experience: {match.get('experience_years', 'N/A')} years")
            logger.info(f"  Reasoning: {match['reasoning']}")
            logger.info("-"*40)

        # Save results to JSON
        output_filename = title.lower().replace(" ", "_") + "_results.json"
        output_path = os.path.join(base_path, "output", output_filename)
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"💾 Results saved to: {output_path}")
        
        all_results[title] = results

    # Save combined results
    combined_path = os.path.join(base_path, "output", "all_jobs_results.json")
    with open(combined_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"\n💾 Combined results saved to: {combined_path}")

    logger.info("\n" + "="*60)
    logger.info("✅ All 6 jobs processed successfully!")
    logger.info("="*60)


if __name__ == "__main__":
    main()
