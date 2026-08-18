"""
Streamlit App for RAG-Based Profile Matching System
Run with: streamlit run app.py
"""

import os
import sys
import json
import streamlit as st
import pandas as pd
from datetime import datetime
import shutil

# Set up paths
base_path = "/content/drive/MyDrive/RAG Based Profile matching"
utils_path = os.path.join(base_path, "utils")

if base_path not in sys.path:
    sys.path.insert(0, base_path)
if utils_path not in sys.path:
    sys.path.insert(0, utils_path)

# Import only the needed functions
try:
    from resume_rag import ResumeRAGSystem
    from job_matcher import JobMatcher
    print("✅ Successfully imported required modules!")
except ImportError as e:
    st.error(f"Import error: {e}")
    st.stop()

# Set page config
st.set_page_config(
    page_title="RAG-Based Profile Matching System",
    page_icon="📄",
    layout="wide"
)

# Initialize session state
if 'results' not in st.session_state:
    st.session_state.results = None
if 'resume_count' not in st.session_state:
    st.session_state.resume_count = 0


def save_uploaded_file(uploaded_file, save_path):
    """Save uploaded file to the resumes directory"""
    try:
        with open(save_path, 'wb') as f:
            f.write(uploaded_file.getbuffer())
        return True
    except Exception as e:
        st.error(f"Error saving file: {str(e)}")
        return False


def add_sample_resume(base_path: str, content: str, filename: str = "sample_resume.txt"):
    """Helper function to add a sample resume"""
    resume_path = os.path.join(base_path, "resumes", filename)
    with open(resume_path, 'w') as f:
        f.write(content)


def run_matching(job_description, top_k=10):
    """Run the matching process and return results"""
    base_path = "/content/drive/MyDrive/RAG Based Profile matching"

    # Initialize RAG system
    rag_system = ResumeRAGSystem(base_path)

    # Check if there are resumes
    resume_files = [f for f in os.listdir(rag_system.resumes_path)
                   if os.path.isfile(os.path.join(rag_system.resumes_path, f))]

    if len(resume_files) == 0:
        return {"error": "No resumes found. Please upload resumes first."}

    # Load resumes if not loaded
    stats = rag_system.get_collection_stats()
    if 'total_chunks' not in stats or stats['total_chunks'] == 0:
        rag_system.load_all_resumes()

    # Initialize job matcher
    matcher = JobMatcher(base_path)

    # Perform matching
    results = matcher.match_job_description(job_description, top_k=top_k)

    return results


# Title
st.title("📄 RAG-Based Profile Matching System")
st.markdown("---")

# Sidebar for resume management
with st.sidebar:
    st.header("📁 Resume Management")

    base_path = "/content/drive/MyDrive/RAG Based Profile matching"
    resume_dir = os.path.join(base_path, "resumes")
    os.makedirs(resume_dir, exist_ok=True)

    resume_files = [f for f in os.listdir(resume_dir)
                   if os.path.isfile(os.path.join(resume_dir, f))]
    st.info(f"📊 Current resumes: {len(resume_files)}")

    # Upload resumes
    st.subheader("Upload Resumes")
    uploaded_files = st.file_uploader(
        "Choose resume files (PDF, DOCX, TXT)",
        type=['pdf', 'docx', 'txt'],
        accept_multiple_files=True
    )

    if uploaded_files:
        if st.button("📤 Upload Selected Files"):
            with st.spinner("Uploading resumes..."):
                saved_count = 0
                for uploaded_file in uploaded_files:
                    save_path = os.path.join(resume_dir, uploaded_file.name)
                    if save_uploaded_file(uploaded_file, save_path):
                        saved_count += 1

                if saved_count > 0:
                    st.success(f"✅ Successfully uploaded {saved_count} resumes!")
                    st.session_state.resume_count = len(os.listdir(resume_dir))
                    st.session_state.results = None
                    st.rerun()
                else:
                    st.error("❌ Failed to upload files. Please try again.")

    # Add sample resumes
    st.subheader("📝 Add Sample Resumes")
    if st.button("Add Sample Resumes"):
        with st.spinner("Adding sample resumes..."):
            sample_resume1 = """JOHN DOE
john.doe@email.com | (555) 123-4567

PROFESSIONAL SUMMARY
Senior Machine Learning Engineer with 8 years of experience in building and deploying ML solutions.
Expert in Python, deep learning, and NLP applications.

TECHNICAL SKILLS
Python, TensorFlow, PyTorch, Scikit-learn, NLP, Computer Vision, AWS, Docker, Kubernetes, SQL, Git

WORK EXPERIENCE
Senior ML Engineer | TechCorp Inc. | 2020-Present
- Led development of NLP models for document processing
- Implemented ML pipelines using TensorFlow and PyTorch
- Deployed models on AWS using Docker and Kubernetes

EDUCATION
Master of Science in Computer Science, Stanford University, 2018
"""

            sample_resume2 = """JANE SMITH
jane.smith@email.com | (555) 987-6543

PROFESSIONAL SUMMARY
Data Scientist with 6 years of experience in analytics and machine learning.
Proficient in Python, R, SQL, and data visualization tools.

TECHNICAL SKILLS
Python, R, SQL, Tableau, Power BI, Pandas, Scikit-learn, Deep Learning, AWS

WORK EXPERIENCE
Senior Data Scientist | AnalyticsPro | 2019-Present
- Built predictive models for customer churn
- Created dashboards using Tableau and Power BI

EDUCATION
Master of Science in Data Science, UC Berkeley, 2019
"""

            add_sample_resume(base_path, sample_resume1, "john_doe_resume.txt")
            add_sample_resume(base_path, sample_resume2, "jane_smith_resume.txt")
            st.success("✅ Sample resumes added successfully!")
            st.session_state.resume_count = len(os.listdir(resume_dir))
            st.session_state.results = None
            st.rerun()

    # Clear all resumes
    if st.button("🗑️ Clear All Resumes"):
        with st.spinner("Clearing resumes..."):
            for file in os.listdir(resume_dir):
                file_path = os.path.join(resume_dir, file)
                if os.path.isfile(file_path):
                    os.remove(file_path)
            # Also clear chroma DB
            chroma_path = os.path.join(base_path, "chroma_db")
            if os.path.exists(chroma_path):
                shutil.rmtree(chroma_path)
                os.makedirs(chroma_path, exist_ok=True)
            st.success("✅ All resumes cleared!")
            st.session_state.resume_count = 0
            st.session_state.results = None
            st.rerun()

    # Display resume list
    if resume_files:
        st.subheader("📋 Resume List")
        for file in resume_files:
            st.text(f"📄 {file}")

# Main content area
col1, col2 = st.columns([2, 1])

with col1:
    st.header("🔍 Job Description")
    job_description = st.text_area(
        "Enter the job description",
        height=200,
        placeholder="Paste the job description here..."
    )

    col_k, col_button = st.columns([1, 2])
    with col_k:
        top_k = st.number_input("Top K matches", min_value=1, max_value=50, value=10)
    with col_button:
        if st.button("🔍 Find Matches", type="primary"):
            if not job_description.strip():
                st.warning("Please enter a job description.")
            else:
                resume_count = len([f for f in os.listdir(resume_dir)
                                  if os.path.isfile(os.path.join(resume_dir, f))])
                if resume_count == 0:
                    st.warning("Please upload resumes first.")
                else:
                    with st.spinner("Processing..."):
                        results = run_matching(job_description, top_k)
                        st.session_state.results = results

with col2:
    st.header("📊 Quick Stats")
    if st.session_state.results and 'error' not in st.session_state.results:
        results = st.session_state.results
        st.metric("Total Candidates", results.get('total_candidates_evaluated', 0))
        st.metric("Top Matches", len(results.get('top_matches', [])))
        st.metric("Required Skills", len(results.get('job_skills_required', [])))
    else:
        st.info("No results yet. Run a match to see statistics.")

# Display results
if st.session_state.results:
    st.markdown("---")
    st.header("📊 Matching Results")

    results = st.session_state.results

    if 'error' in results:
        st.error(f"❌ Error: {results['error']}")
    else:
        if results.get('job_skills_required'):
            st.subheader("🎯 Required Skills")
            st.write(", ".join(results['job_skills_required']))

        if results.get('top_matches'):
            st.subheader(f"🏆 Top {len(results['top_matches'])} Matches")

            match_data = []
            for i, match in enumerate(results['top_matches'], 1):
                match_data.append({
                    "Rank": i,
                    "Candidate": match['candidate_name'],
                    "Match Score": f"{match['match_score']}/100",
                    "Experience": f"{match.get('experience_years', 'N/A')} years",
                    "Matched Skills": ", ".join(match.get('matched_skills', [])[:5])
                })

            df = pd.DataFrame(match_data)
            st.dataframe(df, use_container_width=True)

            for i, match in enumerate(results['top_matches'], 1):
                with st.expander(f"📄 Match #{i}: {match['candidate_name']} - Score: {match['match_score']}/100"):
                    col_score, col_exp = st.columns(2)
                    with col_score:
                        st.metric("Match Score", f"{match['match_score']}/100")
                        st.metric("Semantic Score", f"{match.get('semantic_score', 0)}/100")
                        st.metric("Keyword Score", f"{match.get('keyword_score', 0)}/100")
                    with col_exp:
                        st.metric("Experience", f"{match.get('experience_years', 'N/A')} years")
                        st.metric("Matched Skills", len(match.get('matched_skills', [])))

                    st.subheader("Matched Skills")
                    if match.get('matched_skills'):
                        st.write(", ".join(match['matched_skills']))
                    else:
                        st.write("No specific skills matched")

                    st.subheader("Reasoning")
                    st.write(match.get('reasoning', 'No reasoning available'))

            st.download_button(
                label="📥 Download Results (JSON)",
                data=json.dumps(results, indent=2),
                file_name=f"matching_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )

print("✅ app.py created successfully!")
