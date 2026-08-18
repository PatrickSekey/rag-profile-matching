"""
Job Matching Engine - Part B
Semantic search and ranking for job-resume matching
"""

import os
import sys
import json
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from datetime import datetime
import re

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

# Set up the base path (works in both script and notebook)
try:
    current_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    current_dir = os.getcwd()

# Add paths
base_path = "/content/drive/MyDrive/RAG Based Profile matching"
utils_path = os.path.join(base_path, "utils")

if base_path not in sys.path:
    sys.path.insert(0, base_path)
if utils_path not in sys.path:
    sys.path.insert(0, utils_path)
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# Import from utils and resume_rag
try:
    from utils.embeddings import EmbeddingGenerator
    from resume_rag import ResumeRAGSystem
    print("✅ Successfully imported required modules!")
except ImportError as e:
    print(f"❌ Import error: {e}")
    print(f"   utils_path: {utils_path}")
    print(f"   Files in utils: {os.listdir(utils_path) if os.path.exists(utils_path) else 'Folder not found'}")
    raise

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class JobMatcher:
    """Job matching engine for semantic resume retrieval and ranking"""

    def __init__(self, base_path: str = "/content/drive/MyDrive/RAG Based Profile matching"):
        """
        Initialize job matcher

        Args:
            base_path: Base directory path
        """
        self.base_path = base_path
        self.rag_system = ResumeRAGSystem(base_path)
        self.embedder = self.rag_system.embedder

        # Connect to ChromaDB
        self.db_path = os.path.join(base_path, "chroma_db")
        self.client = chromadb.PersistentClient(
            path=self.db_path,
            settings=Settings(anonymized_telemetry=False)
        )

        self.collection = self.client.get_collection(
            self.rag_system.collection_name
        )

        # Required skill threshold
        self.required_skill_threshold = 0.85  # Similarity threshold for required skills

        logger.info("JobMatcher initialized")

    def match_job_description(self, job_description: str, top_k: int = 10) -> Dict:
        """
        Match job description against resumes and return ranked results

        Args:
            job_description: Job description text
            top_k: Number of top matches to return

        Returns:
            Dictionary with job matches and reasoning
        """
        try:
            # Extract required skills from job description
            required_skills = self._extract_required_skills(job_description)

            # Get job embedding
            job_embedding = self.embedder.get_embeddings(job_description)

            # Perform semantic search
            semantic_results = self._semantic_search(job_embedding, top_k * 2)

            # Perform keyword search for required skills
            keyword_results = self._keyword_search(job_description, required_skills)

            # Hybrid search - combine semantic and keyword results
            hybrid_results = self._hybrid_search(semantic_results, keyword_results)

            # Rank and score candidates
            ranked_results = self._rank_candidates(
                hybrid_results,
                job_description,
                required_skills
            )

            # Get top K results
            top_matches = ranked_results[:top_k]

            # Generate match reasoning
            for match in top_matches:
                match['reasoning'] = self._generate_reasoning(
                    match,
                    job_description,
                    required_skills
                )
                match['relevant_excerpts'] = self._get_relevant_excerpts(
                    match['metadata']['file_path'],
                    match['metadata']['chunk_index']
                )

            # Prepare response
            response = {
                "job_description": job_description[:200] + "...",  # Truncated for output
                "job_skills_required": required_skills,
                "top_matches": top_matches,
                "total_candidates_evaluated": len(hybrid_results),
                "matching_timestamp": datetime.now().isoformat()
            }

            logger.info(f"Found {len(top_matches)} top matches from {len(hybrid_results)} candidates")
            return response

        except Exception as e:
            logger.error(f"Error matching job description: {str(e)}")
            return {
                "error": str(e),
                "job_description": job_description[:200] + "...",
                "top_matches": []
            }

    def _semantic_search(self, query_embedding: np.ndarray, top_k: int = 20) -> List[Dict]:
        """Perform semantic search using vector similarity"""
        try:
            results = self.collection.query(
                query_embeddings=query_embedding.tolist(),
                n_results=top_k,
                include=['documents', 'metadatas', 'distances']
            )

            # Format results
            formatted_results = []
            for i in range(len(results['ids'][0])):
                formatted_results.append({
                    'id': results['ids'][0][i],
                    'document': results['documents'][0][i],
                    'metadata': results['metadatas'][0][i],
                    'distance': results['distances'][0][i],
                    'similarity': 1 - results['distances'][0][i],  # Convert distance to similarity
                })

            return formatted_results

        except Exception as e:
            logger.error(f"Error in semantic search: {str(e)}")
            return []

    def _keyword_search(self, job_description: str, required_skills: List[str]) -> List[Dict]:
        """Perform keyword-based search for required skills"""
        try:
            # Get all documents from collection
            all_docs = self.collection.get(include=['documents', 'metadatas'])

            keyword_results = []
            for i in range(len(all_docs['ids'])):
                doc_text = all_docs['documents'][i].lower()
                metadata = all_docs['metadatas'][i]

                # Count skill matches
                skill_matches = []
                for skill in required_skills:
                    if skill.lower() in doc_text:
                        skill_matches.append(skill)

                if skill_matches:
                    keyword_results.append({
                        'id': all_docs['ids'][i],
                        'metadata': metadata,
                        'skill_matches': skill_matches,
                        'match_count': len(skill_matches)
                    })

            # Sort by number of skill matches
            keyword_results.sort(key=lambda x: x['match_count'], reverse=True)

            return keyword_results

        except Exception as e:
            logger.error(f"Error in keyword search: {str(e)}")
            return []

    def _hybrid_search(self, semantic_results: List[Dict], keyword_results: List[Dict]) -> List[Dict]:
        """Combine semantic and keyword search results"""
        # Create lookup by chunk ID
        combined = {}

        # Add semantic results
        for result in semantic_results:
            chunk_id = result['id']
            combined[chunk_id] = {
                'id': chunk_id,
                'metadata': result['metadata'],
                'similarity_score': result['similarity'],
                'keyword_matches': [],
                'keyword_score': 0,
                'semantic_score': result['similarity'] * 100,  # Convert to 0-100 scale
                'document': result['document']
            }

        # Add keyword results
        for result in keyword_results:
            chunk_id = result['id']
            if chunk_id in combined:
                combined[chunk_id]['keyword_matches'] = result['skill_matches']
                # Keyword score based on number of matches (max 10 skills)
                combined[chunk_id]['keyword_score'] = min(result['match_count'] * 8, 80)
            else:
                combined[chunk_id] = {
                    'id': chunk_id,
                    'metadata': result['metadata'],
                    'similarity_score': 0,
                    'keyword_matches': result['skill_matches'],
                    'keyword_score': min(result['match_count'] * 8, 80),
                    'semantic_score': 0,
                    'document': ''
                }

        # Calculate hybrid score
        for chunk_id, data in combined.items():
            # Hybrid score: 60% semantic, 40% keyword
            data['hybrid_score'] = (0.6 * data['semantic_score'] +
                                   0.4 * data['keyword_score'])
            data['candidate_name'] = data['metadata'].get('candidate_name', 'Unknown')

        # Sort by hybrid score
        results = sorted(combined.values(), key=lambda x: x['hybrid_score'], reverse=True)

        # Aggregate by candidate (same candidate may have multiple chunks)
        return self._aggregate_by_candidate(results)

    def _aggregate_by_candidate(self, results: List[Dict]) -> List[Dict]:
        """Aggregate results by candidate name"""
        candidate_map = {}

        for result in results:
            candidate_name = result['candidate_name']
            if candidate_name not in candidate_map:
                candidate_map[candidate_name] = {
                    'candidate_name': candidate_name,
                    'chunks': [],
                    'best_score': 0,
                    'all_skills': set(),
                    'file_path': result['metadata'].get('file_path', ''),
                    'metadata': result['metadata']
                }

            candidate_map[candidate_name]['chunks'].append(result)
            candidate_map[candidate_name]['best_score'] = max(
                candidate_map[candidate_name]['best_score'],
                result['hybrid_score']
            )
            candidate_map[candidate_name]['all_skills'].update(
                result['metadata'].get('skills', '').split(',')
            )

        # Convert to list and sort by best score
        aggregated = list(candidate_map.values())
        for candidate in aggregated:
            # Get the best chunk for this candidate
            best_chunk = max(candidate['chunks'], key=lambda x: x['hybrid_score'])
            candidate['best_chunk_document'] = best_chunk.get('document', '')
            candidate['semantic_score'] = best_chunk.get('semantic_score', 0)
            candidate['keyword_score'] = best_chunk.get('keyword_score', 0)
            candidate['keyword_matches'] = best_chunk.get('keyword_matches', [])
            candidate['similarity_score'] = best_chunk.get('similarity_score', 0)
            candidate['skills'] = list(candidate['all_skills'])

        aggregated.sort(key=lambda x: x['best_score'], reverse=True)
        return aggregated

    def _rank_candidates(self, candidates: List[Dict], job_description: str, required_skills: List[str]) -> List[Dict]:
        """Rank candidates with detailed scoring"""
        ranked = []

        for candidate in candidates:
            # Base score from hybrid search
            base_score = candidate['best_score']

            # Bonus for required skills
            skill_bonus = 0
            if required_skills:
                skills_found = [s for s in required_skills
                               if s.lower() in ' '.join(candidate['skills']).lower()]
                skill_bonus = (len(skills_found) / len(required_skills)) * 15

            # Bonus for experience
            exp_years = float(candidate['metadata'].get('experience_years', 0))
            exp_bonus = min(exp_years * 2, 10)  # Max 10 bonus

            # Final score (0-100 scale)
            final_score = min(base_score + skill_bonus + exp_bonus, 100)

            # Ensure score is at least 0
            final_score = max(final_score, 0)

            ranked.append({
                'candidate_name': candidate['candidate_name'],
                'resume_path': candidate['file_path'],
                'match_score': round(final_score, 2),
                'matched_skills': [s for s in required_skills
                                  if s.lower() in ' '.join(candidate['skills']).lower()],
                'relevant_excerpts': [],
                'reasoning': '',
                'metadata': candidate['metadata'],
                'skills': candidate['skills'],
                'experience_years': exp_years,
                'semantic_score': round(candidate.get('semantic_score', 0), 2),
                'keyword_score': round(candidate.get('keyword_score', 0), 2),
                'hybrid_score': round(candidate.get('best_score', 0), 2)
            })

        # Sort by match score
        ranked.sort(key=lambda x: x['match_score'], reverse=True)
        return ranked

    def _extract_required_skills(self, job_description: str) -> List[str]:
        """Extract required skills from job description"""
        # Common skill keywords to look for
        skill_keywords = [
            'python', 'java', 'javascript', 'typescript', 'c++', 'go', 'rust',
            'machine learning', 'deep learning', 'nlp', 'computer vision',
            'aws', 'azure', 'gcp', 'docker', 'kubernetes', 'linux',
            'sql', 'mongodb', 'postgresql', 'mysql', 'redis',
            'react', 'angular', 'vue', 'node.js', 'express',
            'django', 'flask', 'spring', 'hibernate',
            'git', 'jenkins', 'ci/cd', 'devops', 'agile', 'scrum',
            'data science', 'analytics', 'power bi', 'tableau',
            'pandas', 'numpy', 'scikit-learn', 'tensorflow', 'pytorch'
        ]

        text_lower = job_description.lower()
        found_skills = []

        for skill in skill_keywords:
            if skill in text_lower:
                found_skills.append(skill)

        return found_skills

    def _generate_reasoning(self, match: Dict, job_description: str, required_skills: List[str]) -> str:
        """Generate reasoning for match score"""
        reasons = []

        # Score components
        score = match['match_score']
        reasons.append(f"Overall match score: {score:.1f}/100")

        # Skills match
        if match['matched_skills']:
            reasons.append(f"Matched skills: {', '.join(match['matched_skills'])}")
        else:
            reasons.append("No specific skills matched from job description")

        # Experience
        exp = match.get('experience_years', 0)
        if exp > 0:
            reasons.append(f"Experience: {exp:.1f} years")
        else:
            reasons.append("Experience details not specified")

        # Semantic match quality
        semantic_score = match.get('semantic_score', 0)
        if semantic_score > 70:
            reasons.append(f"Strong semantic similarity: {semantic_score:.1f}%")
        elif semantic_score > 50:
            reasons.append(f"Moderate semantic similarity: {semantic_score:.1f}%")
        else:
            reasons.append("Low semantic similarity")

        # Keyword match
        keyword_score = match.get('keyword_score', 0)
        if keyword_score > 50:
            reasons.append(f"Good keyword match: {keyword_score:.1f}%")

        return " | ".join(reasons)

    def _get_relevant_excerpts(self, resume_path: str, chunk_index: int) -> List[str]:
        """Get relevant excerpts from resume"""
        # For simplicity, return the document content
        try:
            # Get the chunk from collection
            results = self.collection.get(limit=1)
            # In a real implementation, you'd retrieve the specific chunk
            # For now, return a placeholder
            return ["Relevant excerpt from resume"]
        except:
            return ["Unable to retrieve excerpts"]

    def match_with_required_skills_filter(self, job_description: str, must_have_skills: List[str], top_k: int = 10) -> Dict:
        """
        Match with filtering for must-have skills

        Args:
            job_description: Job description text
            must_have_skills: List of skills that are required
            top_k: Number of top matches

        Returns:
            Dictionary with filtered results
        """
        # First get all matches
        results = self.match_job_description(job_description, top_k=top_k*2)

        if 'error' in results:
            return results

        # Filter by must-have skills
        filtered_matches = []
        for match in results['top_matches']:
            matched_skills_lower = [s.lower() for s in match['matched_skills']]
            must_have_lower = [s.lower() for s in must_have_skills]

            # Check if all must-have skills are present
            if all(skill in matched_skills_lower for skill in must_have_lower):
                filtered_matches.append(match)

        # Update results
        results['top_matches'] = filtered_matches[:top_k]
        results['filtered_by_required_skills'] = must_have_skills
        results['candidates_after_filtering'] = len(filtered_matches)

        return results


# Test the JobMatcher
if __name__ == "__main__":
    print("\n" + "="*60)
    print("🚀 Testing JobMatcher")
    print("="*60)

    # Initialize
    matcher = JobMatcher()
    print("✅ JobMatcher initialized successfully!")

    # Sample job description
    job_description = """
    Senior Machine Learning Engineer

    We are looking for an experienced Machine Learning Engineer with:
    - 5+ years of experience in machine learning and data science
    - Strong Python programming skills
    - Experience with deep learning frameworks (TensorFlow, PyTorch)
    """

    print("\n📝 Matching job description...")
    results = matcher.match_job_description(job_description, top_k=5)

    if 'error' in results:
        print(f"❌ Error: {results['error']}")
    else:
        print(f"✅ Found {len(results['top_matches'])} matches")
        for i, match in enumerate(results['top_matches'], 1):
            print(f"\n{i}. {match['candidate_name']}")
            print(f"   Score: {match['match_score']}/100")
            print(f"   Skills: {', '.join(match['matched_skills'])}")

    print("\n" + "="*60)
    print("✅ JobMatcher is ready to use!")
    print("="*60)
