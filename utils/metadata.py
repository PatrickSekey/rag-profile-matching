"""
Metadata Extraction Module
Extracts key fields from resumes: Name, Skills, Experience Years, Education
"""

import re
from typing import Dict, List, Optional, Tuple
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MetadataExtractor:
    """Extract structured metadata from resume text"""
    
    def __init__(self):
        self.skill_keywords = [
            'python', 'java', 'c++', 'javascript', 'typescript', 'go', 'rust',
            'machine learning', 'deep learning', 'nlp', 'computer vision',
            'aws', 'azure', 'gcp', 'docker', 'kubernetes', 'linux',
            'sql', 'mongodb', 'postgresql', 'mysql', 'redis',
            'react', 'angular', 'vue', 'node.js', 'express',
            'django', 'flask', 'spring', 'hibernate',
            'git', 'jenkins', 'ci/cd', 'devops',
            'agile', 'scrum', 'jira', 'confluence',
            'data science', 'analytics', 'power bi', 'tableau',
            'pandas', 'numpy', 'scikit-learn', 'tensorflow', 'pytorch',
            'llms', 'feature engineering', 'model evaluation',
            'intrusion detection', 'anomaly detection', 'network security',
            'object detection', 'image processing', 'react.js', 'node.js',
            'express.js', 'spring', 'springboot', 'rest apis', 'jwt',
            'mysql', 'mongodb', 'tailwind css', 'streamlit',
            'flutter', 'dart', 'kotlin', 'firebase', 'typescript',
            'postgresql', 'pandas', 'numpy', 'scikit-learn'
        ]
        self.education_keywords = [
            'bachelor', 'master', 'phd', 'mba', 'b.s.', 'm.s.', 'b.tech', 'm.tech',
            'university', 'college', 'institute', 'school', 'academy'
        ]
    
    def extract_metadata(self, text: str) -> Dict:
        """
        Extract structured metadata from resume text
        
        Args:
            text: Resume text content
            
        Returns:
            Dictionary with extracted metadata
        """
        metadata = {
            "name": self._extract_name(text),
            "skills": self._extract_skills(text),
            "experience_years": self._extract_experience_years(text),
            "education": self._extract_education(text),
            "email": self._extract_email(text),
            "phone": self._extract_phone(text)
        }
        
        logger.info(f"Extracted metadata: {metadata['name']} | Skills: {len(metadata['skills'])} | Experience: {metadata['experience_years']} years")
        return metadata
    
    def _extract_name(self, text: str) -> str:
        """Extract name from resume with improved pattern matching"""
        lines = text.split('\n')
        
        # Clean lines and remove empty ones
        clean_lines = [line.strip() for line in lines if line.strip()]
        
        # Check first few lines for name patterns
        for i in range(min(len(clean_lines), 15)):
            line = clean_lines[i]
            
            # Skip lines that are likely not names
            skip_patterns = [
                r'(?i)professional\s+summary',
                r'(?i)education',
                r'(?i)experience',
                r'(?i)skills',
                r'(?i)projects',
                r'(?i)certifications',
                r'(?i)publications',
                r'(?i)research',
                r'(?i)contact',
                r'(?i)email',
                r'(?i)phone',
                r'(?i)linkedin',
                r'(?i)github',
            ]
            
            # Skip if it matches any skip pattern
            skip = False
            for pattern in skip_patterns:
                if re.match(pattern, line, re.IGNORECASE):
                    skip = True
                    break
            if skip:
                continue
            
            # Check for name patterns (more comprehensive)
            name_patterns = [
                r'^([A-Z][a-z]+)\s+([A-Z][a-z]+)\s+([A-Z][a-z]+)$',  # First Middle Last
                r'^([A-Z][a-z]+)\s+([A-Z]\.)\s+([A-Z][a-z]+)$',  # First M. Last
                r'^([A-Z][a-z]+)\s+([A-Z][a-z]+)$',  # First Last
                r'^([A-Z][A-Za-z\s]+)$',  # All caps name (common in resumes)
            ]
            
            for pattern in name_patterns:
                match = re.match(pattern, line)
                if match:
                    return match.group(0).strip()
            
            # Check if line has "|" and starts with what looks like a name
            if '|' in line and len(line.split('|')[0].strip().split()) >= 2:
                potential_name = line.split('|')[0].strip()
                if len(potential_name.split()) >= 2 and re.match(r'^[A-Z][a-z]+\s+[A-Z][a-z]+', potential_name):
                    return potential_name
        
        # If no name found in first lines, check for name with email or phone
        email = self._extract_email(text)
        if email:
            email_index = text.find(email)
            if email_index > 10:
                before_email = text[:email_index].strip()
                lines_before = before_email.split('\n')
                if lines_before:
                    last_line = lines_before[-1].strip()
                    if len(last_line.split()) >= 2 and re.match(r'^[A-Z][a-z]+\s+[A-Z][a-z]+', last_line):
                        return last_line
        
        return "Unknown Candidate"
    
    def _extract_skills(self, text: str) -> List[str]:
        """Extract skills from resume text"""
        text_lower = text.lower()
        found_skills = []
        
        for skill in self.skill_keywords:
            if skill in text_lower:
                found_skills.append(skill.title())
        
        # Remove duplicates while preserving order
        seen = set()
        unique_skills = []
        for skill in found_skills:
            if skill.lower() not in seen:
                seen.add(skill.lower())
                unique_skills.append(skill)
        
        return unique_skills
    
    def _extract_experience_years(self, text: str) -> float:
        """
        Extract total years of experience from resume using multiple methods
        
        Methods:
        1. Look for explicit "X years of experience" patterns
        2. Extract from date ranges like (2021-Present) or (2021-2024)
        3. Calculate total from multiple job entries
        4. Estimate from education graduation year
        """
        text_lower = text.lower()
        current_year = datetime.now().year
        
        # ============================================
        # METHOD 1: Look for explicit "X years of experience"
        # ============================================
        patterns = [
            r'(\d+)\s*\+\s*years?\s+(?:of\s+)?experience',
            r'(\d+)\s+years?\s+(?:of\s+)?experience',
            r'experience\s*:?\s*(\d+)\s*\+\s*years?',
            r'over\s+(\d+)\s+years?\s+(?:of\s+)?experience',
            r'(\d+)\s+years?\s+experience',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text_lower)
            if match:
                years = float(match.group(1))
                if 0 < years < 60:  # Sanity check
                    logger.info(f"Found explicit experience: {years} years")
                    return years
        
        # ============================================
        # METHOD 2: Extract from date ranges
        # ============================================
        total_years = 0.0
        experience_entries = []
        
        # Pattern 1: (YYYY-Present) or (YYYY-YYYY)
        date_range_patterns = [
            r'\((\d{4})\s*[-–]\s*Present\)',      # (2021-Present)
            r'\((\d{4})\s*[-–]\s*(\d{4})\)',       # (2021-2024)
            r'(\d{4})\s*[-–]\s*Present',            # 2021-Present
            r'(\d{4})\s*[-–]\s*(\d{4})',            # 2021-2024
            r'(\d{4})\s*to\s*(?:Present|\d{4})',    # 2021 to Present
            r'(\d{4})\s*-\s*(?:Present|\d{4})',     # 2021 - Present
        ]
        
        for pattern in date_range_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                try:
                    if isinstance(match, tuple):
                        if len(match) == 2:
                            start_year = int(match[0])
                            end_year = int(match[1])
                            years = end_year - start_year
                            if 0 < years < 60:
                                experience_entries.append(years)
                                total_years += years
                        elif len(match) == 1:
                            start_year = int(match[0])
                            # Check if "Present" is in the match context
                            years = current_year - start_year
                            if 0 < years < 60:
                                experience_entries.append(years)
                                total_years += years
                    else:
                        # Single year match
                        start_year = int(match)
                        years = current_year - start_year
                        if 0 < years < 60:
                            experience_entries.append(years)
                            total_years += years
                except (ValueError, TypeError):
                    continue
        
        # If we found entries, log and return the total
        if experience_entries:
            logger.info(f"Found {len(experience_entries)} experience entries: {experience_entries}")
            logger.info(f"Total experience from date ranges: {total_years} years")
            return total_years
        
        # ============================================
        # METHOD 3: Look for job duration patterns
        # ============================================
        # Pattern: "Jan 2020 - Present" or "Jan 2020 - Dec 2024"
        duration_patterns = [
            r'([A-Z][a-z]+)\s+(\d{4})\s*[-–]\s*([A-Z][a-z]+)\s+(\d{4})',
            r'([A-Z][a-z]+)\s+(\d{4})\s*[-–]\s*Present',
            r'(\d{4})\s*[-–]\s*(\d{4})',
        ]
        
        for pattern in duration_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                try:
                    if len(match) == 4 and 'Present' not in match:
                        # Month Year - Month Year
                        start_year = int(match[1])
                        end_year = int(match[3])
                        years = end_year - start_year
                        if 0 < years < 60:
                            total_years += years
                    elif len(match) == 2 and 'Present' in str(match):
                        # Month Year - Present
                        start_year = int(match[1])
                        years = current_year - start_year
                        if 0 < years < 60:
                            total_years += years
                    elif len(match) == 2:
                        # Year - Year
                        start_year = int(match[0])
                        end_year = int(match[1])
                        years = end_year - start_year
                        if 0 < years < 60:
                            total_years += years
                except (ValueError, TypeError):
                    continue
        
        if total_years > 0:
            logger.info(f"Total experience from durations: {total_years} years")
            return total_years
        
        # ============================================
        # METHOD 4: Estimate from education year
        # ============================================
        # Look for graduation years
        edu_patterns = [
            r'(?:B\.?S\.?|M\.?S\.?|B\.?Tech|M\.?Tech|Bachelor|Master|Ph\.?D)\s*[,\s]+(\d{4})',
            r'Graduated\s+(\d{4})',
            r'Class of (\d{4})',
            r'(\d{4})\s*[-–]\s*(\d{4})\s*$',  # Degree years at end of line
        ]
        
        grad_years = []
        for pattern in edu_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                try:
                    if isinstance(match, tuple) and len(match) == 2:
                        # Both start and end years - use end year
                        grad_years.append(int(match[1]))
                    else:
                        grad_years.append(int(match))
                except (ValueError, TypeError):
                    continue
        
        if grad_years:
            latest_grad = max(grad_years)
            if 2000 <= latest_grad <= current_year:
                experience = current_year - latest_grad - 1  # Rough estimate
                if 0 < experience < 50:
                    logger.info(f"Estimated experience from graduation year ({latest_grad}): {experience} years")
                    return experience
        
        # If all methods fail, return 0
        logger.info("No experience detected")
        return 0.0
    
    def _extract_education(self, text: str) -> List[str]:
        """Extract education information from resume"""
        text_lower = text.lower()
        education_items = []
        
        # Find sentences containing education keywords
        sentences = re.split(r'[.!?]+', text)
        for sentence in sentences:
            sentence_lower = sentence.lower()
            if any(keyword in sentence_lower for keyword in self.education_keywords):
                clean_sentence = sentence.strip()
                if len(clean_sentence) > 10 and len(clean_sentence) < 300:
                    education_items.append(clean_sentence.title())
        
        # Remove duplicates
        seen = set()
        unique_education = []
        for item in education_items:
            if item not in seen:
                seen.add(item)
                unique_education.append(item)
        
        return unique_education[:3]
    
    def _extract_email(self, text: str) -> Optional[str]:
        """Extract email address from text"""
        email_pattern = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
        match = re.search(email_pattern, text)
        return match.group(0) if match else None
    
    def _extract_phone(self, text: str) -> Optional[str]:
        """Extract phone number from text"""
        phone_patterns = [
            r'\+?\d{1,3}[\s-]?\(?\d{3}\)?[\s-]?\d{3}[\s-]?\d{4}',  # International format
            r'\(\d{3}\)\s*\d{3}-\d{4}',  # (123) 456-7890
            r'\d{3}-\d{3}-\d{4}',  # 123-456-7890
            r'\d{3}\.\d{3}\.\d{4}',  # 123.456.7890
            r'\d{10}'  # 1234567890
        ]
        
        for pattern in phone_patterns:
            match = re.search(pattern, text)
            if match:
                return match.group(0)
        
        return None
