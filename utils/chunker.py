"""
Intelligent Chunking Module
Chunks resumes by preserving natural sections (Education, Experience, Skills, etc.)
"""

import re
from typing import List, Dict, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class IntelligentChunker:
    """Chunk documents intelligently preserving section boundaries"""

    def __init__(self, chunk_size: int = 500, overlap: int = 50):
        """
        Initialize chunker with parameters

        Args:
            chunk_size: Maximum size of each chunk in characters
            overlap: Overlap between chunks to preserve context
        """
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.section_headers = [
            'education', 'experience', 'skills', 'certifications',
            'work experience', 'professional summary', 'objective',
            'projects', 'accomplishments', 'publications',
            'languages', 'interests', 'references', 'contact',
            'technical skills', 'soft skills', 'leadership'
        ]

    def chunk_document(self, text: str, metadata: Optional[Dict] = None) -> List[Dict[str, any]]:
        """
        Chunk document intelligently preserving sections

        Args:
            text: Document text to chunk
            metadata: Optional metadata to attach to chunks

        Returns:
            List of chunk dictionaries with text and metadata
        """
        if not text or len(text.strip()) == 0:
            logger.warning("Empty text provided for chunking")
            return []

        # Try section-based chunking first
        chunks = self._chunk_by_sections(text)

        # If section chunking produces no results or single large chunk, fall back to sliding window
        if not chunks or (len(chunks) == 1 and len(chunks[0]['text']) > self.chunk_size * 2):
            logger.info("Section chunking insufficient, falling back to sliding window")
            chunks = self._chunk_sliding_window(text)

        # Add metadata to each chunk
        for i, chunk in enumerate(chunks):
            chunk['chunk_id'] = i
            chunk['total_chunks'] = len(chunks)
            if metadata:
                chunk['source_metadata'] = metadata

        logger.info(f"Document chunked into {len(chunks)} chunks")
        return chunks

    def _chunk_by_sections(self, text: str) -> List[Dict[str, any]]:
        """Chunk text by detecting section headers"""
        lines = text.split('\n')
        chunks = []
        current_chunk = ""
        current_section = "general"

        # Pattern to match section headers (capitalized words followed by newline or colon)
        section_pattern = r'^([A-Z][A-Za-z\s]+):?\s*$'

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Check if this line is a section header
            header_match = re.match(section_pattern, line)
            if header_match and len(line) < 50:  # Avoid matching long lines
                # Save previous section if it has content
                if current_chunk and len(current_chunk.strip()) > 0:
                    chunks.append({
                        'text': current_chunk.strip(),
                        'section': current_section
                    })

                # Start new section
                current_section = header_match.group(1).lower()
                current_chunk = line + "\n"
            else:
                # If section header detected earlier, add to current section
                if current_chunk or line:
                    current_chunk += line + "\n"

                    # If chunk is getting large, split it
                    if len(current_chunk) > self.chunk_size:
                        chunks.append({
                            'text': current_chunk.strip(),
                            'section': current_section
                        })
                        current_chunk = ""

        # Add the last chunk
        if current_chunk and len(current_chunk.strip()) > 0:
            chunks.append({
                'text': current_chunk.strip(),
                'section': current_section
            })

        # If no sections were detected, treat entire document as one chunk
        if not chunks:
            chunks.append({
                'text': text,
                'section': 'general'
            })

        return chunks

    def _chunk_sliding_window(self, text: str) -> List[Dict[str, any]]:
        """Chunk text using a sliding window approach"""
        chunks = []
        text_length = len(text)
        start = 0

        while start < text_length:
            end = min(start + self.chunk_size, text_length)

            # If not at the end, try to break at a sentence boundary
            if end < text_length:
                # Look for sentence boundary within the last 50 characters
                search_start = max(start + self.chunk_size - 50, start)
                sentence_boundary = self._find_sentence_boundary(text[search_start:end + 50], search_start)
                if sentence_boundary > start:
                    end = sentence_boundary

            chunk_text = text[start:end].strip()
            if chunk_text:
                chunks.append({
                    'text': chunk_text,
                    'section': 'general'
                })

            # Move start position with overlap
            start = end - self.overlap
            if start >= text_length:
                break

        return chunks

    def _find_sentence_boundary(self, text: str, start_offset: int) -> int:
        """Find a sentence boundary (period, question mark, exclamation)"""
        # Look for period followed by space or newline
        patterns = [r'\.\s+', r'\?\s+', r'!\s+', r'\n\s*']

        earliest_boundary = None
        for pattern in patterns:
            matches = list(re.finditer(pattern, text))
            if matches:
                # Find the first match that's not too close to the start
                for match in matches:
                    pos = match.start() + 1  # Include the punctuation
                    if pos > 20:  # Avoid breaking very early
                        earliest_boundary = start_offset + pos
                        break
            if earliest_boundary:
                break

        return earliest_boundary if earliest_boundary else start_offset + len(text)
