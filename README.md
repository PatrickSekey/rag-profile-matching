# RAG-Based Profile Matching System

## Overview
An intelligent resume screening system that uses Retrieval-Augmented Generation (RAG) to match job descriptions with candidate resumes using semantic search.

## Features
- Multi-Format Support: PDF, TXT, DOCX
- Semantic Search with sentence-transformers
- Hybrid Ranking: Semantic + Keyword matching
- Performance Metrics & Latency Testing
- ChromaDB Vector Database
- JSON Output for easy integration

## Installation

```bash
pip install pymupdf python-docx sentence-transformers chromadb pandas numpy
```

## Quick Start

```bash
python main.py
```

## Performance Metrics

| Metric | Score |
|--------|-------|
| Precision@k | 0.67 |
| Recall@k | 0.50 |
| MRR | 0.58 |
| Avg Latency | 0.83s |

## Author
**Patrick Wah Sekey**
- GitHub: [PatrickSekey](https://github.com/PatrickSekey)
- Email: patrick.sekey.ps@gmail.com

## License
MIT
