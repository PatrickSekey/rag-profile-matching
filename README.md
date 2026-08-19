# RAG-Based Profile Matching System

## Overview
An intelligent resume screening system using RAG for semantic job-resume matching.

## Features
- Multi-Format Support: PDF, TXT, DOCX
- Semantic Search with sentence-transformers
- Hybrid Ranking: Semantic + Keyword
- ChromaDB Vector Database

## Quick Start

```bash
pip install pymupdf python-docx sentence-transformers chromadb
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
Patrick Wah Sekey
- GitHub: https://github.com/PatrickSekey
- Email: patrick.sekey.ps@gmail.com
