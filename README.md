# Amazon Entity Resolution Pipeline

An end-to-end entity resolution and record linkage pipeline designed to match records across multiple data sources.

## Overview

This project implements a scalable entity resolution workflow combining deterministic preprocessing, candidate generation, fuzzy matching, similarity features, machine learning, candidate re-ranking, and validation.

The pipeline was optimized and executed on an AWS EC2 environment with 8 vCPUs and approximately 64 GB RAM.

## Pipeline

`	ext
Raw Records
     |
     v
Data Normalization
     |
     v
Blocking / Candidate Generation
     |
     v
Similarity & Fuzzy Matching Features
     |
     v
LightGBM Pair Classification
     |
     v
Candidate Re-ranking
     |
     v
Threshold Tuning
     |
     v
Entity Matching
     |
     v
Submission Validation
## Key Techniques

- Record normalization and preprocessing
- Blocking-based candidate generation
- Sparse similarity retrieval
- Fuzzy string matching with RapidFuzz
- Feature engineering for candidate pairs
- LightGBM-based pair classification
- Candidate re-ranking
- Held-out threshold tuning
- Submission and ID validation
- Chunked processing for memory-efficient retrieval
- Parallel processing using available CPU cores

## Validation

The pipeline includes a held-out validation stage for evaluating pair precision, pair recall, blocking recall, and F0.5-based matching performance.

During the EC2 execution, the pipeline reported:

- Held-out pair precision: 0.9926
- Held-out pair recall: 0.9612
- Blocking recall: 0.9768
- Example held-out F0.5 scores: 0.9747 and 0.9824

These are internal held-out validation results from the pipeline and are not the final competition leaderboard score.

## AWS Execution

The full pipeline was executed on an AWS EC2 instance configured with:

- 8 vCPUs
- ~64 GB RAM
- Ubuntu
- Python virtual environment
- JupyterLab

The pipeline was monitored for CPU and memory usage during large-scale candidate retrieval and matching.

## Repository Structure

.
├── ER_final_pipeline_optimized.py
├── ER_Optimized_Run.ipynb
├── requirements.txt
├── README.md
└── .gitignore

## Installation

Create a virtual environment:

`ash
python -m venv venv
