# Enterprise Customer Onboarding Pipeline

AI agent-driven workflow that ingests unstructured customer data from AWS S3, parses it using LLMs, and writes normalized records to a legacy CRM REST API with retry logic and failure handling.

---

## Features

- Automated ingestion of unstructured customer files
- LLM-powered data extraction and normalization
- AWS S3 event-driven processing pipeline
- Retry and validation logic for failed API requests
- Dead-letter queue support using AWS SQS
- Legacy CRM REST API integration

---

## Tech Stack

- GPT-4o — LLM parsing of unstructured data
- AWS S3 — Raw file storage and event triggers
- AWS SQS — Dead-letter queue for failed records
- Python — Orchestration, validation, and retry logic

---

## Setup

Install dependencies:

```bash
pip install -r requirements.txt
```

Create environment variables file:

```bash
cp .env.example .env
```

Add your API key inside `.env`:

```env
OPENAI_API_KEY=your_api_key
```

Run the pipeline:

```bash
python src/pipeline.py
```

---
---

## Workflow

1. Customer files uploaded to AWS S3
2. Event trigger starts processing pipeline
3. LLM extracts and structures customer information
4. Validation and normalization applied
5. Data pushed to CRM REST API
6. Failed records retried automatically
7. Persistent failures routed to AWS SQS dead-letter queue

---

## Use Case

Designed for enterprise onboarding workflows where customer information arrives in inconsistent or unstructured formats and must be transformed into standardized CRM-ready records automatically.

```
