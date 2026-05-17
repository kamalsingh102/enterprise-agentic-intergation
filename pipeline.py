"""
Enterprise Customer Onboarding Pipeline
========================================
Ingests unstructured customer data from AWS S3, parses it using an LLM,
and writes normalised records to a legacy CRM REST API with full resilience.

Flow:
  S3 file → fetch → LLM parse → validate → CRM write (w/ retry) → audit
                                                     ↓ on exhaustion
                                                  Dead-letter queue
"""

import json
import logging
import os
import time
from dataclasses import asdict, dataclass
from typing import Optional
from openai import OpenAI
import boto3
import requests

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("onboarding.pipeline")

# ── Config (override via environment variables) ───────────────────────────────
S3_BUCKET    = os.getenv("S3_BUCKET", "acme-onboarding-raw")
CRM_BASE_URL = os.getenv("CRM_BASE_URL", "https://crm.internal/api/v1")
DLQ_URL      = os.getenv("DLQ_URL", "https://sqs.us-east-1.amazonaws.com/123456789/onboarding-dlq")
MAX_RETRIES  = int(os.getenv("MAX_RETRIES", "5"))
BACKOFF_BASE = float(os.getenv("BACKOFF_BASE", "1.0"))   # seconds

# ── AWS / Anthropic clients ───────────────────────────────────────────────────
openai_client = OpenAI()                           # reads ANTHROPIC_API_KEY
s3     = boto3.client("s3")
sqs    = boto3.client("sqs")


# ── Data model ────────────────────────────────────────────────────────────────
@dataclass
class CustomerRecord:
    name:     str
    email:    str
    company:  str
    plan:     str
    phone:    Optional[str] = None
    metadata: Optional[dict] = None


# ── Stage 1: Ingest from S3 ───────────────────────────────────────────────────
def fetch_from_s3(bucket: str, key: str) -> str:
    """Download and decode a raw onboarding file from S3."""
    obj = s3.get_object(Bucket=bucket, Key=key)
    content = obj["Body"].read().decode("utf-8")
    logger.info("Fetched s3://%s/%s (%d chars)", bucket, key, len(content))
    return content


# ── Stage 2: Parse with LLM ───────────────────────────────────────────────────
_SYSTEM_PROMPT = """You are a data extraction agent for a customer onboarding system.

Given raw, unstructured customer data (emails, forms, freeform text, CSV rows —
any format), extract the fields listed below and return ONLY a valid JSON object.
No prose, no markdown code fences, no explanation — just the JSON.

Required fields:
  name     (string) — customer full name
  email    (string) — primary email address
  company  (string) — organisation name
  plan     (string) — product plan (e.g. Free, Pro, Enterprise)

Optional fields:
  phone    (string or null) — phone number if present
  metadata (object or null) — any other key-value pairs found in the data

If a required field is absent or genuinely ambiguous, set its value to null.
Never hallucinate values that are not present in the input."""


def parse_with_llm(raw_content: str) -> CustomerRecord:
    response = openai_client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user",   "content": raw_content}
        ],
        max_tokens=512,
    )

    text = response.choices[0].message.content.strip()

    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"LLM returned non-JSON output: {text!r}") from exc

    return CustomerRecord(
        name=data.get("name"),
        email=data.get("email"),
        company=data.get("company"),
        plan=data.get("plan"),
        phone=data.get("phone"),
        metadata=data.get("metadata"),
    )

# ── Stage 2b: Validate ────────────────────────────────────────────────────────
_REQUIRED_FIELDS = ("name", "email", "company", "plan")
_VALID_PLANS     = {"free", "pro", "enterprise", "starter", "business"}


def validate(record: CustomerRecord) -> list[str]:
    """Return a list of validation error strings (empty list = valid)."""
    errors = []

    for field in _REQUIRED_FIELDS:
        if not getattr(record, field):
            errors.append(f"Missing required field: '{field}'")

    if record.email and "@" not in record.email:
        errors.append(f"Invalid email format: '{record.email}'")

    if record.plan and record.plan.lower() not in _VALID_PLANS:
        errors.append(
            f"Unknown plan '{record.plan}'. Expected one of: {sorted(_VALID_PLANS)}"
        )

    return errors


# ── Stage 3: Write to CRM with exponential backoff ────────────────────────────
def post_to_crm(record: CustomerRecord) -> dict:
    """
    POST a normalised record to the legacy CRM REST API.

    Retry strategy:
      - 429 (rate limit): honour Retry-After header if present, else backoff
      - 5xx (server error): exponential backoff
      - 4xx other than 429: non-retryable, raise immediately
      - Timeout: treat as transient, backoff and retry
    """
    payload = {
        "full_name":     record.name,
        "email":         record.email,
        "organisation":  record.company,
        "product_plan":  record.plan,
        "phone":         record.phone,
        "custom_fields": record.metadata or {},
    }

    last_error: Optional[str] = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.post(
                f"{CRM_BASE_URL}/customers",
                json=payload,
                timeout=10,
                headers={"Content-Type": "application/json"},
            )

            if resp.status_code == 200:
                logger.info("CRM write succeeded on attempt %d", attempt)
                return resp.json()

            if resp.status_code == 429:
                retry_after = float(resp.headers.get("Retry-After", 0))
                wait = max(retry_after, BACKOFF_BASE * (2 ** (attempt - 1)))
                logger.warning(
                    "Rate limited (429). Waiting %.1fs (attempt %d/%d)",
                    wait, attempt, MAX_RETRIES,
                )
                time.sleep(wait)
                last_error = f"HTTP 429 on attempt {attempt}"
                continue

            if resp.status_code >= 500:
                wait = BACKOFF_BASE * (2 ** (attempt - 1))
                logger.warning(
                    "Server error %d. Waiting %.1fs (attempt %d/%d)",
                    resp.status_code, wait, attempt, MAX_RETRIES,
                )
                time.sleep(wait)
                last_error = f"HTTP {resp.status_code} on attempt {attempt}"
                continue

            # Non-retryable 4xx
            raise ValueError(
                f"Non-retryable CRM error {resp.status_code}: {resp.text}"
            )

        except requests.Timeout:
            wait = BACKOFF_BASE * (2 ** (attempt - 1))
            logger.warning(
                "Request timeout. Waiting %.1fs (attempt %d/%d)",
                wait, attempt, MAX_RETRIES,
            )
            time.sleep(wait)
            last_error = f"Timeout on attempt {attempt}"

    raise RuntimeError(
        f"CRM write failed after {MAX_RETRIES} attempts. Last error: {last_error}"
    )


# ── Stage 3b: Dead-letter queue ───────────────────────────────────────────────
def send_to_dlq(s3_key: str, record: Optional[CustomerRecord], error: str) -> None:
    """Push a failed record to SQS for alerting and manual replay."""
    message = {
        "s3_key": s3_key,
        "record": asdict(record) if record else None,
        "error":  error,
        "ts":     time.time(),
    }
    sqs.send_message(QueueUrl=DLQ_URL, MessageBody=json.dumps(message))
    logger.error("Sent to DLQ — key: %s | error: %s", s3_key, error)


# ── Stage 4: Audit log ────────────────────────────────────────────────────────
def audit(s3_key: str, status: str, detail: str) -> None:
    """Emit a structured JSON audit event (pipe to CloudWatch / Datadog / etc.)."""
    logger.info(
        json.dumps({
            "event":  "onboarding_pipeline",
            "s3_key": s3_key,
            "status": status,
            "detail": detail,
            "ts":     time.time(),
        })
    )


# ── Orchestrator ──────────────────────────────────────────────────────────────
def handle_onboarding_event(s3_bucket: str, s3_key: str) -> None:
    """
    Main entry point — invoked by Lambda on S3 ObjectCreated event.

    Full pipeline:
      fetch → LLM parse → validate → CRM write → audit (success or failure)
    """
    record: Optional[CustomerRecord] = None

    try:
        # 1. Ingest
        raw = fetch_from_s3(s3_bucket, s3_key)

        # 2. Parse
        record = parse_with_llm(raw)
        logger.info("Parsed record: %s", asdict(record))

        # 2b. Validate
        errors = validate(record)
        if errors:
            raise ValueError(f"Validation failed: {errors}")

        # 3. Write to CRM (retry/backoff built in)
        crm_response = post_to_crm(record)

        # 4. Audit success
        audit(s3_key, "success", f"CRM id={crm_response.get('id')}")

    except Exception as exc:
        send_to_dlq(s3_key, record, str(exc))
        audit(s3_key, "error", str(exc))
        raise   # re-raise so Lambda marks invocation as failed


# ── Lambda handler shim ───────────────────────────────────────────────────────
def lambda_handler(event: dict, context) -> dict:
    """AWS Lambda entry point triggered by S3 event notification."""
    for record in event.get("Records", []):
        bucket = record["s3"]["bucket"]["name"]
        key    = record["s3"]["object"]["key"]
        handle_onboarding_event(bucket, key)
    return {"statusCode": 200, "body": "OK"}


# ── Local test harness ────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    # Allow passing a local file path for testing without real S3
    if len(sys.argv) > 1:
        local_path = sys.argv[1]
        with open(local_path) as f:
            raw = f.read()

        print("\n── Raw input ──────────────────────────────────────")
        print(raw)

        record = parse_with_llm(raw)
        print("\n── Parsed record ──────────────────────────────────")
        print(json.dumps(asdict(record), indent=2))

        errors = validate(record)
        if errors:
            print("\n── Validation errors ──────────────────────────────")
            for e in errors:
                print(" ✗", e)
        else:
            print("\n✓ Validation passed")
    else:
        # Simulate a real S3 event
        handle_onboarding_event(S3_BUCKET, "customers/2026-05-16/acme_corp.txt")
