"""
JobLens — AI-Powered Job Notification Analyzer & Fact-Checker
=============================================================
Flask backend with Tavily (free web search) + Groq (free, no-credit-card
LLM tier) integration — llama-3.3-70b-versatile synthesizes Tavily's
search results for fact-checking + Google Dorking, openai/gpt-oss-120b
extracts structured JSON — plus .ics calendar file generation.

Usage:
    set GROQ_API_KEY=your_key_here
    set TAVILY_API_KEY=your_key_here
    python app.py
"""

import os
import sys
import time
import uuid
import logging
import socket
import ipaddress
from datetime import datetime, timedelta
from typing import Optional, List
from io import BytesIO
from urllib.parse import urlparse, urljoin
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from flask import (
    Flask, request, jsonify, render_template,
    Response, abort, redirect, url_for
)
from pydantic import BaseModel, Field, ValidationError
import openai
from openai import OpenAI
import pdfplumber
import requests
from bs4 import BeautifulSoup

import db

# ─── Logging ────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
)
logger = logging.getLogger("joblens")


# ─────────────────────────────────────────────────────────────────
#  Pydantic Models — strict JSON schema for Groq output
# ─────────────────────────────────────────────────────────────────

class JobDates(BaseModel):
    """Important dates related to the job notification."""
    start_date: Optional[str] = Field(
        None, description="Application start date (DD-MM-YYYY or descriptive)"
    )
    last_date: Optional[str] = Field(
        None, description="Last date to submit the application"
    )
    exam_date: Optional[str] = Field(
        None, description="Expected examination date"
    )
    age_cutoff_date: Optional[str] = Field(
        None, description="Reference date for age calculation"
    )


class VacancyBreakdown(BaseModel):
    """A single category's vacancy count."""
    category: str = Field(
        ..., description="Vacancy category, e.g. General, OBC, SC, ST, EWS, PwD"
    )
    count: str = Field(
        ..., description="Number of vacancies for this category (as given, may include notes)"
    )


class VacancyInfo(BaseModel):
    """Total and category-wise vacancy counts."""
    total: Optional[str] = Field(
        None, description="Total number of vacancies across all categories"
    )
    breakdown: Optional[List[VacancyBreakdown]] = Field(
        None, description="Category-wise vacancy breakdown"
    )


class FeeBreakdown(BaseModel):
    """A single category's application fee."""
    category: str = Field(
        ...,
        description="Applicant category, e.g. General, OBC, SC/ST, Female, Ex-Servicemen",
    )
    amount: str = Field(
        ..., description="Fee amount for this category, as given (include currency)"
    )


class FeeInfo(BaseModel):
    """Category-wise application fee details."""
    breakdown: Optional[List[FeeBreakdown]] = Field(
        None, description="Category-wise application fee breakdown"
    )
    payment_mode: Optional[str] = Field(
        None, description="Accepted payment modes, e.g. online/net banking/UPI"
    )
    notes: Optional[str] = Field(
        None, description="Fee exemptions, refund policy, or other notes"
    )


class SelectionStage(BaseModel):
    """A single stage of the selection process."""
    stage: str = Field(
        ..., description="Name of the selection stage, e.g. 'Tier I — Prelims'"
    )
    description: Optional[str] = Field(
        None, description="Brief description of what this stage involves"
    )


class ApplyStep(BaseModel):
    """A single step in the how-to-apply guide."""
    step_number: int = Field(
        ..., description="Sequential step number, starting at 1"
    )
    instruction: str = Field(
        ..., description="What the applicant should do at this step"
    )


class JobDetails(BaseModel):
    """
    Comprehensive structured output for a government or private
    job notification, extracted and fact-checked by Groq.
    """
    job_title: str = Field(
        ..., description="Full title of the recruitment / job position"
    )
    official_pdf_link: Optional[str] = Field(
        None,
        description="Direct URL to the official notification PDF file",
    )
    official_apply_portal: Optional[str] = Field(
        None,
        description="URL of the official online application portal",
    )
    dates: Optional[JobDates] = Field(
        None, description="Key dates associated with the notification"
    )
    pay_scale: Optional[str] = Field(
        None, description="Salary / pay band / CTC details"
    )
    vacancies: Optional[VacancyInfo] = Field(
        None, description="Total and category-wise vacancy count"
    )
    eligibility: Optional[str] = Field(
        None,
        description="Educational qualifications, age limits, and other criteria",
    )
    selection_process: Optional[List[SelectionStage]] = Field(
        None,
        description="Ordered stages of the selection process (written, interview, etc.)",
    )
    fees: Optional[FeeInfo] = Field(
        None, description="Application fee details by category"
    )
    documents_checklist: Optional[List[str]] = Field(
        None, description="List of documents required at the time of application or verification"
    )
    how_to_apply_guide: Optional[List[ApplyStep]] = Field(
        None, description="Sequential, numbered step-by-step guide on how to apply"
    )
    study_resources: Optional[List[str]] = Field(
        None,
        description="Recommended books, syllabus links, or preparation resources",
    )
    helpdesk_info: Optional[str] = Field(
        None,
        description="Helpdesk contact information (phone, email, address)",
    )
    fact_check_summary: Optional[str] = Field(
        None,
        description="Short plain-language summary of the fact-check: what was "
        "verified against official sources and how well it matched.",
    )
    discrepancies: Optional[List[str]] = Field(
        None,
        description="Concrete mismatches or unverifiable claims found between the "
        "provided content and official sources. Empty/null if everything checked out.",
    )
    confidence_level: Optional[str] = Field(
        None,
        description="Overall confidence in the extracted details: 'High', 'Medium', or 'Low'.",
    )


def _to_strict_json_schema(model: type[BaseModel]) -> dict:
    """
    Convert a Pydantic model to Groq/OpenAI-compatible "strict" JSON
    Schema: every object requires additionalProperties=false and lists
    ALL of its properties (including optional ones) in "required".
    Optional fields stay nullable via Pydantic's existing anyOf-null
    branch — strict mode just insists the key always be present.
    """
    schema = model.model_json_schema()

    def _tighten(node):
        if not isinstance(node, dict):
            return
        if node.get("type") == "object" and "properties" in node:
            node["additionalProperties"] = False
            node["required"] = list(node["properties"].keys())
            for prop in node["properties"].values():
                _tighten(prop)
        for key in ("anyOf", "oneOf", "allOf"):
            for sub in node.get(key, []):
                _tighten(sub)
        if "items" in node:
            _tighten(node["items"])

    _tighten(schema)
    for definition in schema.get("$defs", {}).values():
        _tighten(definition)

    return schema


_JOB_DETAILS_SCHEMA = _to_strict_json_schema(JobDetails)


# ─────────────────────────────────────────────────────────────────
#  Flask Application
# ─────────────────────────────────────────────────────────────────

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16 MB upload limit
app.secret_key = os.environ.get("FLASK_SECRET", os.urandom(32).hex())

# Ensure the SQLite schema exists before any request is served.
db.init_db()


# ─────────────────────────────────────────────────────────────────
#  Jinja filter — safe freeform-text → HTML (paragraphs, bullets, bold)
# ─────────────────────────────────────────────────────────────────

import re
from markupsafe import Markup, escape


@app.template_filter("fmt")
def format_text_filter(raw: Optional[str]) -> Markup:
    """
    Render freeform AI text as safe HTML: paragraphs, bullet/numbered lists,
    and **bold**. Everything is HTML-escaped first; only a small whitelist of
    tags is ever emitted. Mirrors the client-side formatter so server-rendered
    pages match the look.
    """
    if not raw:
        return Markup("")

    bullet_re = re.compile(r"^\s*[-•*]\s+(.*)$")
    number_re = re.compile(r"^\s*\d+[.)]\s+(.*)$")

    blocks: list[dict] = []
    current: Optional[dict] = None
    for line in raw.replace("\r\n", "\n").split("\n"):
        line = line.strip()
        if not line:
            current = None
            continue
        b, n = bullet_re.match(line), number_re.match(line)
        if b:
            if not current or current["type"] != "ul":
                current = {"type": "ul", "items": []}
                blocks.append(current)
            current["items"].append(b.group(1))
        elif n:
            if not current or current["type"] != "ol":
                current = {"type": "ol", "items": []}
                blocks.append(current)
            current["items"].append(n.group(1))
        else:
            if not current or current["type"] != "p":
                current = {"type": "p", "items": []}
                blocks.append(current)
            current["items"].append(line)

    def inline(text: str) -> str:
        safe = str(escape(text))
        return re.sub(r"\*\*(.+?)\*\*", r'<strong class="font-semibold text-slate-900">\1</strong>', safe)

    out: list[str] = []
    for block in blocks:
        if block["type"] == "ul":
            lis = "".join(f'<li>{inline(i)}</li>' for i in block["items"])
            out.append(f'<ul class="list-disc pl-5 space-y-1 my-2 marker:text-accent">{lis}</ul>')
        elif block["type"] == "ol":
            lis = "".join(f'<li>{inline(i)}</li>' for i in block["items"])
            out.append(f'<ol class="list-decimal pl-5 space-y-1 my-2 marker:text-accent marker:font-semibold">{lis}</ol>')
        else:
            out.append(f'<p class="mb-2 last:mb-0">{"<br>".join(inline(i) for i in block["items"])}</p>')
    return Markup("".join(out))


@app.template_global("status_of")
def status_of(last_date_str: Optional[str]) -> dict:
    """
    Derive an application status from the last date. Returns a dict with
    ``key`` (open/soon/today/closed/unknown), ``label``, ``days`` (int or None),
    and an ISO ``iso`` date the frontend can use for a live countdown.
    """
    dt = _parse_date(last_date_str) if last_date_str else None
    if dt is None:
        return {"key": "unknown", "label": "Date unknown", "days": None, "iso": None}
    days = (dt.date() - datetime.now().date()).days
    iso = dt.strftime("%Y-%m-%d")
    if days < 0:
        return {"key": "closed", "label": "Closed", "days": days, "iso": iso}
    if days == 0:
        return {"key": "today", "label": "Closes today", "days": 0, "iso": iso}
    if days <= 7:
        return {"key": "soon", "label": f"{days}d left", "days": days, "iso": iso}
    return {"key": "open", "label": f"{days}d left", "days": days, "iso": iso}

ALLOWED_EXTENSIONS = {"pdf"}
GROQ_BASE_URL = "https://api.groq.com/openai/v1"
# groq/compound's own free-tier reliability is poor: its internal tool
# orchestration secretly shares openai/gpt-oss-120b's 8,000 TPM bucket,
# which a handful of calls exhausts (confirmed via response headers) and
# then it rejects unrelated small requests with a misleading 413. So web
# search is done ourselves via Tavily (predictable, generous free tier),
# and a plain (non-agentic) model synthesizes the results.
GROQ_ANALYSIS_MODEL = os.environ.get("GROQ_ANALYSIS_MODEL", "llama-3.3-70b-versatile")
GROQ_MODEL = os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b")
GROQ_MAX_RETRIES = 3
GROQ_RETRY_BASE_DELAY = 2  # seconds; doubles each retry
GROQ_RETRYABLE_CODES = {408, 429, 500, 502, 503, 504}
# Keep prompts comfortably under the free-tier per-request token budget.
GROQ_MAX_CONTENT_CHARS = 6_000

TAVILY_URL = "https://api.tavily.com/search"
TAVILY_MAX_RESULTS = 8


# ─────────────────────────────────────────────────────────────────
#  Helper — file validation
# ─────────────────────────────────────────────────────────────────

def _allowed_file(filename: str) -> bool:
    """Return True if the filename has an allowed extension."""
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


# ─────────────────────────────────────────────────────────────────
#  Helper — SSRF guard for user-supplied URLs
# ─────────────────────────────────────────────────────────────────

def _is_safe_url(url: str) -> bool:
    """
    Return True only for http(s) URLs whose hostname resolves solely to
    public, routable addresses. Blocks loopback, private, link-local,
    and other internal ranges to guard against SSRF (e.g. a pasted URL
    pointing at localhost or a cloud metadata endpoint).
    """
    try:
        parsed = urlparse(url)
        if parsed.scheme not in ("http", "https") or not parsed.hostname:
            return False
        addr_infos = socket.getaddrinfo(parsed.hostname, None)
    except (socket.gaierror, UnicodeError, ValueError):
        return False

    if not addr_infos:
        return False

    for info in addr_infos:
        ip = ipaddress.ip_address(info[4][0])
        if (
            ip.is_private
            or ip.is_loopback
            or ip.is_link_local
            or ip.is_reserved
            or ip.is_multicast
            or ip.is_unspecified
        ):
            return False
    return True


# ─────────────────────────────────────────────────────────────────
#  Helper — PDF text extraction
# ─────────────────────────────────────────────────────────────────

def extract_pdf_text(file_stream) -> str:
    """
    Extract all readable text from a PDF using pdfplumber.
    Handles both file-path and file-like objects.
    """
    text_parts: list[str] = []
    with pdfplumber.open(file_stream) as pdf:
        for i, page in enumerate(pdf.pages):
            page_text = page.extract_text()
            if page_text:
                text_parts.append(f"--- Page {i + 1} ---\n{page_text}")
    return "\n\n".join(text_parts)


# ─────────────────────────────────────────────────────────────────
#  Helper — URL content fetcher
# ─────────────────────────────────────────────────────────────────

def fetch_url_content(url: str) -> str:
    """
    Fetch readable text from a URL.  Automatically detects whether the
    response is a PDF (by Content-Type or URL extension) and routes
    to the PDF extractor if needed.
    """
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/126.0.0.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    }

    # Follow redirects manually, re-validating each hop, so a redirect
    # can't bounce a validated public URL to an internal address.
    next_url = url
    for _ in range(5):
        if not _is_safe_url(next_url):
            raise ValueError("Redirected to a URL that cannot be fetched.")
        resp = requests.get(next_url, timeout=25, headers=headers, allow_redirects=False)
        if resp.is_redirect and resp.headers.get("Location"):
            next_url = urljoin(next_url, resp.headers["Location"])
            continue
        break
    else:
        raise ValueError("Too many redirects while fetching the URL.")

    resp.raise_for_status()

    content_type = resp.headers.get("Content-Type", "").lower()

    # PDF response — extract text
    if "application/pdf" in content_type or url.lower().endswith(".pdf"):
        logger.info("URL returned PDF content; extracting text…")
        return extract_pdf_text(BytesIO(resp.content))

    # HTML response — strip boilerplate
    soup = BeautifulSoup(resp.text, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header", "aside", "noscript"]):
        tag.decompose()
    text = soup.get_text(separator="\n", strip=True)

    # Truncate to stay within the model's context limits
    max_chars = 20_000
    if len(text) > max_chars:
        logger.info("Truncating HTML content from %d to %d chars", len(text), max_chars)
        text = text[:max_chars]

    return text


# ─────────────────────────────────────────────────────────────────
#  Helper — Groq analysis (two-step pipeline)
# ─────────────────────────────────────────────────────────────────

def _is_quota_error(e: "openai.APIStatusError") -> bool:
    """
    True for an 'insufficient_quota' condition — a billing/plan problem,
    not a transient rate limit. Providers report both under the same
    HTTP 429 status, so the error body's inner code is the only way to
    tell them apart. Retrying quota errors would never succeed.
    """
    body = getattr(e, "body", None)
    error = body.get("error", body) if isinstance(body, dict) else {}
    return isinstance(error, dict) and error.get("code") == "insufficient_quota"


def _call_with_retry(fn):
    """
    Call fn() with exponential-backoff retry for transient errors
    (rate limits, server overload/5xx). This keeps a single transient
    blip from failing the whole analysis.
    """
    last_error = None
    for attempt in range(1, GROQ_MAX_RETRIES + 1):
        try:
            return fn()
        except openai.APIStatusError as e:
            last_error = e
            if (
                _is_quota_error(e)
                or e.status_code not in GROQ_RETRYABLE_CODES
                or attempt == GROQ_MAX_RETRIES
            ):
                raise
            delay = GROQ_RETRY_BASE_DELAY * (2 ** (attempt - 1))
            logger.warning(
                "Groq request failed (%s), retrying in %ds… (attempt %d/%d)",
                e.status_code, delay, attempt, GROQ_MAX_RETRIES,
            )
            time.sleep(delay)
    raise last_error


def _derive_search_query(content: str) -> str:
    """Heuristic search query: the first meaningful line of the notification."""
    return " ".join(content.strip().split())[:150]


def _tavily_search(query: str) -> str:
    """
    Run one Tavily web search and return a compact text block (AI answer
    + top results with URLs) for the analysis prompt. Best-effort: on
    any failure, log and return a placeholder so a Tavily hiccup can't
    take down the whole analysis.
    """
    api_key = os.environ.get("TAVILY_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "TAVILY_API_KEY environment variable is not set. "
            "Please set it before running the application."
        )

    try:
        resp = requests.post(
            TAVILY_URL,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            },
            json={
                "query": query,
                "search_depth": "advanced",
                "max_results": TAVILY_MAX_RESULTS,
                "include_answer": "advanced",
            },
            timeout=20,
        )
        resp.raise_for_status()
        data = resp.json()
    except requests.RequestException as e:
        logger.warning("Tavily search failed (%s); continuing without web search.", e)
        return "(Web search unavailable for this analysis.)"

    parts = []
    if data.get("answer"):
        parts.append(f"AI search summary: {data['answer']}")
    for r in data.get("results", []):
        snippet = (r.get("content") or "")[:400]
        parts.append(f"- {r.get('title', '')} ({r.get('url', '')}): {snippet}")

    return "\n".join(parts) if parts else "(No web search results found.)"


def analyze_with_groq(content: str, source_url: str = "") -> dict:
    """
    Two-step pipeline:
      1. Tavily web search (fact-check + Google Dorking for the official
         PDF) synthesized by llama-3.3-70b-versatile on Groq
      2. openai/gpt-oss-120b on Groq — structured extraction into the
         JobDetails schema via strict JSON Schema mode
    """
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "GROQ_API_KEY environment variable is not set. "
            "Please set it before running the application."
        )

    client = OpenAI(api_key=api_key, base_url=GROQ_BASE_URL)

    if len(content) > GROQ_MAX_CONTENT_CHARS:
        logger.info(
            "Truncating notification content from %d to %d chars for Groq's request-size limit",
            len(content), GROQ_MAX_CONTENT_CHARS,
        )
        content = content[:GROQ_MAX_CONTENT_CHARS]

    logger.info("Step 1a — Searching the web via Tavily…")
    search_query = _derive_search_query(content) + " official notification PDF recruitment"
    search_results_text = _tavily_search(search_query)

    # ── Step 1b: Analysis, grounded in the Tavily search results ────
    search_prompt = f"""You are an expert government and private job notification analyst.

TASK — Analyze the job notification content below, using the SEARCH RESULTS
(already gathered from a live web search) to:

1. **Fact-Check** every claim: dates, vacancy counts, pay scales, eligibility criteria.
   Flag any discrepancies you find between the provided content and official sources.

2. **Locate the official PDF** — the search results may include a direct .gov.in/.nic.in
   .pdf URL for the official notification. Return it if present.

3. **Official Application Portal** — find the exact URL where candidates apply online.

4. **Supplement** any missing information found in the search results.

{"Source URL: " + source_url if source_url else "Source: Uploaded PDF document"}

=== NOTIFICATION CONTENT (START) ===
{content}
=== NOTIFICATION CONTENT (END) ===

=== SEARCH RESULTS (START) ===
{search_results_text}
=== SEARCH RESULTS (END) ===

Provide a thorough, well-organized analysis covering EVERY one of these areas:
• Job title and recruiting organization
• Official PDF link (direct .pdf URL)
• Official application portal URL
• Important dates: application start, last date, exam date, age cutoff date
• Pay scale / salary / CTC
• Vacancies (total and category-wise breakdown)
• Eligibility (educational qualifications, age limits, relaxations)
• Selection process (all stages)
• Application fees (by category: General, OBC, SC/ST, etc.)
• Documents checklist (for application and/or verification)
• Step-by-step how-to-apply guide
• Study resources (syllabus, recommended books, past papers links)
• Helpdesk / contact information (phone, email, address)
• Fact-check results and any discrepancies found"""

    logger.info("Step 1b — Sending search-grounded analysis request to Groq…")
    search_response = _call_with_retry(lambda: client.chat.completions.create(
        model=GROQ_ANALYSIS_MODEL,
        messages=[{"role": "user", "content": search_prompt}],
    ))
    analysis_text = search_response.choices[0].message.content
    if not analysis_text:
        raise ValueError(
            "Groq returned an empty response during search-grounded analysis. "
            "Please try again."
        )
    logger.info("Step 1 complete (%d chars returned).", len(analysis_text))

    # ── Step 2: Structured extraction ───────────────────────────
    structure_prompt = f"""You are a data extraction specialist. Based on the verified analysis
below, extract ALL job notification details into the specified JSON schema.

Rules:
- Be precise and thorough. Include every piece of information available.
- For date fields, prefer the format "DD-MM-YYYY". Use a descriptive string if exact date is unknown.
- For list fields (documents_checklist, study_resources), return arrays of strings.
- If a field truly has no data, use null.
- For eligibility, include both educational qualifications AND age limits, as a single freeform text block.
- For vacancies, set "total" to the overall vacancy count and "breakdown" to a list of {{category, count}} entries for each category (General, OBC, SC, ST, EWS, PwD, etc.). Omit "breakdown" if no category-wise split is available.
- For fees, set "breakdown" to a list of {{category, amount}} entries per applicant category, "payment_mode" to the accepted payment methods, and "notes" to any exemptions or refund policy. Omit fields with no data.
- For selection_process, return an ordered list of stages, each with a short "stage" name (e.g. "Tier I — Prelims") and an optional "description" of what it involves.
- For how_to_apply_guide, return an ordered list of steps, each with a sequential "step_number" starting at 1 and an "instruction" describing what the applicant should do.
- For official_pdf_link, only include a direct .pdf URL, not a general page link.
- For fact_check_summary, write 1-2 sentences on what the analysis verified against official sources and how well it matched.
- For discrepancies, list each concrete mismatch or unverifiable claim as a short string (e.g. "Vacancy count differs: notice says 5000, official portal shows 4800"). Use null or an empty list if nothing conflicted.
- For confidence_level, output exactly "High", "Medium", or "Low" based on how strongly the details were corroborated by official sources.

=== VERIFIED ANALYSIS (START) ===
{analysis_text}
=== VERIFIED ANALYSIS (END) ==="""

    logger.info("Step 2 — Sending structured extraction request to Groq…")
    structured_response = _call_with_retry(lambda: client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "user", "content": structure_prompt}],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "JobDetails",
                "strict": True,
                "schema": _JOB_DETAILS_SCHEMA,
            },
        },
    ))

    raw = structured_response.choices[0].message.content
    if not raw:
        raise ValueError("Groq returned an empty response during structured extraction.")

    parsed = JobDetails.model_validate_json(raw)
    result = parsed.model_dump()
    # Carry the Step-1 verified analysis so it can ground later chat /
    # eligibility checks without re-fetching. The caller strips this key
    # before it is stored inside data_json.
    result["_analysis_text"] = analysis_text
    logger.info("Step 2 complete — structured data extracted.")
    return result


# ─────────────────────────────────────────────────────────────────
#  Helper — ICS calendar file generator
# ─────────────────────────────────────────────────────────────────

def _parse_date(date_str: str) -> Optional[datetime]:
    """Attempt to parse a date string in multiple common formats."""
    formats = [
        "%d-%m-%Y", "%d/%m/%Y", "%Y-%m-%d",
        "%d %B %Y", "%d %b %Y",
        "%B %d, %Y", "%b %d, %Y",
        "%d.%m.%Y", "%Y/%m/%d",
    ]
    for fmt in formats:
        try:
            return datetime.strptime(date_str.strip(), fmt)
        except (ValueError, AttributeError):
            continue
    return None


def generate_ics(summary: str, date_str: str, description: str = "") -> str:
    """
    Generate a valid RFC 5545 .ics calendar event with two reminders:
    one at 7 days and one at 2 days before the event.
    """
    dt = _parse_date(date_str)
    if dt is None:
        # If parsing fails, default to 30 days from now
        dt = datetime.now() + timedelta(days=30)
        logger.warning("Could not parse date '%s'; defaulting to %s", date_str, dt.date())

    uid = str(uuid.uuid4())
    now_stamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    event_date = dt.strftime("%Y%m%d")

    # Escape special characters in description for ICS
    safe_desc = (
        description
        .replace("\\", "\\\\")
        .replace("\n", "\\n")
        .replace(",", "\\,")
        .replace(";", "\\;")
    )

    ics = (
        "BEGIN:VCALENDAR\r\n"
        "VERSION:2.0\r\n"
        "PRODID:-//JobLens//AI Job Notification Analyzer//EN\r\n"
        "CALSCALE:GREGORIAN\r\n"
        "METHOD:PUBLISH\r\n"
        "BEGIN:VEVENT\r\n"
        f"UID:{uid}\r\n"
        f"DTSTAMP:{now_stamp}\r\n"
        f"DTSTART;VALUE=DATE:{event_date}\r\n"
        f"DTEND;VALUE=DATE:{event_date}\r\n"
        f"SUMMARY:{summary}\r\n"
        f"DESCRIPTION:{safe_desc}\r\n"
        "STATUS:CONFIRMED\r\n"
        "BEGIN:VALARM\r\n"
        "TRIGGER:-P2D\r\n"
        "ACTION:DISPLAY\r\n"
        f"DESCRIPTION:2 days left — {summary}\r\n"
        "END:VALARM\r\n"
        "BEGIN:VALARM\r\n"
        "TRIGGER:-P7D\r\n"
        "ACTION:DISPLAY\r\n"
        f"DESCRIPTION:1 week left — {summary}\r\n"
        "END:VALARM\r\n"
        "END:VEVENT\r\n"
        "END:VCALENDAR\r\n"
    )
    return ics


# ─────────────────────────────────────────────────────────────────
#  Helper — "Am I eligible?" and grounded Q&A (reuse Groq, no Tavily)
# ─────────────────────────────────────────────────────────────────

def _groq_client() -> OpenAI:
    """Shared Groq (OpenAI-compatible) client."""
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise EnvironmentError("GROQ_API_KEY environment variable is not set.")
    return OpenAI(api_key=api_key, base_url=GROQ_BASE_URL)


class EligibilityVerdict(BaseModel):
    """Structured result of comparing an applicant profile to a notification."""
    verdict: str = Field(..., description="'eligible', 'not_eligible', or 'unclear'")
    computed_age: Optional[str] = Field(
        None, description="Applicant age at the age-cutoff date, if computable"
    )
    reasons: List[str] = Field(
        default_factory=list, description="Short points supporting the verdict"
    )
    blockers: List[str] = Field(
        default_factory=list, description="Criteria the applicant fails or that are unclear"
    )


_ELIGIBILITY_SCHEMA = _to_strict_json_schema(EligibilityVerdict)


def _compute_age(dob_str: str, cutoff_str: str = "") -> Optional[str]:
    """Age (in years) at the cutoff date, or today if no cutoff. Best-effort."""
    dob = _parse_date(dob_str)
    if dob is None:
        return None
    ref = _parse_date(cutoff_str) if cutoff_str else None
    ref = ref or datetime.now()
    years = ref.year - dob.year - ((ref.month, ref.day) < (dob.month, dob.day))
    return str(years)


def check_eligibility(profile: dict, job: dict) -> dict:
    """Ask Groq whether the applicant meets the notification's criteria."""
    client = _groq_client()
    eligibility_text = job.get("eligibility") or "(No eligibility text extracted.)"
    dates = job.get("dates") or {}
    cutoff = dates.get("age_cutoff_date") or ""
    computed_age = _compute_age(profile.get("dob", ""), cutoff)

    prompt = f"""You are an eligibility adjudicator for Indian government/private job
recruitment. Decide whether the applicant meets the notification's criteria.

APPLICANT PROFILE:
- Date of birth: {profile.get('dob') or 'unknown'}
- Computed age at cutoff{f" ({cutoff})" if cutoff else ""}: {computed_age or 'unknown'}
- Category: {profile.get('category') or 'unknown'}
- Gender: {profile.get('gender') or 'unknown'}
- Highest qualification: {profile.get('qualification') or 'unknown'}

NOTIFICATION ELIGIBILITY CRITERIA:
{eligibility_text}

Rules:
- Consider age limits (apply category-based relaxation if the criteria mention it),
  educational qualification, and any category/gender conditions.
- verdict = "eligible" only if the applicant clearly meets every stated criterion;
  "not_eligible" if they clearly fail at least one; "unclear" if key info is missing.
- reasons: short bullet points explaining the decision.
- blockers: criteria the applicant fails OR that couldn't be determined.
- computed_age: echo the computed age if provided, else null."""

    resp = _call_with_retry(lambda: client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "user", "content": prompt}],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "EligibilityVerdict",
                "strict": True,
                "schema": _ELIGIBILITY_SCHEMA,
            },
        },
    ))
    raw = resp.choices[0].message.content
    if not raw:
        raise ValueError("Groq returned an empty eligibility response.")
    result = EligibilityVerdict.model_validate_json(raw).model_dump()
    if not result.get("computed_age") and computed_age:
        result["computed_age"] = computed_age
    return result


def answer_question(job: dict, analysis_text: str, question: str) -> str:
    """Answer a question grounded ONLY in this notification's stored data."""
    client = _groq_client()
    import json as _json
    context = analysis_text or ""
    if not context.strip():
        context = _json.dumps(job, ensure_ascii=False)
    if len(context) > GROQ_MAX_CONTENT_CHARS:
        context = context[:GROQ_MAX_CONTENT_CHARS]

    prompt = f"""You are JobLens, answering a candidate's question about ONE specific
job notification. Answer ONLY from the verified information below. If the answer
isn't present, say so plainly and suggest checking the official notification —
do not invent details.

Keep the answer concise and specific. Use short bullets when listing multiple items.

=== VERIFIED NOTIFICATION INFO (START) ===
{context}
=== VERIFIED NOTIFICATION INFO (END) ===

CANDIDATE QUESTION: {question}"""

    resp = _call_with_retry(lambda: client.chat.completions.create(
        model=GROQ_ANALYSIS_MODEL,
        messages=[{"role": "user", "content": prompt}],
    ))
    answer = resp.choices[0].message.content
    return answer or "Sorry, I couldn't generate an answer. Please try again."


# ─────────────────────────────────────────────────────────────────
#  Routes
# ─────────────────────────────────────────────────────────────────

@app.errorhandler(413)
def handle_file_too_large(_error):
    """Return JSON (not Flask's default HTML page) so the frontend can parse it."""
    return jsonify({"error": "File too large. Maximum upload size is 16 MB."}), 413


def _vacancy_int(notif: dict) -> int:
    """Best-effort integer from a notification's total-vacancy string."""
    vac = (notif.get("data") or {}).get("vacancies") or {}
    total = vac.get("total")
    if not total:
        return 0
    digits = re.sub(r"[^\d]", "", str(total))
    return int(digits) if digits else 0


@app.route("/")
def index():
    """Home dashboard — aggregate view of all tracked notifications."""
    notifications = db.list_notifications()
    profile = db.get_profile()

    closing_soon = 0
    total_vacancies = 0
    eligible = 0
    for n in notifications:
        st = status_of(n.get("last_date"))
        if st["key"] in ("soon", "today"):
            closing_soon += 1
        total_vacancies += _vacancy_int(n)
        elig = n.get("eligibility") or {}
        if elig.get("verdict") == "eligible":
            eligible += 1

    kpis = {
        "tracked": len(notifications),
        "closing_soon": closing_soon,
        "eligible": eligible,
        "total_vacancies": total_vacancies,
    }
    return render_template(
        "home.html", notifications=notifications, profile=profile, kpis=kpis
    )


@app.route("/notification/<notif_id>")
def notification_detail(notif_id):
    """Redesigned detail view for a single saved notification."""
    notif = db.get_notification(notif_id)
    if not notif:
        abort(404)
    profile = db.get_profile()
    return render_template("detail.html", notif=notif, profile=profile)


@app.route("/analyze", methods=["POST"])
def analyze():
    """
    Analyze a job notification from a URL or uploaded PDF.
    Returns structured JSON with fact-checked job details.

    Accepts:
        - form field "url": a notification page URL
        - form file  "pdf": a .pdf file upload
    """
    try:
        url = request.form.get("url", "").strip()
        pdf_file = request.files.get("pdf")
        content = ""
        source_url = ""

        # ── PDF upload path ──
        if pdf_file and pdf_file.filename:
            if not _allowed_file(pdf_file.filename):
                return jsonify({"error": "Only PDF files are accepted. Please upload a .pdf file."}), 400
            logger.info("Processing uploaded PDF: %s", pdf_file.filename)
            content = extract_pdf_text(pdf_file)
            if not content.strip():
                return jsonify({
                    "error": (
                        "Could not extract text from the PDF. "
                        "The file may be scanned/image-based. "
                        "Try a text-based PDF or paste the notification URL instead."
                    )
                }), 400

        # ── URL path ──
        elif url:
            if not url.startswith(("http://", "https://")):
                url = "https://" + url
            if not _is_safe_url(url):
                return jsonify({
                    "error": "This URL can't be fetched. Please provide a public http(s) notification URL."
                }), 400
            source_url = url
            logger.info("Fetching URL content: %s", url)
            content = fetch_url_content(url)
            if not content.strip():
                return jsonify({"error": "No readable content found at the provided URL."}), 400

        # ── No input ──
        else:
            return jsonify({
                "error": "Please provide a notification URL or upload a PDF file."
            }), 400

        # ── Analyze with Groq ──
        result = analyze_with_groq(content, source_url)
        analysis_text = result.pop("_analysis_text", "")

        # ── Persist so the Home dashboard / watchlist can track it ──
        notif_id = db.save_notification(
            data=result,
            analysis_text=analysis_text,
            source_type="pdf" if source_url == "" else "url",
            source_url=source_url,
        )
        return jsonify({"success": True, "id": notif_id, "data": result})

    except EnvironmentError as e:
        logger.error("Environment error: %s", e)
        return jsonify({"error": str(e)}), 500
    except requests.exceptions.Timeout:
        return jsonify({"error": "The URL took too long to respond. Please try again."}), 408
    except requests.exceptions.ConnectionError:
        return jsonify({"error": "Could not connect to the URL. Please check the link and try again."}), 400
    except requests.RequestException as e:
        return jsonify({"error": f"Failed to fetch URL: {str(e)}"}), 400
    except ValidationError:
        return jsonify({"error": "AI returned data that didn't match the expected format. Please try again."}), 500
    except openai.APIStatusError as e:
        logger.error("Groq API error: %s", e)
        if _is_quota_error(e):
            return jsonify({
                "error": (
                    "Your Groq account has no available quota. Check usage limits "
                    "at console.groq.com/settings/limits."
                )
            }), 402
        if e.status_code == 413:
            return jsonify({
                "error": "The notification content is too long for Groq's free-tier request limit. Try a shorter/simpler source page."
            }), 413
        if e.status_code in GROQ_RETRYABLE_CODES:
            return jsonify({
                "error": "Groq is currently experiencing high demand. Please try again in a minute."
            }), 503
        return jsonify({"error": f"Groq API error: {e.message}"}), 502
    except Exception as e:
        logger.exception("Unexpected error during analysis")
        return jsonify({"error": f"Analysis failed: {str(e)}"}), 500


@app.route("/download-ics")
def download_ics():
    """
    Generate and download an .ics calendar file for a job-related date.

    Query params:
        summary     — event title (e.g., "Last Date to Apply — SSC CGL 2026")
        date        — event date string (e.g., "15-08-2026")
        description — optional event description
    """
    summary = request.args.get("summary", "Job Deadline Reminder")
    date_str = request.args.get("date", "")
    description = request.args.get("description", "")

    if not date_str:
        abort(400, description="The 'date' query parameter is required.")

    ics_content = generate_ics(summary, date_str, description)

    # Sanitize filename
    safe_name = "".join(c if c.isalnum() or c in " _-" else "_" for c in summary)[:50]
    filename = f"{safe_name}.ics"

    return Response(
        ics_content,
        mimetype="text/calendar",
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"',
        },
    )


# ─────────────────────────────────────────────────────────────────
#  API — notifications (list / pin / delete)
# ─────────────────────────────────────────────────────────────────

@app.route("/api/notifications")
def api_list_notifications():
    """Return all saved notifications as JSON (for live refresh)."""
    return jsonify({"notifications": db.list_notifications()})


@app.route("/api/notifications/<notif_id>/pin", methods=["POST"])
def api_toggle_pin(notif_id):
    """Toggle the watchlist/pin flag."""
    new_val = db.toggle_pinned(notif_id)
    if new_val is None:
        return jsonify({"error": "Notification not found."}), 404
    return jsonify({"success": True, "pinned": new_val})


@app.route("/api/notifications/<notif_id>", methods=["DELETE"])
def api_delete_notification(notif_id):
    """Delete a notification."""
    if not db.delete_notification(notif_id):
        return jsonify({"error": "Notification not found."}), 404
    return jsonify({"success": True})


# ─────────────────────────────────────────────────────────────────
#  API — applicant profile
# ─────────────────────────────────────────────────────────────────

@app.route("/api/profile", methods=["GET", "POST"])
def api_profile():
    """Read or upsert the single applicant profile."""
    if request.method == "GET":
        return jsonify({"profile": db.get_profile()})

    payload = request.get_json(silent=True) or {}
    profile = db.save_profile(
        dob=(payload.get("dob") or "").strip(),
        category=(payload.get("category") or "").strip(),
        gender=(payload.get("gender") or "").strip(),
        qualification=(payload.get("qualification") or "").strip(),
    )
    return jsonify({"success": True, "profile": profile})


# ─────────────────────────────────────────────────────────────────
#  API — "Am I eligible?" and grounded Q&A
# ─────────────────────────────────────────────────────────────────

@app.route("/api/notifications/<notif_id>/eligibility", methods=["POST"])
def api_eligibility(notif_id):
    """Run (and cache) an eligibility check for the saved profile."""
    notif = db.get_notification(notif_id)
    if not notif:
        return jsonify({"error": "Notification not found."}), 404
    profile = db.get_profile()
    if not profile or not profile.get("dob"):
        return jsonify({"error": "Set your profile first to check eligibility."}), 400

    try:
        verdict = check_eligibility(profile, notif["data"])
    except openai.APIStatusError as e:
        logger.error("Groq eligibility error: %s", e)
        return jsonify({"error": "Eligibility check failed — please try again."}), 503
    except Exception as e:
        logger.exception("Eligibility check failed")
        return jsonify({"error": f"Eligibility check failed: {e}"}), 500

    db.set_eligibility(notif_id, verdict)
    return jsonify({"success": True, "eligibility": verdict})


@app.route("/api/notifications/<notif_id>/ask", methods=["POST"])
def api_ask(notif_id):
    """Answer a question grounded in this notification's stored data."""
    notif = db.get_notification(notif_id)
    if not notif:
        return jsonify({"error": "Notification not found."}), 404
    question = ((request.get_json(silent=True) or {}).get("question") or "").strip()
    if not question:
        return jsonify({"error": "Please enter a question."}), 400

    try:
        answer = answer_question(notif["data"], notif.get("analysis_text") or "", question)
    except openai.APIStatusError as e:
        logger.error("Groq ask error: %s", e)
        return jsonify({"error": "Couldn't answer right now — please try again."}), 503
    except Exception as e:
        logger.exception("Ask failed")
        return jsonify({"error": f"Couldn't answer: {e}"}), 500

    return jsonify({"success": True, "answer": answer})


# ─────────────────────────────────────────────────────────────────
#  Compare + print + all-events calendar
# ─────────────────────────────────────────────────────────────────

@app.route("/compare")
def compare():
    """Side-by-side comparison of 2–3 saved notifications."""
    ids = [i for i in request.args.get("ids", "").split(",") if i]
    notifs = [n for n in (db.get_notification(i) for i in ids[:3]) if n]
    if len(notifs) < 2:
        return redirect(url_for("index"))
    return render_template("compare.html", notifs=notifs, profile=db.get_profile())


@app.route("/notification/<notif_id>/print")
def notification_print(notif_id):
    """Print / PDF-friendly view of a notification."""
    notif = db.get_notification(notif_id)
    if not notif:
        abort(404)
    return render_template("print.html", notif=notif)


@app.route("/notification/<notif_id>/calendar.ics")
def notification_calendar(notif_id):
    """One .ics containing every known event (start / last / exam) for the job."""
    notif = db.get_notification(notif_id)
    if not notif:
        abort(404)
    data = notif["data"]
    title = data.get("job_title") or "Job Notification"
    dates = data.get("dates") or {}
    events = [
        ("start_date", "Application Opens"),
        ("last_date", "Last Date to Apply"),
        ("exam_date", "Exam Day"),
    ]

    blocks = []
    for key, label in events:
        val = dates.get(key)
        if not val:
            continue
        single = generate_ics(f"{label} — {title}", val, f"{label} for {title}")
        # Keep only the VEVENT block so we can bundle several into one calendar.
        start = single.find("BEGIN:VEVENT")
        end = single.find("END:VEVENT") + len("END:VEVENT\r\n")
        if start != -1 and end != -1:
            blocks.append(single[start:end])

    if not blocks:
        abort(400, description="No dated events to add to calendar.")

    ics_content = (
        "BEGIN:VCALENDAR\r\n"
        "VERSION:2.0\r\n"
        "PRODID:-//JobLens//AI Job Notification Analyzer//EN\r\n"
        "CALSCALE:GREGORIAN\r\n"
        "METHOD:PUBLISH\r\n"
        + "".join(blocks)
        + "END:VCALENDAR\r\n"
    )
    safe_name = "".join(c if c.isalnum() or c in " _-" else "_" for c in title)[:50]
    return Response(
        ics_content,
        mimetype="text/calendar",
        headers={"Content-Disposition": f'attachment; filename="{safe_name}.ics"'},
    )


# ─────────────────────────────────────────────────────────────────
#  Entry point
# ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Non-console stdout (piped, redirected, or a non-UTF-8 Windows codepage)
    # would otherwise crash on the box-drawing/checkmark characters below.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_DEBUG", "0") == "1"

    print("\n" + "═" * 58)
    print("  JobLens — AI Job Notification Analyzer")
    print("═" * 58)
    print(f"  Server  : http://localhost:{port}")
    print(f"  Debug   : {debug}")
    print(f"  Groq Key   : {'✓ Set' if os.environ.get('GROQ_API_KEY') else '✗ NOT SET'}")
    print(f"  Tavily Key : {'✓ Set' if os.environ.get('TAVILY_API_KEY') else '✗ NOT SET'}")
    print("═" * 58 + "\n")

    app.run(host="0.0.0.0", port=port, debug=debug)
