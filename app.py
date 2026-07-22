"""
TrueNotice — AI-Powered Job Notification Analyzer & Fact-Checker
=================================================================
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
from werkzeug.middleware.proxy_fix import ProxyFix
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


class EligibilityRequirement(BaseModel):
    """A specific eligibility criterion not covered by the base applicant profile
    (age, reservation category, gender, qualification are tracked separately)."""
    key: str = Field(
        ...,
        description="Short stable snake_case identifier, e.g. 'domicile_state', "
        "'local_language', 'experience_years', 'pwd_status'",
    )
    question: str = Field(
        ..., description="A short, direct question to ask the applicant, e.g. "
        "'Are you domiciled in Kerala?' or 'How many years of relevant experience do you have?'"
    )
    criterion: str = Field(
        ..., description="The exact source text of the criterion this question checks"
    )


class ResourceLink(BaseModel):
    """A single useful link found in (or verified against) the notification."""
    label: str = Field(
        ..., description="Short human label, e.g. 'Apply online', 'Official notification PDF', "
        "'Free mock test', 'Syllabus PDF', 'Previous year papers', 'Admit card', 'Result'"
    )
    url: str = Field(
        ..., description="The exact, working URL. MUST come verbatim from the notification "
        "or the verified search results — never guessed or constructed."
    )
    category: Optional[str] = Field(
        None,
        description="One of: apply, notification, mock_test, syllabus, previous_papers, "
        "admit_card, result, official_site, other",
    )


class JobDetails(BaseModel):
    """
    Comprehensive structured output for ANY job / recruitment notification —
    government, PSU, private company, campus, or internship — extracted and
    fact-checked by Groq.
    """
    job_title: str = Field(
        ..., description="Full title of the recruitment / job position"
    )
    organization: Optional[str] = Field(
        None,
        description="Full name of the recruiting body / employer / company exactly as it "
        "appears in the notification, e.g. 'Staff Selection Commission', "
        "'Institute of Banking Personnel Selection', 'Infosys Limited', 'Tata Steel'. "
        "Null only if the notification genuinely never names the organization.",
    )
    job_location: Optional[str] = Field(
        None,
        description="Primary work location(s) / posting place as stated, e.g. "
        "'Bengaluru, Karnataka', 'All India', 'Multiple locations'. Null if not stated.",
    )
    employment_type: Optional[str] = Field(
        None,
        description="Nature of employment as stated, e.g. 'Permanent', 'Contract', "
        "'Full-time', 'Part-time', 'Internship', 'Apprenticeship'. Null if not stated.",
    )
    work_mode: Optional[str] = Field(
        None,
        description="Work arrangement if stated: 'On-site', 'Remote', or 'Hybrid'. Null otherwise.",
    )
    experience_required: Optional[str] = Field(
        None,
        description="Required work experience as stated, e.g. 'Freshers', '2-4 years', "
        "'Minimum 5 years in civil engineering'. Null if not stated.",
    )
    key_skills: Optional[List[str]] = Field(
        None,
        description="Key skills / competencies the notification asks for (most relevant for "
        "private / technical roles), as short strings. Null/empty if none are listed.",
    )
    official_pdf_link: Optional[str] = Field(
        None,
        description="Direct URL to the official notification PDF file",
    )
    official_apply_portal: Optional[str] = Field(
        None,
        description="The single most accurate URL where candidates actually apply online. "
        "Prefer the exact application page over a generic homepage.",
    )
    important_links: Optional[List[ResourceLink]] = Field(
        None,
        description="Every other genuinely useful link found for this notification: mock "
        "tests / practice sets, syllabus, previous-year papers, admit-card page, result "
        "page, the recruiter's official website, brochures, etc. One entry per link with a "
        "label, exact url, and category. Only include links whose URL actually appears in "
        "the notification or verified search results. Null/empty if none.",
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
    eligibility_requirements: Optional[List[EligibilityRequirement]] = Field(
        None,
        description="Specific eligibility criteria that need applicant facts beyond age, "
        "reservation category, gender, and qualification (e.g. domicile/state residency, "
        "local language proficiency, minimum work experience, marital status, physical/medical "
        "standards, PwD or ex-serviceman status). Empty/null if the notification has none.",
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

# Render sits in front of this app as a reverse proxy — without this, Flask
# sees the internal http:// connection, not the public https://truenotice.me
# request, which would corrupt every canonical/OG/sitemap URL built from
# request.url_root.
app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1)

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


_SPEC_LABEL_RE = re.compile(
    # A "Label:" that begins the string, follows a sentence/clause break, or a newline.
    # The label may contain letters, digits, spaces, and common punctuation incl. dashes.
    r"(?:^|(?<=[.;])\s+|\n)([A-Z][A-Za-z0-9][A-Za-z0-9 /&().‐-―\-]{0,44}?):\s+"
)
_SPEC_SENTENCE_RE = re.compile(r"(?<=[a-z0-9\)\]])\.\s+(?=[A-Z0-9])")


def _spec_split_items(body: str) -> list[str]:
    """Break a section body into clean, individually-scannable items."""
    body = (body or "").strip().strip(".; ")
    if not body:
        return []
    items: list[str] = []
    for part in re.split(r";\s*", body):
        # A part may still contain a trailing stray sentence — split those too,
        # but not on abbreviation dots (guarded by the surrounding-char classes).
        for sub in _SPEC_SENTENCE_RE.split(part):
            sub = sub.strip().strip(".; ")
            if sub:
                items.append(sub)
    return items


@app.template_filter("spec")
def spec_filter(raw: Optional[str]) -> list:
    """
    Parse a dense freeform 'spec' string (pay scale, eligibility, fees notes,
    helpdesk) into structured blocks so the template can render a neat,
    scannable layout instead of a wall of text.

    Returns a list of {"label": str|None, "items": [str, ...]}.
    """
    if not raw or not raw.strip():
        return []
    text = raw.replace("\r\n", "\n").strip()
    matches = list(_SPEC_LABEL_RE.finditer(text))
    blocks: list[dict] = []

    if not matches:
        return [{"label": None, "items": _spec_split_items(text)}]

    # Any text before the first label is an unlabeled lead-in.
    lead = text[: matches[0].start()].strip(".; \n")
    if lead:
        blocks.append({"label": None, "items": _spec_split_items(lead)})

    for i, m in enumerate(matches):
        label = m.group(1).strip()
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        blocks.append({"label": label, "items": _spec_split_items(text[start:end])})
    return blocks


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
    search_prompt = f"""You are an expert job & recruitment notification analyst. The
notification may come from ANY source — a government body, PSU, bank, private company,
startup, or a campus placement drive. Handle all of them; do not assume it is an
Indian government exam.

TASK — Analyze the job notification content below, using the SEARCH RESULTS
(already gathered from a live web search) to:

1. **Fact-Check** every claim: dates, vacancy counts, pay/CTC, eligibility criteria.
   Flag any discrepancies you find between the provided content and official sources.

2. **Find the official links** — from the search results, identify the recruiter's
   official application page (the exact URL where people apply), the official
   notification/advertisement PDF, and the organization's official website.

3. **Find genuinely useful extra links** that appear in the content or search results:
   free/official mock tests & practice sets, the syllabus, previous-year question papers,
   the admit-card / hall-ticket page, and the result page. Only report links whose URL
   actually appears — never invent a URL.

4. **Supplement** any missing information found in the search results.

CRITICAL: Do not fabricate any value or URL. If something is not stated in the content
or the search results, say it is not stated rather than guessing.

{"Source URL: " + source_url if source_url else "Source: Uploaded PDF document"}

=== NOTIFICATION CONTENT (START) ===
{content}
=== NOTIFICATION CONTENT (END) ===

=== SEARCH RESULTS (START) ===
{search_results_text}
=== SEARCH RESULTS (END) ===

Provide a thorough, well-organized analysis covering EVERY one of these areas
(omit an area only if the source genuinely has nothing on it):
• Job title and recruiting organization / employer
• Job location(s), employment type (permanent/contract/full-time/internship) and work mode (on-site/remote/hybrid)
• Official application URL, official notification PDF, official website
• Other useful links: mock tests / practice, syllabus, previous-year papers, admit card, result
• Important dates: application start, last date, exam/interview date, age cutoff date
• Pay scale / salary / CTC
• Vacancies (total and any category-wise breakdown)
• Eligibility (educational qualifications, age limits, experience, and any relaxations — only if applicable)
• Key skills / competencies required (especially for private / technical roles)
• Selection process (all stages)
• Application fees (by category where applicable) — note if there is no fee
• Documents checklist (for application and/or verification)
• Step-by-step how-to-apply guide
• Study / preparation resources
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
- CRITICAL — never invent, guess, or round a value. Only emit a fact that is
  explicitly present in the verified analysis below. If a field is not stated,
  set it to null (or an empty list) rather than filling a plausible-looking value.
- For job_title, use the exact recruitment title. For organization, give the full
  name of the recruiting body / employer / company as written (e.g. "Staff Selection
  Commission" or "Infosys Limited", not an abbreviation or the website domain). Null if never named.
- Populate job_location, employment_type, work_mode, experience_required and key_skills
  whenever the notification states them — these matter for private / corporate roles too.
  Leave any of them null if not stated (never guess).
- For official_apply_portal, give the single most accurate URL where candidates actually
  apply. For important_links, add every OTHER useful link found — mock tests / practice,
  syllabus, previous-year papers, admit-card page, result page, official website, brochures —
  each with a label, exact url, and category (apply/notification/mock_test/syllabus/
  previous_papers/admit_card/result/official_site/other). ONLY include a link whose URL
  literally appears in the verified analysis. Never invent or guess a URL. Null/empty if none.
- For date fields, prefer the format "DD-MM-YYYY". Use a descriptive string if exact date is unknown.
- For list fields (documents_checklist, study_resources), return arrays of strings.
- If a field truly has no data, use null.
- For eligibility, include both educational qualifications AND age limits, as a single freeform text block.
- For eligibility_requirements, scan the eligibility clauses for anything that needs an applicant fact
  BEYOND age, education qualification, reservation category, and gender (those four are already tracked
  separately) — e.g. domicile/state residency, local language proficiency, minimum work experience,
  marital status, physical/medical standards, PwD or ex-serviceman status, community/caste sub-certificates.
  Emit one entry per such criterion with a short snake_case "key", a direct "question" to ask the applicant,
  and the source "criterion" text. Return null/empty if there are none.
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
        "PRODID:-//TrueNotice//AI Job Notification Analyzer//EN\r\n"
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

    extra_answers = profile.get("extra") or {}
    requirements = job.get("eligibility_requirements") or []
    extra_block = "(none)"
    if requirements:
        extra_block = "\n".join(
            f"- {r.get('question')}: {extra_answers.get(r.get('key')) or 'unknown'}"
            for r in requirements
        )

    prompt = f"""You are an eligibility adjudicator for Indian government/private job
recruitment. Decide whether the applicant meets the notification's criteria.

APPLICANT PROFILE:
- Date of birth: {profile.get('dob') or 'unknown'}
- Computed age at cutoff{f" ({cutoff})" if cutoff else ""}: {computed_age or 'unknown'}
- Category: {profile.get('category') or 'unknown'}
- Gender: {profile.get('gender') or 'unknown'}
- Highest qualification: {profile.get('qualification') or 'unknown'}

NOTIFICATION-SPECIFIC FACTS (collected for this notification's own extra criteria):
{extra_block}

NOTIFICATION ELIGIBILITY CRITERIA:
{eligibility_text}

Rules:
- Consider age limits (apply category-based relaxation if the criteria mention it),
  educational qualification, category/gender conditions, and the notification-specific
  facts above (domicile, language, experience, etc. — whichever apply to this notification).
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

    prompt = f"""You are TrueNotice, answering a candidate's question about ONE specific
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


def _parse_iso(value: Optional[str]) -> Optional[datetime]:
    """Parse an ISO-8601 timestamp (as stored in created_at). None on failure."""
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except (ValueError, TypeError):
        return None


@app.template_global("vac_int")
def vac_int(notif: dict) -> int:
    """Template-facing wrapper over _vacancy_int for the 'most vacancies' sort."""
    return _vacancy_int(notif)


@app.template_global("org_of")
def org_of(notif: dict) -> Optional[str]:
    """
    Truthful recruiting-organization label for a notification: the extracted
    organization name when the model captured one, otherwise the source
    domain. Never fabricated — falls back to None if neither exists.
    """
    data = notif.get("data") or {}
    org = (data.get("organization") or "").strip()
    if org:
        return org
    return _org_name_from_url(notif.get("source_url")) or _org_name_from_url(
        data.get("official_apply_portal")
    )


@app.route("/")
def index():
    """Public landing page — the marketing front door for TrueNotice."""
    notifications = db.list_notifications()
    # A few honest counts so the page can show live proof instead of stock claims.
    stats = {
        "tracked": len(notifications),
        # A spread across sectors so the strip signals "any job", not just govt exams.
        "sources": ["SSC", "UPSC", "IBPS", "Railway RRB", "State PSCs", "PSUs", "Private employers", "Company careers"],
    }
    return render_template("landing.html", stats=stats)


@app.route("/dashboard")
def dashboard():
    """Home dashboard — aggregate view of all tracked notifications."""
    notifications = db.list_notifications()
    profile = db.get_profile()

    closing_soon = 0
    total_vacancies = 0
    eligible = 0
    eligibility_checked = 0
    added_this_week = 0
    now = datetime.now()
    for n in notifications:
        st = status_of(n.get("last_date"))
        if st["key"] in ("soon", "today"):
            closing_soon += 1
        total_vacancies += _vacancy_int(n)
        elig = n.get("eligibility") or {}
        if elig.get("verdict"):
            eligibility_checked += 1
        if elig.get("verdict") == "eligible":
            eligible += 1
        created = _parse_iso(n.get("created_at"))
        if created and (now - created).days < 7:
            added_this_week += 1

    active = sum(
        1 for n in notifications if status_of(n.get("last_date"))["key"] != "closed"
    )

    kpis = {
        "tracked": len(notifications),
        "closing_soon": closing_soon,
        "eligible": eligible,
        "eligibility_checked": eligibility_checked,
        "total_vacancies": total_vacancies,
        "added_this_week": added_this_week,
    }
    meta = {
        "last_updated": now.strftime("%d %b %Y, %H:%M"),
        "active": active,
        "closing_this_week": closing_soon,
    }
    return render_template(
        "home.html", notifications=notifications, profile=profile, kpis=kpis, meta=meta
    )


def _org_name_from_url(url: Optional[str]) -> Optional[str]:
    """Best-effort, truthful organization identifier: the source's domain."""
    if not url:
        return None
    try:
        return urlparse(url).netloc or None
    except ValueError:
        return None


def build_job_posting_ld(notif: dict) -> dict:
    """
    schema.org JobPosting JSON-LD — makes the page eligible for Google's
    Jobs rich results and gives AI crawlers unambiguous, structured facts
    (dates, vacancies, pay) instead of forcing them to parse prose.
    """
    data = notif["data"]
    dates = data.get("dates") or {}
    vacancies = data.get("vacancies") or {}
    org_name = (
        (data.get("organization") or "").strip()
        or _org_name_from_url(notif.get("source_url"))
        or _org_name_from_url(data.get("official_apply_portal"))
        or _org_name_from_url(data.get("official_pdf_link"))
    )

    description_parts = []
    if data.get("eligibility"):
        description_parts.append(f"Eligibility: {data['eligibility']}")
    if data.get("pay_scale"):
        description_parts.append(f"Pay scale: {data['pay_scale']}")
    if vacancies.get("total"):
        description_parts.append(f"Total vacancies: {vacancies['total']}")
    if data.get("fact_check_summary"):
        description_parts.append(f"Fact-check: {data['fact_check_summary']}")
    description = " ".join(description_parts) or data.get("job_title", "")

    posted = (notif.get("created_at") or "")[:10] or None
    valid_through = None
    if dates.get("last_date"):
        dt = _parse_date(dates["last_date"])
        if dt:
            valid_through = dt.strftime("%Y-%m-%d") + "T23:59:59"

    ld = {
        "@context": "https://schema.org",
        "@type": "JobPosting",
        "title": data.get("job_title"),
        "description": description,
        "hiringOrganization": {
            "@type": "Organization",
            "name": org_name or "Recruiting organization — see official notification",
        },
        "directApply": bool(data.get("official_apply_portal")),
    }

    # Location — use the stated place when we have it; only fall back to a
    # country when we don't, and mark remote roles as telecommute.
    location = (data.get("job_location") or "").strip()
    if location:
        ld["jobLocation"] = {
            "@type": "Place",
            "address": {"@type": "PostalAddress", "addressLocality": location},
        }
    else:
        ld["jobLocation"] = {
            "@type": "Place",
            "address": {"@type": "PostalAddress", "addressCountry": "IN"},
        }
    if (data.get("work_mode") or "").strip().lower() == "remote":
        ld["jobLocationType"] = "TELECOMMUTE"

    # Employment type — map free text onto schema.org's controlled tokens.
    et_raw = (data.get("employment_type") or "").lower()
    et_map = {
        "full": "FULL_TIME", "permanent": "FULL_TIME", "regular": "FULL_TIME",
        "part": "PART_TIME", "contract": "CONTRACTOR", "temporary": "TEMPORARY",
        "intern": "INTERN", "apprentice": "OTHER",
    }
    for needle, token in et_map.items():
        if needle in et_raw:
            ld["employmentType"] = token
            break

    if posted:
        ld["datePosted"] = posted
    if valid_through:
        ld["validThrough"] = valid_through
    if data.get("official_apply_portal"):
        ld["url"] = data["official_apply_portal"]
    return ld


def build_faq_ld(notif: dict) -> Optional[dict]:
    """
    FAQPage JSON-LD auto-generated from extracted fields — the single
    highest-leverage GEO asset here: the Q/A phrasing matches how people
    actually query AI answer engines and Google's "People also ask."
    """
    data = notif["data"]
    dates = data.get("dates") or {}
    vacancies = data.get("vacancies") or {}
    title = data.get("job_title") or "this notification"

    qa = []
    if dates.get("last_date"):
        qa.append((
            f"What is the last date to apply for {title}?",
            f"The last date to apply for {title} is {dates['last_date']}.",
        ))
    if dates.get("exam_date"):
        qa.append((
            f"When is the exam for {title}?",
            f"The exam for {title} is scheduled on {dates['exam_date']}.",
        ))
    if vacancies.get("total"):
        qa.append((
            f"How many vacancies are there in {title}?",
            f"{title} has {vacancies['total']} total vacancies.",
        ))
    if data.get("eligibility"):
        qa.append((f"What is the eligibility criteria for {title}?", data["eligibility"]))
    if data.get("pay_scale"):
        qa.append((f"What is the pay scale for {title}?", data["pay_scale"]))
    if data.get("fees") and (data["fees"].get("breakdown") or data["fees"].get("notes")):
        fee_bits = [
            f"{b['category']}: {b['amount']}"
            for b in (data["fees"].get("breakdown") or []) if b.get("category")
        ]
        fee_text = "; ".join(fee_bits)
        if data["fees"].get("notes"):
            fee_text = (fee_text + ". " if fee_text else "") + data["fees"]["notes"]
        if fee_text:
            qa.append((f"What is the application fee for {title}?", fee_text))

    if not qa:
        return None
    return {
        "@context": "https://schema.org",
        "@type": "FAQPage",
        "mainEntity": [
            {"@type": "Question", "name": q, "acceptedAnswer": {"@type": "Answer", "text": a}}
            for q, a in qa
        ],
    }


@app.route("/notification/<notif_id>")
def notification_detail(notif_id):
    """Redesigned detail view for a single saved notification."""
    notif = db.get_notification(notif_id)
    if not notif:
        abort(404)
    profile = db.get_profile()
    return render_template(
        "detail.html",
        notif=notif,
        profile=profile,
        job_ld=build_job_posting_ld(notif),
        faq_ld=build_faq_ld(notif),
    )


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


@app.route("/api/profile/extra", methods=["POST"])
def api_profile_extra():
    """Merge notification-specific eligibility answers (domicile, language, etc.)
    into the profile's dynamic field store."""
    payload = request.get_json(silent=True) or {}
    answers = payload.get("answers")
    if not isinstance(answers, dict):
        return jsonify({"error": "answers must be an object"}), 400
    answers = {str(k).strip(): str(v).strip() for k, v in answers.items() if str(k).strip()}
    profile = db.save_profile_extra(answers)
    return jsonify({"success": True, "profile": profile})


# ─────────────────────────────────────────────────────────────────
#  API — "Am I eligible?" and grounded Q&A
# ─────────────────────────────────────────────────────────────────

@app.route("/api/notifications/<notif_id>/eligibility", methods=["POST"])
def api_eligibility(notif_id):
    """Run (and cache) an eligibility check for the saved profile.

    Notifications carry their own dynamic eligibility_requirements (fields
    beyond age/category/gender/qualification, e.g. domicile or local-language
    proficiency). If any of those aren't yet answered, respond with
    need_fields instead of running the check, so the caller can prompt for
    exactly the facts this notification requires.
    """
    notif = db.get_notification(notif_id)
    if not notif:
        return jsonify({"error": "Notification not found."}), 404
    profile = db.get_profile()
    if not profile or not profile.get("dob"):
        return jsonify({"error": "Set your profile first to check eligibility."}), 400

    requirements = notif["data"].get("eligibility_requirements") or []
    extra_answers = profile.get("extra") or {}
    missing = [r for r in requirements if not (extra_answers.get(r.get("key")) or "").strip()]
    if missing:
        return jsonify({"need_fields": missing})

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


# ---------------------------------------------------------------------------
# Trust and legal pages
# ---------------------------------------------------------------------------

@app.context_processor
def inject_site_contact():
    """Expose the configured support address to pages and the shared footer."""
    return {
        "site_contact_email": os.getenv("CONTACT_EMAIL", "support@truenotice.app"),
        "adsense_publisher": "ca-pub-9140574176918803",
        "adsense_slots": {
            "dashboard": os.getenv("ADSENSE_SLOT_DASHBOARD", ""),
            "detail": os.getenv("ADSENSE_SLOT_DETAIL", ""),
            "legal": os.getenv("ADSENSE_SLOT_LEGAL", ""),
        },
    }


@app.route("/about")
def about():
    return render_template("about.html")


@app.route("/contact")
def contact():
    return render_template("contact.html")


@app.route("/privacy-policy")
def privacy_policy():
    return render_template("privacy-policy.html")


@app.route("/terms-of-service")
def terms_of_service():
    return render_template("terms-of-service.html")


@app.route("/cookie-policy")
def cookie_policy():
    return render_template("cookie-policy.html")


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
        "PRODID:-//TrueNotice//AI Job Notification Analyzer//EN\r\n"
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
#  SEO / GEO — robots.txt, sitemap.xml, llms.txt
# ─────────────────────────────────────────────────────────────────

# Search engines get the default `*` group. AI answer-engine crawlers get
# their own explicit groups (same rules) so the site's AI-friendly stance
# is unambiguous to anyone auditing robots.txt — some crawlers only
# recognize the version that matches their exact user-agent token.
_AI_CRAWLERS = [
    "GPTBot", "ChatGPT-User", "OAI-SearchBot",   # OpenAI
    "ClaudeBot", "anthropic-ai", "Claude-Web",    # Anthropic
    "PerplexityBot", "Perplexity-User",            # Perplexity
    "Google-Extended",                              # Google Gemini / AI Overviews training
    "CCBot",                                        # Common Crawl (feeds many LLMs)
    "Applebot-Extended",                            # Apple Intelligence
]


@app.route("/robots.txt")
def robots_txt():
    """Allow indexing of public content; keep API/print/compare out of search."""
    disallow = ["/api/", "/compare", "/*/print"]
    lines = ["User-agent: *"] + [f"Disallow: {p}" for p in disallow] + [""]
    for bot in _AI_CRAWLERS:
        lines.append(f"User-agent: {bot}")
        lines.extend(f"Disallow: {p}" for p in disallow)
        lines.append("")
    lines.append(f"Sitemap: {request.url_root.rstrip('/')}/sitemap.xml")
    return Response("\n".join(lines), mimetype="text/plain")


@app.route("/sitemap.xml")
def sitemap_xml():
    """XML sitemap covering the home page and every saved notification."""
    base = request.url_root.rstrip("/")
    urls = [
        {"loc": base + "/", "changefreq": "weekly", "priority": "1.0", "lastmod": None},
        {"loc": base + "/dashboard", "changefreq": "daily", "priority": "0.9", "lastmod": None},
        {"loc": base + "/about", "changefreq": "monthly", "priority": "0.5", "lastmod": None},
        {"loc": base + "/contact", "changefreq": "monthly", "priority": "0.5", "lastmod": None},
        {"loc": base + "/privacy-policy", "changefreq": "yearly", "priority": "0.3", "lastmod": None},
        {"loc": base + "/terms-of-service", "changefreq": "yearly", "priority": "0.3", "lastmod": None},
        {"loc": base + "/cookie-policy", "changefreq": "yearly", "priority": "0.3", "lastmod": None},
    ]
    for n in db.list_notifications():
        lastmod = (n.get("created_at") or "")[:10] or None
        urls.append({
            "loc": f"{base}/notification/{n['id']}",
            "changefreq": "daily",
            "priority": "0.8",
            "lastmod": lastmod,
        })

    parts = ['<?xml version="1.0" encoding="UTF-8"?>',
             '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">']
    for u in urls:
        parts.append("<url>")
        parts.append(f"<loc>{escape(u['loc'])}</loc>")
        if u["lastmod"]:
            parts.append(f"<lastmod>{u['lastmod']}</lastmod>")
        parts.append(f"<changefreq>{u['changefreq']}</changefreq>")
        parts.append(f"<priority>{u['priority']}</priority>")
        parts.append("</url>")
    parts.append("</urlset>")
    return Response("".join(parts), mimetype="application/xml")


@app.route("/llms.txt")
def llms_txt():
    """
    llms.txt (llmstxt.org) — a curated, plain-text map of the site for AI
    crawlers and RAG pipelines, mirroring what robots.txt/sitemap.xml do
    for search engines.
    """
    notifs = db.list_notifications()
    base = request.url_root.rstrip("/")
    lines = [
        "# TrueNotice",
        "",
        "> TrueNotice tracks government and private job notifications, "
        "fact-checks every detail (dates, vacancies, pay, eligibility) "
        "against official sources using AI, and tells applicants whether "
        "they personally qualify.",
        "",
        "Each notification page states its confidence level and lists any "
        "discrepancies found versus the official source — cite the "
        "specific notification page, not this index, when referencing a "
        "date, vacancy count, or eligibility detail.",
        "",
        "## Notifications",
        "",
    ]
    for n in notifs[:200]:
        last_date = n.get("last_date")
        suffix = f" — last date to apply: {last_date}" if last_date else ""
        lines.append(f"- [{n['job_title']}]({base}/notification/{n['id']}){suffix}")
    lines += [
        "",
        "## Site",
        "",
        f"- [Home]({base}/) — what TrueNotice does and how verification works",
        f"- [Dashboard]({base}/dashboard) — your tracked notifications register",
    ]
    return Response("\n".join(lines), mimetype="text/plain")


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
    print("  TrueNotice — AI Job Notification Analyzer")
    print("═" * 58)
    print(f"  Server  : http://localhost:{port}")
    print(f"  Debug   : {debug}")
    print(f"  Groq Key   : {'✓ Set' if os.environ.get('GROQ_API_KEY') else '✗ NOT SET'}")
    print(f"  Tavily Key : {'✓ Set' if os.environ.get('TAVILY_API_KEY') else '✗ NOT SET'}")
    print("═" * 58 + "\n")

    app.run(host="0.0.0.0", port=port, debug=debug)
