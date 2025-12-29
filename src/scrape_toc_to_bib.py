#!/usr/bin/env python3
"""
Scrape Curran Associates "webtoc" PDFs (table of contents) for ICLR proceedings
and write a BibLaTeX-compatible .bib file.

Optionally adds OpenReview forum URLs by bulk-downloading OpenReview notes per year
and joining on normalized title (author overlap tie-break; fuzzy fallback).

This version includes a --force-ipv4 option to work around broken IPv6 networks.

Dependencies:
  pip install pymupdf openreview-py
Optional:
  pip install rapidfuzz
"""

from __future__ import annotations

import argparse
import collections
import difflib
import json
import os
import re
import socket
import tempfile
import unicodedata
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    import fitz  # PyMuPDF
except ImportError as e:
    raise SystemExit("Missing dependency: PyMuPDF. Install with: pip install pymupdf") from e


# ----------------------------
# IPv4 forcing (workaround for broken IPv6)
# ----------------------------

def force_ipv4_only() -> None:
    """
    Monkey-patch socket.getaddrinfo to return only AF_INET results.
    Affects this Python process only.
    """
    orig_getaddrinfo = socket.getaddrinfo

    def ipv4_getaddrinfo(host, port, family=0, type=0, proto=0, flags=0):
        res = orig_getaddrinfo(host, port, family, type, proto, flags)
        return [r for r in res if r[0] == socket.AF_INET]

    socket.getaddrinfo = ipv4_getaddrinfo  # type: ignore[assignment]


# ----------------------------
# Text helpers
# ----------------------------

ACRONYMS = {
    "VAE", "CNN", "RNN", "LSTM", "GAN", "BERT", "GPT", "LSH", "ICLR", "RELU",
    "NLP", "RL", "GNN", "VQ", "AE", "QA", "SVM", "PCA", "MCMC", "SGD", "ELBO",
    "VIT", "MLP", "KL", "MSE",
}
STOPWORDS = {
    "a", "an", "the", "and", "or", "of", "for", "to", "in", "with", "without",
    "on", "between", "by", "from", "into", "via", "using", "use", "towards",
    "toward", "learn", "learning", "deep", "neural", "networks",
}

SECTION_RE = re.compile(
    r"^(VOLUME\s+\d+|POSTER\s+PRESENTATIONS|POSTERS\s+PRESENTATIONS|POSTER\s+PAPERS|"
    r"ORAL\s+PRESENTATIONS|ORAL\s+PAPERS|SPOTLIGHT\s+PRESENTATIONS|SPOTLIGHT\s+PAPERS|"
    r"INVITED|KEYNOTE|WORKSHOP|TUTORIAL|AUTHOR\s+INDEX|SUBJECT\s+INDEX)\b",
    re.I,
)
NOISE_RE = re.compile(
    r"^(www\.|Curran Associates|Additional copies|Phone:|Fax:|Email:|Web:)",
    re.I,
)

def is_mostly_upper(s: str) -> bool:
    letters = [c for c in s if c.isalpha()]
    if not letters:
        return False
    upper = sum(1 for c in letters if c.isupper())
    return (upper / len(letters)) > 0.7

def smart_title_case(title: str) -> str:
    t = title.strip()
    if is_mostly_upper(t):
        t = t.lower().title()
        small = {"A","An","The","And","Or","Of","For","To","In","On","With","Without","Between","By","From","Into","Via"}
        words = t.split()
        out = []
        for i, w in enumerate(words):
            out.append(w.lower() if (i > 0 and w in small) else w)
        t = " ".join(out)

    def fix_token(tok: str) -> str:
        m = re.match(r"^([^A-Za-z0-9]*)([A-Za-z0-9\-]+)([^A-Za-z0-9]*)$", tok)
        if not m:
            return tok
        pre, core, post = m.groups()
        up = core.upper()
        if up in ACRONYMS:
            return pre + up + post
        return tok

    return " ".join(fix_token(w) for w in t.split())

def protect_acronyms_for_biblatex(title: str) -> str:
    return re.sub(r"\b([A-Z]{2,})\b", r"{\1}", title)

def normalize_author_list(authors_raw: str) -> str:
    s = " ".join(authors_raw.replace("\u00a0", " ").split())
    s = re.sub(r"\s*,\s*and\s+", ", ", s, flags=re.I)
    s = re.sub(r"\s+and\s+", ", ", s, flags=re.I)
    parts = [p.strip() for p in s.split(",") if p.strip()]
    parts = [re.sub(r"^\band\b\s+", "", p, flags=re.I).strip() for p in parts]
    parts = [p.rstrip(".") for p in parts]
    return " and ".join(parts)

def slugify_ascii(s: str) -> str:
    s = unicodedata.normalize("NFKD", s)
    s = s.encode("ascii", "ignore").decode("ascii")
    s = re.sub(r"[^A-Za-z0-9]+", "", s)
    return s

def title_slug(title: str, max_words: int = 3) -> str:
    words = re.findall(r"[A-Za-z0-9]+", title)
    picked: List[str] = []
    for w in words:
        wl = w.lower()
        if wl in STOPWORDS:
            continue
        picked.append(w)
        if len(picked) >= max_words:
            break
    return "".join(picked) or "Paper"

def title_first_word(title: str) -> str:
    words = re.findall(r"[A-Za-z0-9]+", title)
    return words[0] if words else "Paper"

def first_author_lastname(author_field: str) -> str:
    first = author_field.split(" and ")[0]
    tokens = re.findall(r"[A-Za-zÀ-ÖØ-öø-ÿ\-']+", first)
    return tokens[-1] if tokens else "Anon"

def make_key(author_field: str, year: int, title: str) -> str:
    last = (slugify_ascii(first_author_lastname(author_field)) or "anon").lower()
    word = (slugify_ascii(title_first_word(title)) or "paper").lower()
    return f"{last}_{word}_{year}"

def strip_section_prefix(title: str) -> str:
    return re.sub(
        r"^(ORAL\s+PAPERS|POSTER\s+PAPERS|SPOTLIGHT\s+PAPERS|POSTERS?\s+PRESENTATIONS?|"
        r"POSTER\s+PRESENTATIONS?|SPOTLIGHT\s+PRESENTATIONS?)\s+",
        "",
        title,
        flags=re.I,
    ).strip()

def bib_escape(s: str) -> str:
    s = s.replace("\\", "\\\\")
    s = s.replace("&", "\\&")
    s = s.replace("%", "\\%")
    s = s.replace("#", "\\#")
    return s

def ordinal(n: int) -> str:
    if 11 <= (n % 100) <= 13:
        suf = "th"
    else:
        suf = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suf}"

def iclr_edition(year: int) -> int:
    # ICLR 2013 is 1st => 2017 is 5th
    return year - 2012


# ----------------------------
# PDF parsing
# ----------------------------

@dataclass
class TocEntry:
    title: str
    authors_raw: str
    start_page: int
    pages: str = ""

def extract_lines(pdf_path: str) -> List[str]:
    doc = fitz.open(pdf_path)
    try:
        lines: List[str] = []
        for i in range(doc.page_count):
            txt = doc.load_page(i).get_text("text")
            lines.extend([ln.rstrip() for ln in txt.splitlines()])
        return lines
    finally:
        doc.close()

def parse_old(pdf_path: str) -> List[TocEntry]:
    lines = extract_lines(pdf_path)

    start = 0
    for idx, ln in enumerate(lines):
        if re.search(r"TABLE OF CONTENTS", ln, re.I):
            start = idx + 1
            break
    lines = lines[start:]

    def looks_like_page_line(ln: str) -> bool:
        return bool(re.search(r"\.{5,}\s*\d+\s*$", ln)) or bool(
            re.search(r"\s{3,}\d+\s*$", ln) and ("," not in ln)
        )

    def extract_title_page(ln: str) -> Tuple[Optional[str], Optional[int]]:
        m = re.search(r"^(.*?)(?:\.{5,}|\s{3,})\s*(\d+)\s*$", ln)
        if not m:
            return None, None
        return m.group(1).strip(), int(m.group(2))

    entries: List[TocEntry] = []
    buf: List[str] = []
    i = 0
    while i < len(lines):
        ln = lines[i].strip()
        if (not ln) or SECTION_RE.match(ln) or NOISE_RE.match(ln):
            buf = []
            i += 1
            continue

        if looks_like_page_line(lines[i]):
            title_part, page = extract_title_page(lines[i])
            if title_part is None or page is None:
                i += 1
                continue
            title = strip_section_prefix(" ".join(buf + [title_part]).strip())
            buf = []
            i += 1

            auth: List[str] = []
            while i < len(lines):
                s = lines[i].strip()
                if not s:
                    if auth:
                        i += 1
                        break
                    i += 1
                    continue
                if SECTION_RE.match(s) or NOISE_RE.match(s):
                    if auth:
                        break
                    i += 1
                    continue
                if looks_like_page_line(lines[i]) and auth:
                    break
                auth.append(s)
                i += 1

            authors_raw = " ".join(auth).strip()
            if authors_raw and len(title) > 8:
                entries.append(TocEntry(title=title, authors_raw=authors_raw, start_page=page))
        else:
            if not re.fullmatch(r"\d+", ln) and not NOISE_RE.match(ln):
                buf.append(ln)
            i += 1

    return entries

def parse_new(pdf_path: str) -> List[TocEntry]:
    lines = extract_lines(pdf_path)

    start = 0
    for idx, ln in enumerate(lines):
        if "Title and Authors" in ln:
            start = idx + 1
            break

    entries: List[TocEntry] = []
    buf: List[str] = []

    for ln in lines[start:]:
        s = ln.strip()
        if not s:
            continue
        if s in ("Page#", "Page #"):
            continue
        if ("International Conference on Learning Representations" in s) or s.startswith("(ICLR") or ("ISBN" in s):
            continue

        if re.fullmatch(r"\d+", s):
            page = int(s)
            block = buf
            buf = []
            if not block:
                continue

            split_idx = None
            for j, l in enumerate(block):
                if "," in l:
                    split_idx = j
                    break
            if split_idx is None:
                split_idx = max(len(block) - 1, 0)

            title = " ".join(block[:split_idx]).strip()
            authors_raw = " ".join(block[split_idx:]).strip()
            if title and authors_raw:
                entries.append(TocEntry(title=title, authors_raw=authors_raw, start_page=page))
        else:
            buf.append(s)

    return entries

def detect_format(pdf_path: str) -> str:
    lines = extract_lines(pdf_path)
    return "old" if any(re.search(r"TABLE OF CONTENTS", l, re.I) for l in lines) else "new"

def compute_page_ranges(entries: List[TocEntry]) -> List[TocEntry]:
    es = sorted(entries, key=lambda e: e.start_page)
    for idx, e in enumerate(es):
        start = e.start_page
        end = None
        if idx + 1 < len(es):
            nxt = es[idx + 1].start_page
            if nxt > start:
                end = nxt - 1
        e.pages = f"{start}-{end}" if (end is not None and end >= start) else f"{start}"
    return es


# ----------------------------
# Input discovery
# ----------------------------

def iter_pdf_paths(input_path: str) -> List[str]:
    if os.path.isdir(input_path):
        out: List[str] = []
        for root, _, files in os.walk(input_path):
            for f in files:
                if f.lower().endswith(".pdf"):
                    out.append(os.path.join(root, f))
        return sorted(out)

    if input_path.lower().endswith(".zip"):
        tmpdir = tempfile.mkdtemp(prefix="toc_zip_")
        with zipfile.ZipFile(input_path) as zf:
            zf.extractall(tmpdir)
        return iter_pdf_paths(tmpdir)

    raise SystemExit(f"Unsupported input: {input_path} (expected a folder or .zip)")

def year_from_filename(path: str) -> Optional[int]:
    m = re.search(r"(?:ICLR|iclr)(\d{4})", os.path.basename(path))
    return int(m.group(1)) if m else None


# ----------------------------
# OpenReview integration
# ----------------------------

# OpenReview content keys (vary across years / API versions)
_TITLE_KEYS = (
    "title",
    "paper_title",
    "paper title",
    "submission_title",
    "submission title",
)
_AUTHOR_KEYS = (
    "authors",
    "authorids",
    "author_ids",
    "author names",
    "author_names",
)

def _unwrap_openreview_value(val: Any) -> Any:
    if val is None:
        return None
    cur = val
    while True:
        if isinstance(cur, dict):
            if "value" in cur:
                cur = cur.get("value")
                continue
            if "values" in cur:
                cur = cur.get("values")
                continue
            return cur
        for attr in ("value", "values"):
            if hasattr(cur, attr):
                try:
                    got = getattr(cur, attr)
                    if callable(got):
                        continue
                    cur = got
                    break
                except Exception:
                    pass
        else:
            return cur

def _note_to_json(note: Any) -> Dict[str, Any]:
    if isinstance(note, dict):
        return note
    tj = getattr(note, "to_json", None)
    if callable(tj):
        try:
            j = tj()
            if isinstance(j, dict):
                return j
        except Exception:
            pass
    out: Dict[str, Any] = {}
    for k in ("id", "_id", "content"):
        try:
            out[k] = getattr(note, k)
        except Exception:
            pass
    return out

def _content_dict(note: Any) -> Dict[str, Any]:
    if isinstance(note, dict):
        c = note.get("content", None)
        if isinstance(c, dict):
            return c
    j = _note_to_json(note)
    c = j.get("content", None)
    if isinstance(c, dict):
        return c
    c2 = getattr(note, "content", None)
    if isinstance(c2, dict):
        return c2
    try:
        return dict(c2)  # type: ignore[arg-type]
    except Exception:
        return {}

def _content_get(content: Any, key: str) -> Any:
    if content is None:
        return None
    if not isinstance(content, dict):
        try:
            content = dict(content)  # type: ignore[arg-type]
        except Exception:
            return None
    val = content.get(key)
    if val is None:
        lk = key.lower()
        for kk in content.keys():
            if str(kk).lower() == lk:
                val = content.get(kk)
                break
    return _unwrap_openreview_value(val)

def _openreview_get_title(note: Any) -> str:
    c = _content_dict(note)
    for k in _TITLE_KEYS:
        t = _content_get(c, k)
        if not t:
            continue
        if isinstance(t, str):
            return t.strip()
        if isinstance(t, list) and t and isinstance(t[0], str):
            return " ".join(x.strip() for x in t if str(x).strip()).strip()
    for kk in c.keys():
        if "title" in str(kk).lower():
            t = _content_get(c, str(kk))
            if isinstance(t, str) and t.strip():
                return t.strip()
    return ""

def _openreview_get_authors(note: Any) -> List[str]:
    c = _content_dict(note)
    for k in _AUTHOR_KEYS:
        a = _content_get(c, k)
        if not a:
            continue
        if isinstance(a, str):
            if "," in a:
                parts = [p.strip() for p in a.split(",") if p.strip()]
                if parts:
                    return parts
            return [a.strip()]
        if isinstance(a, list):
            out: List[str] = []
            for x in a:
                x = _unwrap_openreview_value(x)
                if x is None:
                    continue
                s = str(x).strip()
                if s:
                    out.append(s)
            return out
    for kk in c.keys():
        if "author" in str(kk).lower():
            a = _content_get(c, str(kk))
            if isinstance(a, str) and a.strip():
                return [a.strip()]
    return []

def _note_id(note: Any) -> str:
    j = _note_to_json(note)
    nid = j.get("id") or j.get("_id") or getattr(note, "id", "") or getattr(note, "_id", "")
    return str(nid or "").strip()

def _normalize_title_for_join(title: str) -> str:
    t = title.lower().strip()
    t = re.sub(r"\{|\}", "", t)
    t = re.sub(r"[\u2010-\u2015]", "-", t)  # hyphen variants
    t = re.sub(r"[^a-z0-9]+", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t

def _probe_notes_one(client: Any, invitation: str) -> Tuple[int, List[Any]]:
    """
    Try to fetch a handful of notes for an invitation cheaply.
    Works with openreview-py return types that may be iterable but not indexable.
    """
    get_notes = getattr(client, "get_notes", None)
    if callable(get_notes):
        try:
            notes = get_notes(invitation=invitation, limit=5)
            it = iter(notes)
            samples = [n for n in (next(it, None) for _ in range(5)) if n is not None]
            if samples:
                return (len(samples), samples)
            return (0, [])
        except Exception:
            return (0, [])

    get_all_notes = getattr(client, "get_all_notes", None)
    if callable(get_all_notes):
        try:
            notes = get_all_notes(invitation=invitation)
            it = iter(notes)
            samples = [n for n in (next(it, None) for _ in range(5)) if n is not None]
            if samples:
                return (len(samples), samples)
            return (0, [])
        except Exception:
            return (0, [])

    return (0, [])

def _find_working_submission_invitation(client: Any, year: int) -> Optional[str]:
    venue_prefixes = [
        f"ICLR.cc/{year}/conference",
        f"ICLR.cc/{year}/Conference",
    ]
    # Prefer lowercase /-/submission first (empirically needed for ICLR 2017)
    suffixes = [
        "/-/submission",
        "/-/Submission",
        "/-/blind_submission",
        "/-/Blind_Submission",
        "/-/paper",
        "/-/Paper",
        "/-/accepted_paper",
        "/-/Accepted_Paper",
        "/-/accepted",
        "/-/Accepted",
    ]
    for vp in venue_prefixes:
        for suf in suffixes:
            inv = vp + suf
            found, samples = _probe_notes_one(client, inv)
            if found and any(_openreview_get_title(n) for n in samples):
                return inv
    return None

def _build_year_index_from_notes(notes: List[Any]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for n in notes:
        title = _openreview_get_title(n)
        if not title:
            continue
        authors = _openreview_get_authors(n)
        lastnames: List[str] = []
        for a in authors:
            toks = re.findall(r"[A-Za-zÀ-ÖØ-öø-ÿ\-']+", a)
            if toks:
                lastnames.append(toks[-1].lower())
        nid = _note_id(n)
        if not nid:
            continue
        out.append(
            {
                "id": nid,
                "title": title,
                "title_norm": _normalize_title_for_join(title),
                "author_lastnames": sorted(set(lastnames)),
            }
        )
    return out

def _choose_best_candidate(
    pdf_title: str,
    pdf_author_lastnames: set[str],
    candidates: List[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    if not candidates:
        return None
    if len(candidates) == 1:
        return candidates[0]
    # tie-break by author overlap
    best = None
    best_score = -1
    for c in candidates:
        ov = len(pdf_author_lastnames.intersection(set(c.get("author_lastnames", []))))
        if ov > best_score:
            best_score = ov
            best = c
    return best or candidates[0]

def _fuzzy_match(pdf_title_norm: str, all_notes: List[Dict[str, Any]], min_ratio: float) -> Optional[Dict[str, Any]]:
    try:
        from rapidfuzz.fuzz import ratio as rf_ratio  # type: ignore
        best = None
        best_score = 0
        for n in all_notes:
            r = rf_ratio(pdf_title_norm, n["title_norm"]) / 100.0
            if r > best_score:
                best_score = r
                best = n
        if best is None or best_score < min_ratio:
            return None
        return best
    except Exception:
        best = None
        best_ratio = 0.0
        for n in all_notes:
            r = difflib.SequenceMatcher(None, pdf_title_norm, n["title_norm"]).ratio()
            if r > best_ratio:
                best_ratio = r
                best = n
        if best is None or best_ratio < min_ratio:
            return None
        return best

def download_openreview_year_index(
    year: int,
    cache_dir: str,
    refresh: bool,
    min_fuzzy: float,
    verbose: bool = False,
) -> List[Dict[str, Any]]:
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)
    cache_file = cache_path / f"openreview_{year}.json"
    if cache_file.exists() and not refresh:
        return json.loads(cache_file.read_text(encoding="utf-8") or "[]")

    try:
        import openreview  # type: ignore
    except ImportError as e:
        raise SystemExit("Missing dependency for OpenReview URLs: openreview-py. Install with: pip install openreview-py") from e

    baseurl_api2 = os.environ.get("OPENREVIEW_BASEURL", "https://api2.openreview.net")
    baseurl_api1 = os.environ.get("OPENREVIEW_BASEURL_API1", "https://api.openreview.net")
    username = os.environ.get("OPENREVIEW_USERNAME")
    password = os.environ.get("OPENREVIEW_PASSWORD")

    def mk_client_api2() -> Any:
        if username and password:
            return openreview.api.OpenReviewClient(baseurl=baseurl_api2, username=username, password=password)
        return openreview.api.OpenReviewClient(baseurl=baseurl_api2)

    def mk_client_api1() -> Any:
        if username and password:
            return openreview.Client(baseurl=baseurl_api1, username=username, password=password)
        return openreview.Client(baseurl=baseurl_api1)

    def try_download(client: Any, label: str) -> Tuple[List[Dict[str, Any]], Optional[str]]:
        invitation = _find_working_submission_invitation(client, year)
        if not invitation:
            if verbose:
                print(f"[openreview] {year}: {label} no working submission invitation found by probing")
            return [], None
        try:
            notes = client.get_all_notes(invitation=invitation)
        except Exception as e:
            if verbose:
                print(f"[openreview] {year}: {label} invitation {invitation} -> error fetching notes: {repr(e)}")
            return [], invitation
        idx = _build_year_index_from_notes(notes)
        if verbose:
            total = len(notes) if hasattr(notes, "__len__") else len(idx)
            print(f"[openreview] {year}: {label} invitation {invitation} -> {total} notes fetched, {len(idx)} with titles")
        return idx, invitation

    api2_ok: List[Dict[str, Any]] = []
    api1_ok: List[Dict[str, Any]] = []

    try:
        api2_ok, _ = try_download(mk_client_api2(), "api2")
    except Exception as e:
        if verbose or os.environ.get("OPENREVIEW_DEBUG_EXCEPTIONS"):
            print(f"[openreview] {year}: api2 exception during download/probe: {repr(e)}")
        api2_ok = []

    try:
        api1_ok, _ = try_download(mk_client_api1(), "api1")
    except Exception as e:
        if verbose or os.environ.get("OPENREVIEW_DEBUG_EXCEPTIONS"):
            print(f"[openreview] {year}: api1 exception during download/probe: {repr(e)}")
        api1_ok = []

    chosen: List[Dict[str, Any]] = []
    used = None
    if api2_ok and (len(api2_ok) >= max(50, int(0.6 * (len(api1_ok) or 0)))):
        chosen = api2_ok
        used = f"api2({baseurl_api2})"
    elif api1_ok:
        chosen = api1_ok
        used = f"api1({baseurl_api1})"
    elif api2_ok:
        chosen = api2_ok
        used = f"api2({baseurl_api2})"

    if not chosen:
        raise RuntimeError(f"OpenReview download failed for year {year} using both API2 ({baseurl_api2}) and API1 ({baseurl_api1}).")

    cache_file.write_text(json.dumps(chosen, ensure_ascii=False), encoding="utf-8")
    if verbose:
        print(f"[openreview] {year}: selected {used} ({len(chosen)} notes)")
    return chosen

def add_openreview_urls(
    bib_entries: List[Dict[str, str]],
    cache_dir: str,
    refresh: bool,
    min_fuzzy: float,
    verbose: bool,
) -> None:
    years = sorted({int(e["year"]) for e in bib_entries})
    year_index: Dict[int, List[Dict[str, Any]]] = {}
    year_title_map: Dict[int, Dict[str, List[Dict[str, Any]]]] = {}

    for y in years:
        idx = download_openreview_year_index(y, cache_dir=cache_dir, refresh=refresh, min_fuzzy=min_fuzzy, verbose=verbose)
        year_index[y] = idx
        tm: Dict[str, List[Dict[str, Any]]] = collections.defaultdict(list)
        for n in idx:
            tm[n["title_norm"]].append(n)
        year_title_map[y] = tm

    matched = 0
    for e in bib_entries:
        y = int(e["year"])
        pdf_title_norm = _normalize_title_for_join(e["__match_title"])
        pdf_lastnames = set()
        for a in e["author"].split(" and "):
            toks = re.findall(r"[A-Za-zÀ-ÖØ-öø-ÿ\-']+", a)
            if toks:
                pdf_lastnames.add(toks[-1].lower())

        candidates = year_title_map[y].get(pdf_title_norm, [])
        best = _choose_best_candidate(e["__match_title"], pdf_lastnames, candidates)
        if best is None:
            best = _fuzzy_match(pdf_title_norm, year_index[y], min_ratio=min_fuzzy)

        if best is not None:
            e["url"] = f"https://openreview.net/forum?id={best['id']}"
            matched += 1

    if verbose:
        print(f"[openreview] matched {matched}/{len(bib_entries)} entries with URLs")

def load_isbn_map(path: str) -> Dict[str, str]:
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    if isinstance(data, dict):
        return {str(k): str(v) for k, v in data.items() if v is not None}
    return {}

def add_isbn_fields(bib_entries: List[Dict[str, str]], isbn_map: Dict[str, str]) -> None:
    for e in bib_entries:
        isbn = isbn_map.get(str(e.get("year", "")))
        if isbn:
            e["isbn"] = isbn


# ----------------------------
# Bib writing
# ----------------------------

def write_bib(entries: List[Dict[str, str]], output_bib: str) -> None:
    key_counts: collections.Counter[str] = collections.Counter()

    with open(output_bib, "w", encoding="utf-8") as f:
        for e in entries:
            key_base = make_key(e["author"], int(e["year"]), e["title"])
            key = key_base
            if key_counts[key_base]:
                key = f"{key_base}{chr(ord('a') + key_counts[key_base])}"
            key_counts[key_base] += 1

            lines: List[str] = [f"@inproceedings{{{key},"]

            for field in ("title", "author", "booktitle", "year", "pages", "isbn", "url"):
                val = e.get(field, "")
                if not val:
                    continue
                lines.append(f"  {field} = {{{bib_escape(val)}}},")

            lines[-1] = lines[-1].rstrip(",")
            lines.append("}\n")
            f.write("\n".join(lines))


# ----------------------------
# Main
# ----------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("input", help="Zip file containing PDFs or a folder with PDFs")
    ap.add_argument("output_bib", help="Output .bib path")
    ap.add_argument("--verbose", action="store_true", help="Print per-file counts and OpenReview stats")
    ap.add_argument("--add-openreview-urls", action="store_true", help="Join against OpenReview and add url fields")
    ap.add_argument("--openreview-cache-dir", default="openreview_cache", help="Cache directory for OpenReview JSON")
    ap.add_argument("--refresh-openreview-cache", action="store_true", help="Ignore cache and re-download OpenReview per year")
    ap.add_argument("--openreview-min-fuzzy", type=float, default=0.92, help="Fuzzy match threshold (0..1)")
    ap.add_argument("--force-ipv4", action="store_true", help="Force IPv4 only (workaround for broken IPv6)")
    ap.add_argument("--isbn-map", default="data/iclr_isbn_by_year.json", help="Path to JSON map of year->ISBN")
    args = ap.parse_args()

    if args.force_ipv4:
        force_ipv4_only()

    pdf_paths = iter_pdf_paths(args.input)
    if not pdf_paths:
        raise SystemExit("No PDFs found in input.")

    bib_entries: List[Dict[str, str]] = []

    for pdf_path in pdf_paths:
        year = year_from_filename(pdf_path)
        if year is None:
            continue
        fmt = detect_format(pdf_path)
        raw = parse_old(pdf_path) if fmt == "old" else parse_new(pdf_path)
        raw = compute_page_ranges(raw)

        for r in raw:
            match_title = r.title.strip()
            title = protect_acronyms_for_biblatex(smart_title_case(match_title))
            author = normalize_author_list(r.authors_raw)
            booktitle = f"{ordinal(iclr_edition(year))} International Conference on Learning Representations ({{ICLR}} {year})"
            bib_entries.append(
                {
                    "__match_title": match_title,  # used only for OpenReview join
                    "title": title,
                    "author": author,
                    "booktitle": booktitle,
                    "year": str(year),
                    "pages": r.pages,
                }
            )

        if args.verbose:
            print(f"{os.path.basename(pdf_path)}  [{fmt}]  {len(raw)} entries")

    if args.add_openreview_urls:
        add_openreview_urls(
            bib_entries,
            cache_dir=args.openreview_cache_dir,
            refresh=args.refresh_openreview_cache,
            min_fuzzy=args.openreview_min_fuzzy,
            verbose=args.verbose,
        )
    isbn_map = load_isbn_map(args.isbn_map)
    if isbn_map:
        add_isbn_fields(bib_entries, isbn_map)

    bib_entries.sort(key=lambda e: (int(e["year"]), e.get("pages", ""), e.get("title", "")))
    # Drop internal helper field
    for e in bib_entries:
        e.pop("__match_title", None)

    write_bib(bib_entries, args.output_bib)

    if args.verbose:
        print(f"Wrote {len(bib_entries)} entries to {args.output_bib}")

if __name__ == "__main__":
    main()
