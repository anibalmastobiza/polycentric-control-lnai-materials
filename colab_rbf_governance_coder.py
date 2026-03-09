"""Colab-ready semi-automated governance coder.

Copy this file's contents into a Google Colab notebook, or run it as a script in a
Python environment that has access to uploaded documents.
"""

# Cell 1: optional dependencies for PDF support
# !pip -q install pypdf

# Cell 2: imports and coding logic
import json
import math
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Iterable
from zipfile import ZipFile
from xml.etree import ElementTree as ET

try:
    from google.colab import files  # type: ignore
except Exception:  # pragma: no cover
    files = None


STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "being", "by", "for", "from",
    "has", "have", "if", "in", "into", "is", "it", "its", "of", "on", "or", "that",
    "the", "their", "there", "these", "this", "to", "was", "were", "which", "with",
}

DIMENSIONS = {
    "decisional_plurality": {
        "label": "Decisional plurality",
        "prototypes": [
            "Authority is distributed across multiple independent institutions, regulators, boards, and oversight bodies.",
            "Decision-making power is shared across agencies, auditors, organizations, and review bodies rather than centralized in one actor.",
        ],
        "cue_terms": {
            "multiple", "distributed", "shared", "independent", "plural", "authorities",
            "agencies", "regulators", "board", "committee", "auditor", "supervisory",
            "oversight body", "decision center",
        },
        "action_terms": {
            "delegate", "share", "assign", "review", "approve", "authorize", "coordinate",
            "supervise", "veto",
        },
    },
    "citizen_contestability": {
        "label": "Citizen contestability",
        "prototypes": [
            "Affected persons can challenge, appeal, request review, or contest AI-mediated decisions.",
            "Citizens and impacted groups have intelligible channels for redress, complaint, or revision.",
        ],
        "cue_terms": {
            "appeal", "challenge", "contest", "complaint", "redress", "review", "remedy",
            "reconsideration", "grievance", "affected persons", "right to explanation",
        },
        "action_terms": {
            "request", "challenge", "appeal", "review", "reverse", "correct", "escalate",
            "complain",
        },
    },
    "runtime_oversight": {
        "label": "Runtime oversight",
        "prototypes": [
            "The system is monitored during deployment with logging, intervention, suspension, and ongoing supervision.",
            "Oversight continues at runtime through monitoring, incident response, auditing, and the power to stop or modify the system.",
        ],
        "cue_terms": {
            "runtime", "monitoring", "post-deployment", "logging", "incident", "intervention",
            "shutdown", "suspend", "real-time", "continuous", "audit trail", "supervision",
        },
        "action_terms": {
            "monitor", "log", "suspend", "stop", "override", "intervene", "audit", "track",
            "respond",
        },
    },
    "public_participation": {
        "label": "Public participation",
        "prototypes": [
            "Citizens, civil society, or affected communities participate directly in governance and deliberation.",
            "Public consultation, deliberative forums, and civil society inclusion are institutionalized in the governance process.",
        ],
        "cue_terms": {
            "public participation", "citizens", "civil society", "consultation", "deliberation",
            "stakeholder forum", "mini-public", "community", "participatory", "public input",
        },
        "action_terms": {
            "consult", "deliberate", "participate", "engage", "include", "represent", "co-design",
        },
    },
}

WORD_NS = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}


def normalize_whitespace(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def tokenize(text: str) -> list[str]:
    tokens = re.findall(r"[a-z0-9][a-z0-9-]{1,}", text.lower())
    return [token for token in tokens if token not in STOPWORDS and not token.isdigit()]


def segment_text(text: str) -> list[str]:
    raw_paragraphs = [chunk.strip() for chunk in re.split(r"\n\s*\n+", text) if chunk.strip()]
    segments = []
    for paragraph in raw_paragraphs:
        if len(paragraph) <= 500:
            segments.append(paragraph)
            continue
        parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", paragraph)
        for part in parts:
            cleaned = part.strip()
            if len(cleaned) >= 40:
                segments.append(cleaned)
    return segments or [text.strip()]


def compute_idf(tokenized_texts: Iterable[list[str]]) -> dict[str, float]:
    docs = list(tokenized_texts)
    doc_count = len(docs)
    doc_freq = Counter()
    for tokens in docs:
        doc_freq.update(set(tokens))
    return {
        term: math.log((1 + doc_count) / (1 + freq)) + 1.0
        for term, freq in doc_freq.items()
    }


def tfidf_vector(tokens: list[str], idf: dict[str, float]) -> dict[str, float]:
    counts = Counter(tokens)
    total = sum(counts.values())
    if total == 0:
        return {}
    vec = {term: (count / total) * idf[term] for term, count in counts.items() if term in idf}
    norm = math.sqrt(sum(value * value for value in vec.values()))
    if norm == 0.0:
        return {}
    return {term: value / norm for term, value in vec.items()}


def sparse_dot(left: dict[str, float], right: dict[str, float]) -> float:
    if len(left) > len(right):
        left, right = right, left
    return sum(value * right.get(term, 0.0) for term, value in left.items())


def rbf_similarity(left: dict[str, float], right: dict[str, float], gamma: float) -> float:
    if not left or not right:
        return 0.0
    dot = sparse_dot(left, right)
    squared_distance = max(0.0, 2.0 - 2.0 * dot)
    return math.exp(-gamma * squared_distance)


def phrase_hits(text: str, phrases: Iterable[str]) -> int:
    lowered = text.lower()
    return sum(1 for phrase in phrases if phrase.lower() in lowered)


def provisional_score(max_score: float, coverage: int, cue_hits: int, action_hits: int) -> int:
    if max_score >= 0.45 and (coverage >= 2 or cue_hits >= 3) and action_hits >= 1:
        return 2
    if max_score >= 0.28 or cue_hits >= 2:
        return 1
    return 0


def extract_docx_like_text(path: Path, file_type: str) -> str:
    with ZipFile(path) as archive:
        root = ET.fromstring(archive.read("word/document.xml"))
        body = root.find("./w:body", WORD_NS)
        if body is None:
            raise RuntimeError(f"{file_type} file has no document body")

        blocks = []
        for child in list(body):
            tag = child.tag.rsplit("}", 1)[-1]
            if tag == "p":
                text = "".join(node.text for node in child.iterfind(".//w:t", WORD_NS) if node.text)
                text = normalize_whitespace(text)
                if text:
                    blocks.append(text)
            elif tag == "tbl":
                for row in child.findall("./w:tr", WORD_NS):
                    cells = []
                    for cell in row.findall("./w:tc", WORD_NS):
                        text = "".join(node.text for node in cell.iterfind(".//w:t", WORD_NS) if node.text)
                        text = normalize_whitespace(text)
                        if text:
                            cells.append(text)
                    if cells:
                        blocks.append(" | ".join(cells))
        return "\n\n".join(blocks)


def extract_pdf_text(path: Path) -> str:
    try:
        from pypdf import PdfReader
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "pypdf"])
        from pypdf import PdfReader

    reader = PdfReader(str(path))
    pages = []
    for page in reader.pages:
        text = normalize_whitespace(page.extract_text() or "")
        if text:
            pages.append(text)
    return "\n\n".join(pages)


def extract_text(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in {".txt", ".md"}:
        return normalize_whitespace(path.read_text(encoding="utf-8"))
    if suffix in {".docx", ".docm"}:
        return extract_docx_like_text(path, suffix)
    if suffix == ".pdf":
        return extract_pdf_text(path)
    raise RuntimeError(f"Unsupported file type: {suffix}")


def code_text(text: str, gamma: float = 1.5) -> dict:
    segments = segment_text(text)
    prototype_texts = [proto for cfg in DIMENSIONS.values() for proto in cfg["prototypes"]]
    tokenized_segments = [tokenize(segment) for segment in segments]
    tokenized_prototypes = [tokenize(proto) for proto in prototype_texts]
    idf = compute_idf(tokenized_segments + tokenized_prototypes)
    segment_vectors = [tfidf_vector(tokens, idf) for tokens in tokenized_segments]
    full_text = " ".join(segments)

    results = {}
    for dim_key, cfg in DIMENSIONS.items():
        prototype_vectors = [tfidf_vector(tokenize(proto), idf) for proto in cfg["prototypes"]]
        scored_segments = []
        for segment, vector in zip(segments, segment_vectors):
            score = max(rbf_similarity(vector, proto_vec, gamma) for proto_vec in prototype_vectors)
            scored_segments.append((score, segment))

        scored_segments.sort(key=lambda item: item[0], reverse=True)
        max_score = scored_segments[0][0] if scored_segments else 0.0
        coverage = sum(1 for score, _ in scored_segments if score >= 0.28)
        top_text = " ".join(segment for _, segment in scored_segments[:3])
        cue_hits = phrase_hits(full_text, cfg["cue_terms"])
        action_hits = phrase_hits(top_text, cfg["action_terms"])
        score = provisional_score(max_score, coverage, cue_hits, action_hits)

        results[dim_key] = {
            "label": cfg["label"],
            "score": score,
            "max_similarity": round(max_score, 4),
            "coverage": coverage,
            "cue_hits": cue_hits,
            "action_hits_top_segments": action_hits,
            "top_evidence": [
                {"similarity": round(seg_score, 4), "text": segment}
                for seg_score, segment in scored_segments[:3]
            ],
        }
    return results


def print_results(results: dict) -> None:
    for dim_key, info in results.items():
        print(f"\n{info['label']}: {info['score']}")
        print(f"max_similarity={info['max_similarity']} | coverage={info['coverage']} | cue_hits={info['cue_hits']}")
        print("Top evidence:")
        for item in info["top_evidence"]:
            print(f"- ({item['similarity']}) {item['text'][:280]}")


# Cell 3: upload one file from your computer
if files is None:
    raise RuntimeError("This block is intended for Google Colab.")

uploaded = files.upload()
filename = next(iter(uploaded))
path = Path(filename)
text = extract_text(path)

print(f"Loaded: {filename}")
print(f"Characters: {len(text)}")
print()
print(text[:2000])


# Cell 4: run automated coding
results = code_text(text, gamma=1.5)
print_results(results)


# Cell 5: optional JSON export
with open("coding_results.json", "w", encoding="utf-8") as handle:
    json.dump(results, handle, indent=2, ensure_ascii=False)

files.download("coding_results.json")
