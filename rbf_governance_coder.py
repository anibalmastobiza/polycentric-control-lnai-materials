#!/usr/bin/env python3
"""Semi-automated coding assistant for AI governance documents.

This script operationalizes the four dimensions used in Sect. 3.2 of the paper:
decisional plurality, citizen contestability, runtime oversight, and public participation.

It is intentionally modest. The procedure is not a trained classifier and should not be
presented as a replacement for interpretive coding. It is an RBF-prototype scorer that
helps surface evidence passages and assign a provisional 0/1/2 label for each dimension.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Iterable

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


def load_extractor():
    extractor_path = (
        Path(__file__).resolve().parents[1]
        / "skills"
        / "ai-paper-scientist"
        / "scripts"
        / "extract_paper_text.py"
    )
    spec = importlib.util.spec_from_file_location("extract_paper_text", extractor_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load extractor from {extractor_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def tokenize(text: str) -> list[str]:
    tokens = re.findall(r"[a-z0-9][a-z0-9-]{1,}", text.lower())
    return [token for token in tokens if token not in STOPWORDS and not token.isdigit()]


def segment_text(text: str) -> list[str]:
    raw_paragraphs = [chunk.strip() for chunk in re.split(r"\n\s*\n+", text) if chunk.strip()]
    segments: list[str] = []
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
    doc_freq: Counter[str] = Counter()
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


def code_document(text: str, gamma: float) -> dict[str, object]:
    segments = segment_text(text)
    prototype_texts = [proto for cfg in DIMENSIONS.values() for proto in cfg["prototypes"]]
    tokenized_segments = [tokenize(segment) for segment in segments]
    tokenized_prototypes = [tokenize(proto) for proto in prototype_texts]
    idf = compute_idf(tokenized_segments + tokenized_prototypes)
    segment_vectors = [tfidf_vector(tokens, idf) for tokens in tokenized_segments]

    results: dict[str, object] = {}
    full_text = " ".join(segments)

    for dim_key, cfg in DIMENSIONS.items():
        prototype_vectors = []
        for proto in cfg["prototypes"]:
            prototype_vectors.append(tfidf_vector(tokenize(proto), idf))

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


def extract_text(path: Path) -> str:
    extractor = load_extractor()
    data = extractor.extract_content(path)
    return data["text"]


def render_table(results: dict[str, dict[str, object]]) -> str:
    columns = [
        ("decisional_plurality", "DP"),
        ("citizen_contestability", "CC"),
        ("runtime_oversight", "RO"),
        ("public_participation", "PP"),
    ]
    header = ["document", *[short for _, short in columns]]
    lines = [" | ".join(header)]
    lines.append(" | ".join(["---"] * len(header)))
    for doc_name, doc_result in results.items():
        row = [doc_name]
        for dim_key, _ in columns:
            row.append(str(doc_result[dim_key]["score"]))
        lines.append(" | ".join(row))
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Semi-automated RBF-prototype coding for AI governance documents."
    )
    parser.add_argument("documents", nargs="+", help="Paths to governance documents or extracted text files")
    parser.add_argument(
        "--gamma",
        type=float,
        default=1.5,
        help="RBF gamma parameter. Higher values make matching stricter.",
    )
    parser.add_argument(
        "--format",
        choices=("json", "table"),
        default="json",
        help="Output format",
    )
    args = parser.parse_args()

    all_results: dict[str, dict[str, object]] = {}
    for raw_path in args.documents:
        path = Path(raw_path).expanduser()
        if not path.exists():
            raise SystemExit(f"Document not found: {path}")
        text = extract_text(path)
        all_results[path.name] = code_document(text, args.gamma)

    if args.format == "table":
        print(render_table(all_results))
    else:
        print(json.dumps(all_results, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
