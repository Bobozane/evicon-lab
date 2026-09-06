"""Deterministic, local-only WVS English source extraction drafts.

This module reads only the approved English questionnaire PDF.  It does not
translate, call a model, infer missing labels, or create a ProbeSet.  The
output is an explicitly unverified local draft under ``private_wvs_transcriptions``.
"""

from __future__ import annotations

import json
import re
import tomllib
import zlib
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Iterable

from pydantic import ValidationError

from .wvs7_authoring import (
    WVS7AuthoringError,
    WVS7AuthoringPlan,
    WVS7AuthoringPlanItem,
    WVS7LanguageArm,
    WVS7ManualResponseOption,
    WVS7ManualTranscriptionSet,
)


USA_QUESTIONNAIRE = "F00008646-WVS7_Questionnaire_USA_2017_English.pdf"
SOURCE_DRAFT_ORIGIN = "local_pdf_text_extraction_draft"
SOURCE_TRANSCRIBER_ID = "codex_source_extraction_draft"
EXPECTED_VARIABLE_IDS = (
    "Q106", "Q107", "Q108", "Q109", "Q110", "Q111", "Q149", "Q150",
    "Q158", "Q159", "Q160", "Q161", "Q162", "Q163", "Q196", "Q197",
    "Q198", "Q241", "Q243", "Q246", "Q247", "Q248", "Q249",
)


class WVS7SourceTranscriptionError(ValueError):
    """Stable error for local source extraction and draft serialization."""


@dataclass(frozen=True)
class _TextFragment:
    x: float
    y: float
    text: str


@dataclass(frozen=True)
class PdfPageText:
    page_number: int
    lines: tuple[str, ...]


class DeterministicPdfTextExtractor:
    """Small stdlib PDF text extractor for the local FlateDecode questionnaire.

    ``pypdf``/``pdfplumber`` are used when present; the fallback handles the
    simple PDF text operators used by the checked-in questionnaire and keeps
    the project runnable in an offline minimal environment.
    """

    def __init__(self, pdf_path: str | Path):
        self.pdf_path = Path(pdf_path)

    def extract_pages(self, page_numbers: Iterable[int]) -> dict[int, PdfPageText]:
        requested = sorted(set(int(page) for page in page_numbers))
        if not requested or any(page < 1 for page in requested):
            raise WVS7SourceTranscriptionError("questionnaire page list is invalid")
        if not self.pdf_path.is_file():
            raise WVS7SourceTranscriptionError("English questionnaire PDF is missing")
        try:
            import pypdf  # type: ignore[import-not-found]
        except ImportError:
            return self._extract_stdlib(requested)
        try:
            reader = pypdf.PdfReader(str(self.pdf_path))
            if max(requested) > len(reader.pages):
                raise WVS7SourceTranscriptionError("questionnaire page is outside the PDF")
            return {
                page: PdfPageText(page, tuple(_normalise_lines(reader.pages[page - 1].extract_text() or "")))
                for page in requested
            }
        except WVS7SourceTranscriptionError:
            raise
        except Exception as exc:  # pragma: no cover - depends on optional parser
            raise WVS7SourceTranscriptionError("local English questionnaire text extraction failed") from exc

    def _extract_stdlib(self, requested: list[int]) -> dict[int, PdfPageText]:
        raw = self.pdf_path.read_bytes()
        objects = {
            int(number): body
            for number, body in re.findall(rb"(?m)(\d+)\s+0\s+obj\s*(.*?)\s*endobj", raw, re.S)
        }
        pages_object = next((body for body in objects.values() if b"/Type/Pages" in body), None)
        if pages_object is None:
            raise WVS7SourceTranscriptionError("English questionnaire PDF has no page tree")
        match = re.search(rb"/Kids\s*\[(.*?)\]", pages_object, re.S)
        if match is None:
            raise WVS7SourceTranscriptionError("English questionnaire PDF page tree is malformed")
        page_ids = [int(value) for value in re.findall(rb"(\d+)\s+0\s+R", match.group(1))]
        if max(requested) > len(page_ids):
            raise WVS7SourceTranscriptionError("questionnaire page is outside the PDF")
        result: dict[int, PdfPageText] = {}
        for page in requested:
            page_body = objects.get(page_ids[page - 1])
            if page_body is None:
                raise WVS7SourceTranscriptionError("questionnaire page object is missing")
            contents = re.search(rb"/Contents\s*(\d+)\s+0\s+R", page_body)
            if contents is None or int(contents.group(1)) not in objects:
                raise WVS7SourceTranscriptionError("questionnaire page text stream is missing")
            stream_match = re.search(rb"stream\s*\r?\n(.*?)\r?\nendstream", objects[int(contents.group(1))], re.S)
            if stream_match is None:
                raise WVS7SourceTranscriptionError("questionnaire page text stream is malformed")
            try:
                stream = zlib.decompress(stream_match.group(1))
            except zlib.error as exc:
                raise WVS7SourceTranscriptionError("questionnaire page compression is unsupported") from exc
            result[page] = PdfPageText(page, tuple(_lines_from_pdf_stream(stream)))
        return result


class WVS7SourceTranscriber:
    """Convenience facade for one approved local English questionnaire PDF."""

    def __init__(self, pdf_path: str | Path):
        self.pdf_path = Path(pdf_path)

    def extract(self, plan: WVS7AuthoringPlan) -> WVS7ManualTranscriptionSet:
        return extract_english_core_source_draft(plan, self.pdf_path)

    def write(
        self,
        plan: WVS7AuthoringPlan,
        output_path: str | Path,
        *,
        overwrite: bool = False,
    ) -> WVS7ManualTranscriptionSet:
        return write_english_core_source_draft(plan, self.pdf_path, output_path, overwrite=overwrite)


def extract_english_core_source_draft(
    plan: WVS7AuthoringPlan,
    pdf_path: str | Path,
) -> WVS7ManualTranscriptionSet:
    """Extract exactly the accepted English 23-item plan into an unverified set."""
    _validate_plan(plan)
    pages = sorted({page for item in plan.items for page in item.source_questionnaire_pages})
    pages_text = DeterministicPdfTextExtractor(pdf_path).extract_pages(pages)
    rows = [_extract_item(item, pages_text, plan.source_review_id) for item in plan.items]
    try:
        return WVS7ManualTranscriptionSet(
            language=WVS7LanguageArm.ENGLISH_CORE,
            source_review_id=plan.source_review_id,
            synthetic_fixture=False,
            items=rows,
        )
    except ValidationError as exc:
        raise WVS7SourceTranscriptionError("extracted draft violates transcription contract") from exc


def write_english_core_source_draft(
    plan: WVS7AuthoringPlan,
    pdf_path: str | Path,
    output_path: str | Path,
    *,
    overwrite: bool = False,
) -> WVS7ManualTranscriptionSet:
    """Write only to the ignored private transcription directory."""
    destination = Path(output_path)
    if "private_wvs_transcriptions" not in destination.parts:
        raise WVS7SourceTranscriptionError("source draft output must be under private_wvs_transcriptions")
    if destination.exists() and not overwrite:
        raise WVS7SourceTranscriptionError("source draft output already exists")
    draft = extract_english_core_source_draft(plan, pdf_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(_draft_toml(draft), encoding="utf-8")
    return draft


def safe_source_draft_summary(draft: WVS7ManualTranscriptionSet) -> dict[str, object]:
    """Return an audit-safe extraction summary without source text or labels."""
    return {
        "status": "blocked",
        "reason": "human_verification_required",
        "language": draft.language.value,
        "extracted_count": len(draft.items),
        "variable_ids": [item.variable_id for item in draft.items],
        "items": [
            {
                "variable_id": item.variable_id,
                "status": "extracted_unverified",
                "pages": item.source_questionnaire_pages,
                "warning_count": len(item.warnings),
                "warnings": item.warnings,
                "missing_fields": _missing_fields(item),
            }
            for item in draft.items
        ],
    }


def _validate_plan(plan: WVS7AuthoringPlan) -> None:
    if plan.language_arm is not WVS7LanguageArm.ENGLISH_CORE:
        raise WVS7SourceTranscriptionError("source extraction accepts only the English core plan")
    ids = tuple(item.variable_id for item in plan.items)
    if ids != EXPECTED_VARIABLE_IDS or "Q48" in ids or len(set(ids)) != 23:
        raise WVS7SourceTranscriptionError("source extraction requires exactly the 23 non-Q48 variables")
    if any(item.source_questionnaire_file != USA_QUESTIONNAIRE for item in plan.items):
        raise WVS7SourceTranscriptionError("source extraction requires the approved USA English questionnaire")
    for item in plan.items:
        if any(page < 1 for page in item.source_questionnaire_pages):
            raise WVS7SourceTranscriptionError("authoring plan contains an invalid questionnaire page")


def _extract_item(item: WVS7AuthoringPlanItem, pages: dict[int, PdfPageText], source_review_id: str) -> object:
    lines = [_canonicalise_q_tokens(line) for page in item.source_questionnaire_pages for line in pages[page].lines]
    marker = re.compile(rf"\bQ\s*{re.escape(item.variable_id[1:])}\b", re.I)
    start = next((index for index, line in enumerate(lines) if marker.search(line)), None)
    warnings: list[str] = []
    if start is None:
        warnings.append("question_marker_not_found")
        block: list[str] = []
    else:
        block = []
        for line in lines[start:]:
            if block and re.search(r"\bQ\s*[0-9]{2,3}\b", line, re.I) and not marker.search(line):
                break
            block.append(line)
    question = _question_from_block(item.variable_id, block)
    options = _options_from_block(item, block, warnings)
    endpoints = _endpoints_from_block(item, block)
    if not question:
        warnings.append("question_text_missing_or_order_uncertain")
        question = "TODO_MANUAL_ENTRY"
    if not options:
        warnings.append("response_options_missing_or_order_uncertain")
    if set(endpoints) != {str(item.scale_min), str(item.scale_max)}:
        warnings.append("endpoint_labels_unresolved")
        endpoints = {}
    if item.variable_id in {"Q149", "Q150"}:
        special = {**item.special_response_codes, "_scoring_mode": "binary_categorical_not_continuous"}
    else:
        special = dict(item.special_response_codes)
    if item.variable_id == "Q111":
        special["3"] = "volunteered_other_exclude_from_ordinary_ordinal_scoring"
    if item.variable_id in {"Q241", "Q243", "Q246", "Q247", "Q248", "Q249"}:
        special["0"] = "volunteered_against_democracy_exclude_from_ordinary_1_to_10_distance"
    if item.variable_id in {"Q158", "Q159", "Q160", "Q161", "Q162", "Q163"}:
        special["_analysis_note"] = "secondary_science_technology_attitude_analysis_not_pure_value_factor"
    return {
        "variable_id": item.variable_id,
        "language": WVS7LanguageArm.ENGLISH_CORE,
        "source_review_id": source_review_id,
        "source_questionnaire_file": item.source_questionnaire_file,
        "source_codebook_file": item.source_codebook_file,
        "source_questionnaire_pages": list(item.source_questionnaire_pages),
        "source_codebook_pages": list(item.source_codebook_pages),
        "source_sha256_prefixes": dict(item.source_sha256_prefixes),
        "question_text": question,
        "response_options": options,
        "scale_min": item.scale_min,
        "scale_max": item.scale_max,
        "high_score_meaning": item.high_score_meaning,
        "reverse_scored": item.reverse_scored,
        "endpoint_labels": endpoints,
        "special_response_codes": special,
        "manual_transcriber_id": SOURCE_TRANSCRIBER_ID,
        "transcribed_on": date.today().isoformat(),
        "transcription_origin": SOURCE_DRAFT_ORIGIN,
        "human_verified": False,
        "warnings": warnings,
    }


def _question_from_block(variable_id: str, block: list[str]) -> str:
    if not block:
        return ""
    first = re.sub(rf"^.*?Q\s*{variable_id[1:]}\.?\s*", "", block[0], flags=re.I).strip()
    text = [first] if first else []
    for line in block[1:]:
        if _is_option_line(line) or _looks_like_scale_row(line):
            break
        if "completely disagree" in line.lower() or "completely agree" in line.lower():
            break
        if line and not re.fullmatch(r"[-–—\s]+", line):
            text.append(line)
    return " ".join(_normalise_lines(" ".join(text))).strip()


def _options_from_block(item: WVS7AuthoringPlanItem, block: list[str], warnings: list[str]) -> list[dict[str, str]]:
    expected = item.response_option_codes
    if item.variable_id in {"Q149", "Q150"}:
        found: dict[str, str] = {}
        for line in block:
            match = re.match(r"\s*([12])\.\s+(.+)", line)
            if match:
                found.setdefault(match.group(1), match.group(2).strip())
        if len(found) == 2:
            return [{"code": code, "label": found[code]} for code in expected]
        warnings.append("structured_option_labels_not_fully_extracted")
    if item.variable_id == "Q111":
        found: dict[str, str] = {}
        for line in block:
            match = re.match(r"\s*([123])\s*[.)-]?\s+(.+)", line)
            if match and match.group(1) in expected:
                found.setdefault(match.group(1), match.group(2).strip())
        if len(found) == 3:
            return [{"code": code, "label": found[code]} for code in expected]
        warnings.append("structured_option_labels_not_fully_extracted")
    warnings.append("ordinal_option_labels_not_separated_in_pdf_text")
    return [{"code": code, "label": code} for code in expected]


def _endpoints_from_block(item: WVS7AuthoringPlanItem, block: list[str]) -> dict[str, str]:
    joined = " ".join(block)
    pairs = (
        ("Completely disagree", "Completely agree"),
        ("A lot worse off", "A lot better off"),
        ("Definitely should have the right", "Definitely should not have the right"),
    )
    for left, right in pairs:
        if left.lower() in joined.lower() and right.lower() in joined.lower():
            return {str(item.scale_min): left, str(item.scale_max): right}
    if item.variable_id in {"Q149", "Q150"}:
        options = {}
        for line in block:
            match = re.match(r"\s*([12])\.\s+(.+)", line)
            if match:
                options[match.group(1)] = match.group(2).strip()
        if set(options) == {"1", "2"}:
            return {"1": options["1"], "2": options["2"]}
    return {}


def _is_option_line(line: str) -> bool:
    return bool(re.match(r"^\s*(?:0|[1-9]|10)\s*[.)-]?\s+\S", line))


def _looks_like_scale_row(line: str) -> bool:
    numbers = re.findall(r"(?<!\d)(?:0|[1-9]|10)(?!\d)", line)
    return len(numbers) >= 3 and all(part.strip().isdigit() for part in numbers)


def _lines_from_pdf_stream(stream: bytes) -> list[str]:
    fragments: list[_TextFragment] = []
    for block in re.findall(rb"BT(.*?)ET", stream, re.S):
        tm = re.search(rb"1\s+0\s+0\s+1\s+(-?[0-9.]+)\s+(-?[0-9.]+)\s+Tm", block)
        if tm is None:
            continue
        x, y = float(tm.group(1)), float(tm.group(2))
        strings = _pdf_literal_strings(block)
        text = "".join(strings).strip()
        if text:
            fragments.append(_TextFragment(x, y, text))
    rows: list[tuple[float, list[_TextFragment]]] = []
    for fragment in sorted(fragments, key=lambda value: (-value.y, value.x)):
        if rows and abs(rows[-1][0] - fragment.y) < 0.8:
            rows[-1][1].append(fragment)
        else:
            rows.append((fragment.y, [fragment]))
    return _normalise_lines("\n".join(" ".join(fragment.text for fragment in sorted(row, key=lambda value: value.x)) for _, row in rows))


def _pdf_literal_strings(data: bytes) -> list[str]:
    result: list[str] = []
    index = 0
    while index < len(data):
        if data[index : index + 1] != b"(":
            index += 1
            continue
        index += 1
        depth = 1
        value = bytearray()
        while index < len(data) and depth:
            char = data[index]
            index += 1
            if char == 0x5C and index < len(data):
                escaped = data[index]
                index += 1
                escapes = {ord("n"): 10, ord("r"): 13, ord("t"): 9, ord("b"): 8, ord("f"): 12}
                if escaped in escapes:
                    value.append(escapes[escaped])
                elif escaped in (ord("("), ord(")"), ord("\\")):
                    value.append(escaped)
                elif 48 <= escaped <= 55:
                    octal = bytes([escaped])
                    while index < len(data) and len(octal) < 3 and 48 <= data[index] <= 55:
                        octal += bytes([data[index]])
                        index += 1
                    value.append(int(octal, 8))
                else:
                    value.append(escaped)
            elif char == 0x28:
                depth += 1
                value.append(char)
            elif char == 0x29:
                depth -= 1
                if depth:
                    value.append(char)
            else:
                value.append(char)
        try:
            result.append(value.decode("latin-1"))
        except UnicodeDecodeError:
            result.append("")
    return result


def _normalise_lines(value: str) -> list[str]:
    return [" ".join(line.split()) for line in value.splitlines() if line.strip()]


def _canonicalise_q_tokens(line: str) -> str:
    """Join PDF-spaced variable markers such as ``Q 1 49`` into ``Q149``."""
    def replace(match: re.Match[str]) -> str:
        return "Q" + "".join(match.group(0)[1:].split())

    return re.sub(r"Q(?:\s*[0-9]){2,3}", replace, line, flags=re.I)


def _missing_fields(item: object) -> list[str]:
    missing: list[str] = []
    for field, value in (
        ("question_text", getattr(item, "question_text", "")),
        ("response_options", getattr(item, "response_options", [])),
        ("endpoint_labels", getattr(item, "endpoint_labels", {})),
    ):
        if not value or value == "TODO_MANUAL_ENTRY":
            missing.append(field)
    return missing


def _draft_toml(draft: WVS7ManualTranscriptionSet) -> str:
    lines = [
        f"language = {_toml_string(draft.language.value)}",
        f"source_review_id = {_toml_string(draft.source_review_id)}",
        "synthetic_fixture = false",
        "",
    ]
    for item in draft.items:
        lines.extend([
            "[[items]]",
            f"variable_id = {_toml_string(item.variable_id)}",
            f"language = {_toml_string(item.language.value)}",
            f"source_review_id = {_toml_string(item.source_review_id)}",
            f"source_questionnaire_file = {_toml_string(item.source_questionnaire_file)}",
            f"source_codebook_file = {_toml_string(item.source_codebook_file)}",
            f"source_questionnaire_pages = {_toml_array(item.source_questionnaire_pages)}",
            f"source_codebook_pages = {_toml_array(item.source_codebook_pages)}",
            f"source_sha256_prefixes = {_toml_mapping(item.source_sha256_prefixes)}",
            f"question_text = {_toml_string(item.question_text)}",
            f"response_options = {_toml_options(item.response_options)}",
            f"scale_min = {item.scale_min}",
            f"scale_max = {item.scale_max}",
            f"high_score_meaning = {_toml_string(item.high_score_meaning or '')}",
            f"reverse_scored = {str(bool(item.reverse_scored)).lower()}",
            f"endpoint_labels = {_toml_mapping(item.endpoint_labels)}",
            f"special_response_codes = {_toml_mapping(item.special_response_codes)}",
            f"manual_transcriber_id = {_toml_string(item.manual_transcriber_id)}",
            f"transcribed_on = {_toml_string(item.transcribed_on.isoformat())}",
            f"transcription_origin = {_toml_string(item.transcription_origin or SOURCE_DRAFT_ORIGIN)}",
            "human_verified = false",
            f"warnings = {_toml_array_strings(item.warnings)}",
            "",
        ])
    return "\n".join(lines)


def _toml_string(value: str) -> str:
    return json.dumps(value, ensure_ascii=True)


def _toml_array(values: list[int]) -> str:
    return "[" + ", ".join(str(value) for value in values) + "]"


def _toml_array_strings(values: list[str]) -> str:
    return "[" + ", ".join(_toml_string(value) for value in values) + "]"


def _toml_mapping(values: dict[str, str]) -> str:
    return "{" + ", ".join(f"{_toml_string(key)} = {_toml_string(value)}" for key, value in sorted(values.items())) + "}"


def _toml_options(values: list[WVS7ManualResponseOption]) -> str:
    if not values:
        return "[]"
    return "[" + ", ".join("{" + f"code = {_toml_string(value.code)}, label = {_toml_string(value.label)}" + "}" for value in values) + "]"


__all__ = [
    "DeterministicPdfTextExtractor",
    "PdfPageText",
    "SOURCE_DRAFT_ORIGIN",
    "SOURCE_TRANSCRIBER_ID",
    "USA_QUESTIONNAIRE",
    "WVS7SourceTranscriptionError",
    "WVS7SourceTranscriber",
    "extract_english_core_source_draft",
    "safe_source_draft_summary",
    "write_english_core_source_draft",
]
