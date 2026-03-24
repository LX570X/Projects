"""
Purpose:
- Main orchestrator that scans source files, runs extraction/report pipelines,
  writes outputs, updates SQLite metadata, and uploads artifacts to MinIO.

Libraries used:
- stdlib (json/hashlib/mimetypes/pathlib/re/time/datetime): core processing helpers.
- repositories/*: SQLite metadata CRUD.
- pipelines/* + agents/*: extraction + report generation.
- services.storage/object_storage: local persistence + object storage upload.
"""

import json
import hashlib
import mimetypes
import os
import re
from datetime import datetime
from time import perf_counter
from pathlib import Path

from app.repositories.artifact_repository import ArtifactRepository
from app.repositories.file_repository import FileRepository
from app.repositories.job_repository import JobRepository
from app.repositories.run_repository import RunRepository
from app.pipelines.audio import extract_text_from_audio
from app.pipelines.documents import extract_text_from_document
from app.pipelines.images import extract_text_and_description_from_image
from app.pipelines.spreadsheet import extract_structured_data_from_spreadsheet
from app.pipelines.video import analyze_video
from app.agents.report_agent import generate_final_report, generate_single_report
from app.services.classifier import get_file_classification
from app.services.db import DB_PATH, init_db
from app.services.object_storage import ObjectStorageService
from app.services.storage import (
    SINGLE_REPORTS_DIR,
    SOURCE_FILES_DIR,
    ensure_base_directories,
    save_final_report_pdf,
    save_final_report_pdf_by_language,
    save_final_report,
    save_final_report_by_language,
    save_final_report_text,
    save_final_report_text_by_language,
    save_raw_data,
    save_single_report_pdf_by_language,
    save_single_report_by_language,
    save_single_report_text_by_language,
)


def _log(level: str, message: str) -> None:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] [{level}] {message}")


def _is_verbose_enabled() -> bool:
    return os.getenv("PROCESSOR_VERBOSE", "false").strip().lower() in {"1", "true", "yes", "on"}


def _log_verbose(message: str) -> None:
    if _is_verbose_enabled():
        _log("INFO", message)


def _extract_by_type(file_path: Path, file_type: str) -> dict:
    if file_type == "image":
        return extract_text_and_description_from_image(file_path)
    if file_type == "audio":
        return extract_text_from_audio(file_path)
    if file_type == "document":
        return extract_text_from_document(file_path)
    if file_type == "spreadsheet":
        return extract_structured_data_from_spreadsheet(file_path)
    if file_type == "video":
        return analyze_video(file_path)

    return {"message": "No extraction available for this file type."}


def _single_report_path_for_file(file_name: str) -> Path:
    return SINGLE_REPORTS_DIR / f"{file_name}.report.json"


def _single_report_path_for_file_by_language(file_name: str, language: str) -> Path:
    lang = (language or "en").strip().lower()
    return SINGLE_REPORTS_DIR / f"{file_name}.report.{lang}.json"


def _single_report_text_path_for_file_by_language(file_name: str, language: str) -> Path:
    lang = (language or "en").strip().lower()
    return SINGLE_REPORTS_DIR / f"{file_name}.report.{lang}.txt"


def _single_report_pdf_path_for_file_by_language(file_name: str, language: str) -> Path:
    lang = (language or "en").strip().lower()
    return SINGLE_REPORTS_DIR / f"{file_name}.report.{lang}.pdf"


def _safe_key_name(name: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9._-]+", "-", name).strip("-._")
    sanitized = re.sub(r"-+", "-", sanitized)
    return sanitized.lower() or "file"


def _sha256_file(file_path: Path) -> str:
    hasher = hashlib.sha256()
    with file_path.open("rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()


def _load_all_single_reports(language: str = "en") -> list[dict]:
    lang = (language or "en").strip().lower()
    pattern = f"*.report.{lang}.json"
    reports: list[dict] = []
    for path in SINGLE_REPORTS_DIR.glob(pattern):
        if not path.is_file():
            continue
        try:
            with open(path, "r", encoding="utf-8") as f:
                reports.append(json.load(f))
        except Exception:
            # Skip invalid/corrupted report files and continue.
            continue
    return reports


def _normalize_single_report_for_final(single_report: dict) -> dict:
    """
    Normalize legacy and new single-report JSON shapes to a unified structure
    expected by the final-report prompt.
    """
    if "executive_summary" in single_report or "report_text" in single_report:
        return single_report

    title = str(single_report.get("file_name", "Incident File Report")).strip() or "Incident File Report"
    summary = str(single_report.get("summary", "")).strip()
    people = single_report.get("people_involved", []) or []
    dates = single_report.get("dates_mentioned", []) or []
    details = single_report.get("important_details", []) or []
    risks = single_report.get("risk_flags", []) or []

    report_text = "\n".join(
        [
            f"Executive Summary\n{summary or 'No summary available.'}",
            "",
            "Key Findings",
            *([f"- {x}" for x in details] or ["- No key findings available."]),
            "",
            "Timeline",
            *([f"- {x}" for x in dates] or ["- No timeline details available."]),
            "",
            "People & Entities",
            *([f"- {x}" for x in people] or ["- No people/entities identified."]),
            "",
            "Risks",
            *([f"- {x}" for x in risks] or ["- No explicit risks identified."]),
            "",
            "Recommended Actions",
            "- Validate extracted findings against source evidence.",
            "- Prioritize risk mitigation and assign owners.",
        ]
    )

    return {
        "file_name": single_report.get("file_name", ""),
        "file_type": single_report.get("file_type", "unknown"),
        "title": title,
        "executive_summary": summary,
        "key_findings": details,
        "timeline": dates,
        "people_entities": people,
        "risks": risks,
        "recommended_actions": [
            "Validate extracted findings against source evidence.",
            "Prioritize risk mitigation and assign owners.",
        ],
        "confidence": "medium",
        "report_text": report_text,
    }


def _is_final_report_meaningful(report: dict) -> bool:
    summary_ok = bool(str(report.get("executive_summary", "")).strip())
    narrative_ok = bool(str(report.get("report_text", "")).strip())
    has_lists_content = any(
        bool(report.get(key))
        for key in [
            "key_people_entities",
            "key_timeline",
            "cross_file_insights",
            "recommended_actions",
        ]
    )
    return summary_ok and narrative_ok and has_lists_content


def _build_fallback_final_report(single_reports: list[dict]) -> dict:
    all_people: set[str] = set()
    all_dates: set[str] = set()
    all_details: list[str] = []
    all_risks: list[str] = []

    for r in single_reports:
        nr = _normalize_single_report_for_final(r)
        all_people.update(nr.get("people_entities", []) or [])
        all_dates.update(nr.get("timeline", []) or [])
        all_details.extend(nr.get("key_findings", []) or [])
        all_risks.extend(nr.get("risks", []) or [])

    insights = []
    if all_details:
        insights.append(f"Compiled {len(all_details)} important detail points across files.")
    if all_risks:
        insights.append(f"Identified {len(all_risks)} risk-related observations.")
    if all_people:
        insights.append("Multiple references to involved people/entities across documents.")

    next_steps = [
        "Review high-priority risk flags and assign owners.",
        "Validate key facts (dates, names, and claims) against source evidence.",
        "Create a consolidated incident timeline from extracted dates and events.",
    ]

    executive_summary = (
        f"Aggregated analysis from {len(single_reports)} single reports. "
        "This fallback summary was generated to avoid weak or empty final-report output."
    )

    report_text = "\n".join(
        [
            "Executive Summary",
            executive_summary,
            "",
            "Cross-File Insights",
            *([f"- {x}" for x in (insights or ["Cross-file synthesis completed with limited explicit overlap."])]),
            "",
            "People & Entities",
            *([f"- {x}" for x in sorted(all_people)] or ["- No people/entities identified."]),
            "",
            "Timeline",
            *([f"- {x}" for x in sorted(all_dates)] or ["- No timeline information identified."]),
            "",
            "Major Risks",
            *([f"- {x}" for x in all_risks] or ["- No explicit risk flags identified."]),
            "",
            "Recommended Actions",
            *([f"- {x}" for x in next_steps]),
        ]
    )

    return {
        "title": "Incident Analysis Final Report",
        "executive_summary": executive_summary,
        "cross_file_insights": insights or ["Cross-file synthesis completed with limited explicit overlap."],
        "key_people_entities": sorted(all_people),
        "key_timeline": sorted(all_dates),
        "major_risks": all_risks,
        "recommended_actions": next_steps,
        "confidence": "medium",
        "report_text": report_text,
    }


def _upload_to_minio_and_record(
    *,
    object_storage: ObjectStorageService,
    artifact_repo: ArtifactRepository,
    file_id: int | None,
    artifact_type: str,
    local_path: Path,
    object_key: str,
    content_type: str,
) -> int | None:
    if not object_storage.enabled:
        return None

    try:
        uploaded = object_storage.upload_file(
            local_path=local_path,
            object_key=object_key,
            content_type=content_type,
        )
        artifact = artifact_repo.create(
            file_id=file_id,
            artifact_type=artifact_type,
            storage_backend="minio",
            bucket=uploaded.get("bucket"),
            object_key=uploaded.get("object_key", object_key),
            etag=uploaded.get("etag"),
            content_type=uploaded.get("content_type"),
            size=uploaded.get("size"),
            version_id=uploaded.get("version_id"),
        )
        return artifact.id
    except Exception as exc:
        _log("WARN", f"MinIO upload skipped for {artifact_type} ({local_path.name}): {exc}")
        return None


def process_source_files() -> dict:
    """
    Process files from source-files/ and save outputs to raw-data/.
    No API upload flow. Files are read directly from disk.
    """
    run_start = perf_counter()
    _log("INFO", "START run: source file processor")

    ensure_base_directories()
    _log_verbose("Base directories verified")

    init_db()
    _log_verbose(f"SQLite metadata initialized at: {DB_PATH}")

    file_repo = FileRepository()
    job_repo = JobRepository()
    artifact_repo = ArtifactRepository()
    run_repo = RunRepository()
    object_storage = ObjectStorageService()

    if object_storage.enabled:
        _log("INFO", f"MinIO enabled: bucket='{object_storage.bucket}', endpoint='{object_storage.endpoint}'")
    elif getattr(object_storage, "init_error", None):
        _log("WARN", f"MinIO unavailable. Falling back to local-only mode: {object_storage.init_error}")
    else:
        _log("INFO", "MinIO disabled: using local filesystem as primary artifact store")

    results = []
    generated_single_reports: list[dict] = []
    generated_single_reports_en = 0
    generated_single_reports_ar = 0
    skipped_files = 0

    _log_verbose(f"Scanning source directory: {SOURCE_FILES_DIR}")
    for file_path in SOURCE_FILES_DIR.iterdir():
        if not file_path.is_file() or file_path.name == ".gitkeep":
            continue

        file_start = perf_counter()
        _log("INFO", f"START file: {file_path.name}")

        file_type = get_file_classification(file_path.name)
        mime = mimetypes.guess_type(str(file_path))[0] or "application/octet-stream"
        checksum = _sha256_file(file_path)
        safe_name = _safe_key_name(file_path.name)
        _log_verbose(f"Classified file '{file_path.name}' as type='{file_type}', mime='{mime}'")

        file_record = file_repo.get_by_checksum(checksum)
        if not file_record:
            file_record = file_repo.create(
                original_name=file_path.name,
                checksum=checksum,
                size=file_path.stat().st_size,
                mime=mime,
                classification=file_type,
                status="received",
            )
            _log_verbose(f"Created metadata file row: file_id={file_record.id}, status=received")
            artifact_repo.create(
                file_id=file_record.id,
                artifact_type="source",
                storage_backend="local",
                object_key=str(file_path),
                content_type=mime,
                size=file_path.stat().st_size,
            )
            _log_verbose(f"Registered source artifact for file_id={file_record.id}")
            _upload_to_minio_and_record(
                object_storage=object_storage,
                artifact_repo=artifact_repo,
                file_id=file_record.id,
                artifact_type="source",
                local_path=file_path,
                object_key=f"source/{file_record.id}/{safe_name}",
                content_type=mime,
            )
        else:
            _log_verbose(
                f"Found existing file row by checksum: file_id={file_record.id}, status={file_record.status}"
            )

        error = None
        extract_job = None
        single_report_job = None

        single_report_path_en = _single_report_path_for_file_by_language(file_path.name, "en")
        single_report_path_ar = _single_report_path_for_file_by_language(file_path.name, "ar")
        single_report_text_path_en = _single_report_text_path_for_file_by_language(file_path.name, "en")
        single_report_text_path_ar = _single_report_text_path_for_file_by_language(file_path.name, "ar")
        single_report_pdf_path_en = _single_report_pdf_path_for_file_by_language(file_path.name, "en")
        single_report_pdf_path_ar = _single_report_pdf_path_for_file_by_language(file_path.name, "ar")

        has_json_reports = single_report_path_en.exists() and single_report_path_ar.exists()
        has_dual_format_reports = (
            has_json_reports
            and single_report_text_path_en.exists()
            and single_report_text_path_ar.exists()
            and single_report_pdf_path_en.exists()
            and single_report_pdf_path_ar.exists()
        )

        if has_dual_format_reports:
            file_repo.update_status(file_record.id, "skipped")
            _log("INFO", f"SKIP file: {file_path.name} (already has EN+AR single reports)")
            skipped_files += 1
            results.append(
                {
                    "file_id": file_record.id,
                    "source_file": str(file_path),
                    "classification": file_type,
                    "status": "skipped",
                    "output_file": None,
                    "single_report_file_en": str(single_report_path_en),
                    "single_report_file_ar": str(single_report_path_ar),
                    "single_report_text_file_en": str(single_report_text_path_en) if single_report_text_path_en.exists() else None,
                    "single_report_text_file_ar": str(single_report_text_path_ar) if single_report_text_path_ar.exists() else None,
                    "single_report_pdf_file_en": str(single_report_pdf_path_en) if single_report_pdf_path_en.exists() else None,
                    "single_report_pdf_file_ar": str(single_report_pdf_path_ar) if single_report_pdf_path_ar.exists() else None,
                    "error": None,
                }
            )
            elapsed = perf_counter() - file_start
            _log("INFO", f"DONE file: {file_path.name} (status=skipped, {elapsed:.2f}s)")
            continue

        try:
            extract_job = job_repo.create(file_id=file_record.id, job_type="extract", state="queued")
            _log_verbose(f"Extract job queued: job_id={extract_job.id}, file_id={file_record.id}")
            job_repo.transition(extract_job.id, "processing")
            _log_verbose(f"Extract job processing: job_id={extract_job.id}, file_id={file_record.id}")

            extracted = _extract_by_type(file_path, file_type)
            _log_verbose(f"Extraction completed in pipeline for file_id={file_record.id}")

            output_path = save_raw_data(file_path.name, extracted)
            _log_verbose(f"Saved raw extraction JSON: {output_path}")
            artifact_repo.create(
                file_id=file_record.id,
                artifact_type="raw_extraction",
                storage_backend="local",
                object_key=str(output_path),
                content_type="application/json",
                size=output_path.stat().st_size,
            )
            _upload_to_minio_and_record(
                object_storage=object_storage,
                artifact_repo=artifact_repo,
                file_id=file_record.id,
                artifact_type="raw_extraction",
                local_path=output_path,
                object_key=f"artifacts/raw/{file_record.id}_{safe_name}.json",
                content_type="application/json",
            )
            job_repo.transition(extract_job.id, "completed")
            _log_verbose(f"Extract job completed: job_id={extract_job.id}, file_id={file_record.id}")

            single_report_job = job_repo.create(file_id=file_record.id, job_type="single_report", state="queued")
            _log_verbose(f"Single-report job queued: job_id={single_report_job.id}, file_id={file_record.id}")
            job_repo.transition(single_report_job.id, "processing")
            _log_verbose(f"Single-report job processing: job_id={single_report_job.id}, file_id={file_record.id}")

            single_report_en = generate_single_report(
                file_name=file_path.name,
                file_type=file_type,
                extraction_data=extracted,
                language="en",
            )
            generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            single_report_data_en = single_report_en.model_dump()
            single_report_data_en["language"] = "en"
            single_report_data_en["generated_at"] = generated_at

            single_report_path_en = save_single_report_by_language(file_path.name, single_report_data_en, "en")
            single_report_text_path_en = save_single_report_text_by_language(
                file_path.name,
                single_report_data_en.get("report_text", ""),
                "en",
            )
            single_report_pdf_path_en = save_single_report_pdf_by_language(
                file_path.name,
                single_report_data_en.get("report_text", ""),
                "en",
            )
            _log_verbose(f"Saved EN single report JSON: {single_report_path_en}")
            _log_verbose(f"Saved EN single report TXT: {single_report_text_path_en}")
            _log_verbose(f"Saved EN single report PDF: {single_report_pdf_path_en}")
            artifact_repo.create(
                file_id=file_record.id,
                artifact_type="single_report_en",
                storage_backend="local",
                object_key=str(single_report_path_en),
                content_type="application/json",
                size=single_report_path_en.stat().st_size,
            )
            artifact_repo.create(
                file_id=file_record.id,
                artifact_type="single_report_en_txt",
                storage_backend="local",
                object_key=str(single_report_text_path_en),
                content_type="text/plain",
                size=single_report_text_path_en.stat().st_size,
            )
            artifact_repo.create(
                file_id=file_record.id,
                artifact_type="single_report_en_pdf",
                storage_backend="local",
                object_key=str(single_report_pdf_path_en),
                content_type="application/pdf",
                size=single_report_pdf_path_en.stat().st_size,
            )
            _upload_to_minio_and_record(
                object_storage=object_storage,
                artifact_repo=artifact_repo,
                file_id=file_record.id,
                artifact_type="single_report_en",
                local_path=single_report_path_en,
                object_key=f"artifacts/single_reports/{file_record.id}_{safe_name}.report.en.json",
                content_type="application/json",
            )
            _upload_to_minio_and_record(
                object_storage=object_storage,
                artifact_repo=artifact_repo,
                file_id=file_record.id,
                artifact_type="single_report_en_txt",
                local_path=single_report_text_path_en,
                object_key=f"artifacts/single_reports/{file_record.id}_{safe_name}.report.en.txt",
                content_type="text/plain",
            )
            _upload_to_minio_and_record(
                object_storage=object_storage,
                artifact_repo=artifact_repo,
                file_id=file_record.id,
                artifact_type="single_report_en_pdf",
                local_path=single_report_pdf_path_en,
                object_key=f"artifacts/single_reports/{file_record.id}_{safe_name}.report.en.pdf",
                content_type="application/pdf",
            )

            single_report_ar = generate_single_report(
                file_name=file_path.name,
                file_type=file_type,
                extraction_data=extracted,
                language="ar",
            )
            single_report_data_ar = single_report_ar.model_dump()
            single_report_data_ar["language"] = "ar"
            single_report_data_ar["generated_at"] = generated_at

            single_report_path_ar = save_single_report_by_language(file_path.name, single_report_data_ar, "ar")
            single_report_text_path_ar = save_single_report_text_by_language(
                file_path.name,
                single_report_data_ar.get("report_text", ""),
                "ar",
            )
            single_report_pdf_path_ar = save_single_report_pdf_by_language(
                file_path.name,
                single_report_data_ar.get("report_text", ""),
                "ar",
            )
            _log_verbose(f"Saved AR single report JSON: {single_report_path_ar}")
            _log_verbose(f"Saved AR single report TXT: {single_report_text_path_ar}")
            _log_verbose(f"Saved AR single report PDF: {single_report_pdf_path_ar}")
            artifact_repo.create(
                file_id=file_record.id,
                artifact_type="single_report_ar",
                storage_backend="local",
                object_key=str(single_report_path_ar),
                content_type="application/json",
                size=single_report_path_ar.stat().st_size,
            )
            artifact_repo.create(
                file_id=file_record.id,
                artifact_type="single_report_ar_txt",
                storage_backend="local",
                object_key=str(single_report_text_path_ar),
                content_type="text/plain",
                size=single_report_text_path_ar.stat().st_size,
            )
            artifact_repo.create(
                file_id=file_record.id,
                artifact_type="single_report_ar_pdf",
                storage_backend="local",
                object_key=str(single_report_pdf_path_ar),
                content_type="application/pdf",
                size=single_report_pdf_path_ar.stat().st_size,
            )
            _upload_to_minio_and_record(
                object_storage=object_storage,
                artifact_repo=artifact_repo,
                file_id=file_record.id,
                artifact_type="single_report_ar",
                local_path=single_report_path_ar,
                object_key=f"artifacts/single_reports/{file_record.id}_{safe_name}.report.ar.json",
                content_type="application/json",
            )
            _upload_to_minio_and_record(
                object_storage=object_storage,
                artifact_repo=artifact_repo,
                file_id=file_record.id,
                artifact_type="single_report_ar_txt",
                local_path=single_report_text_path_ar,
                object_key=f"artifacts/single_reports/{file_record.id}_{safe_name}.report.ar.txt",
                content_type="text/plain",
            )
            _upload_to_minio_and_record(
                object_storage=object_storage,
                artifact_repo=artifact_repo,
                file_id=file_record.id,
                artifact_type="single_report_ar_pdf",
                local_path=single_report_pdf_path_ar,
                object_key=f"artifacts/single_reports/{file_record.id}_{safe_name}.report.ar.pdf",
                content_type="application/pdf",
            )

            job_repo.transition(single_report_job.id, "completed")
            _log_verbose(f"Single-report job completed: job_id={single_report_job.id}, file_id={file_record.id}")
            file_repo.update_status(file_record.id, "processed")
            _log_verbose(f"File marked processed: file_id={file_record.id}")
            generated_single_reports.append(single_report_en.model_dump())
            generated_single_reports_en += 1
            generated_single_reports_ar += 1

        except Exception as exc:
            output_path = None
            single_report_path_en = None
            single_report_path_ar = None
            single_report_text_path_en = None
            single_report_text_path_ar = None
            single_report_pdf_path_en = None
            single_report_pdf_path_ar = None
            error = str(exc)
            _log("ERROR", f"Failed processing file_id={file_record.id}: {error}")

            if single_report_job:
                try:
                    current = job_repo.get_by_id(single_report_job.id)
                    if current and current.state in {"queued", "processing"}:
                        job_repo.mark_failed(single_report_job.id, error)
                except Exception:
                    pass

            if extract_job:
                try:
                    current = job_repo.get_by_id(extract_job.id)
                    if current and current.state in {"queued", "processing"}:
                        job_repo.mark_failed(extract_job.id, error)
                except Exception:
                    pass

            file_repo.update_status(file_record.id, "failed", error)
            _log("ERROR", f"File marked failed: file_id={file_record.id}")

        results.append(
            {
                "file_id": file_record.id,
                "source_file": str(file_path),
                "classification": file_type,
                "status": "success" if error is None else "failed",
                "output_file": str(output_path) if output_path else None,
                "single_report_file_en": str(single_report_path_en) if single_report_path_en else None,
                "single_report_file_ar": str(single_report_path_ar) if single_report_path_ar else None,
                "single_report_text_file_en": str(single_report_text_path_en) if single_report_text_path_en else None,
                "single_report_text_file_ar": str(single_report_text_path_ar) if single_report_text_path_ar else None,
                "single_report_pdf_file_en": str(single_report_pdf_path_en) if single_report_pdf_path_en else None,
                "single_report_pdf_file_ar": str(single_report_pdf_path_ar) if single_report_pdf_path_ar else None,
                "error": error,
            }
        )
        elapsed = perf_counter() - file_start
        _log("INFO", f"DONE file: {file_path.name} (status={'success' if error is None else 'failed'}, {elapsed:.2f}s)")

    # Always regenerate final report using all available single reports
    # (old + newly generated in this run).
    final_report_path = None
    final_report_path_en = None
    final_report_path_ar = None
    final_report_text_path = None
    final_report_text_path_en = None
    final_report_text_path_ar = None
    final_report_pdf_path = None
    final_report_pdf_path_en = None
    final_report_pdf_path_ar = None
    final_report_job_id = None
    final_run_id = None
    all_single_reports_en = _load_all_single_reports(language="en")
    all_single_reports_ar = _load_all_single_reports(language="ar")
    if all_single_reports_en and all_single_reports_ar:
        _log("INFO", f"START final report: {len(all_single_reports_en)} EN and {len(all_single_reports_ar)} AR single report(s)")
        final_report_job = job_repo.create(file_id=None, job_type="final_report", state="queued")
        final_report_job_id = final_report_job.id
        _log_verbose(f"Final-report job queued: job_id={final_report_job.id}")
        job_repo.transition(final_report_job.id, "processing")
        _log_verbose(f"Final-report job processing: job_id={final_report_job.id}")

        run = run_repo.create(
            trigger_type="source_file_processor",
            state="queued",
            input_count=min(len(all_single_reports_en), len(all_single_reports_ar)),
        )
        final_run_id = run.id
        run_repo.mark_processing(run.id)
        _log_verbose(f"Final report run started: run_id={run.id}")

        try:
            normalized_single_reports_en = [_normalize_single_report_for_final(x) for x in all_single_reports_en]
            normalized_single_reports_ar = [_normalize_single_report_for_final(x) for x in all_single_reports_ar]

            final_report_en = generate_final_report(normalized_single_reports_en, language="en")
            final_report_data_en = final_report_en.model_dump()
            final_report_ar = generate_final_report(normalized_single_reports_ar, language="ar")
            final_report_data_ar = final_report_ar.model_dump()
            _log_verbose("LLM final report generation completed")

            if not _is_final_report_meaningful(final_report_data_en):
                _log("WARN", "EN final report not meaningful; using fallback aggregation")
                final_report_data_en = _build_fallback_final_report(all_single_reports_en)
            if not _is_final_report_meaningful(final_report_data_ar):
                _log("WARN", "AR final report not meaningful; using fallback aggregation")
                final_report_data_ar = _build_fallback_final_report(all_single_reports_ar)

            generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            final_report_data_en["language"] = "en"
            final_report_data_en["generated_at"] = generated_at
            final_report_data_ar["language"] = "ar"
            final_report_data_ar["generated_at"] = generated_at

            final_report_path_en = save_final_report_by_language(final_report_data_en, "en")
            final_report_path_ar = save_final_report_by_language(final_report_data_ar, "ar")
            final_report_path = save_final_report(final_report_data_en)
            final_report_text_path_en = save_final_report_text_by_language(final_report_data_en.get("report_text", ""), "en")
            final_report_text_path_ar = save_final_report_text_by_language(final_report_data_ar.get("report_text", ""), "ar")
            final_report_text_path = save_final_report_text(final_report_data_en.get("report_text", ""))
            final_report_pdf_path_en = save_final_report_pdf_by_language(final_report_data_en.get("report_text", ""), "en")
            final_report_pdf_path_ar = save_final_report_pdf_by_language(final_report_data_ar.get("report_text", ""), "ar")
            final_report_pdf_path = save_final_report_pdf(final_report_data_en.get("report_text", ""))
            _log_verbose(f"Saved EN final report JSON: {final_report_path_en}")
            _log_verbose(f"Saved AR final report JSON: {final_report_path_ar}")
            _log_verbose(f"Saved EN final report TXT: {final_report_text_path_en}")
            _log_verbose(f"Saved AR final report TXT: {final_report_text_path_ar}")
            _log_verbose(f"Saved EN final report PDF: {final_report_pdf_path_en}")
            _log_verbose(f"Saved AR final report PDF: {final_report_pdf_path_ar}")

            local_final_artifact_en = artifact_repo.create(
                file_id=None,
                artifact_type="final_report_en",
                storage_backend="local",
                object_key=str(final_report_path_en),
                content_type="application/json",
                size=final_report_path_en.stat().st_size,
            )
            artifact_repo.create(
                file_id=None,
                artifact_type="final_report_ar",
                storage_backend="local",
                object_key=str(final_report_path_ar),
                content_type="application/json",
                size=final_report_path_ar.stat().st_size,
            )
            artifact_repo.create(
                file_id=None,
                artifact_type="final_report_en_txt",
                storage_backend="local",
                object_key=str(final_report_text_path_en),
                content_type="text/plain",
                size=final_report_text_path_en.stat().st_size,
            )
            artifact_repo.create(
                file_id=None,
                artifact_type="final_report_ar_txt",
                storage_backend="local",
                object_key=str(final_report_text_path_ar),
                content_type="text/plain",
                size=final_report_text_path_ar.stat().st_size,
            )
            artifact_repo.create(
                file_id=None,
                artifact_type="final_report_en_pdf",
                storage_backend="local",
                object_key=str(final_report_pdf_path_en),
                content_type="application/pdf",
                size=final_report_pdf_path_en.stat().st_size,
            )
            artifact_repo.create(
                file_id=None,
                artifact_type="final_report_ar_pdf",
                storage_backend="local",
                object_key=str(final_report_pdf_path_ar),
                content_type="application/pdf",
                size=final_report_pdf_path_ar.stat().st_size,
            )

            minio_artifact_id_en = _upload_to_minio_and_record(
                object_storage=object_storage,
                artifact_repo=artifact_repo,
                file_id=None,
                artifact_type="final_report_en",
                local_path=final_report_path_en,
                object_key=(
                    f"artifacts/final_reports/"
                    f"{datetime.now().strftime('%Y%m%d-%H%M%S')}_run-{run.id}.en.json"
                ),
                content_type="application/json",
            )
            _upload_to_minio_and_record(
                object_storage=object_storage,
                artifact_repo=artifact_repo,
                file_id=None,
                artifact_type="final_report_ar",
                local_path=final_report_path_ar,
                object_key=(
                    f"artifacts/final_reports/"
                    f"{datetime.now().strftime('%Y%m%d-%H%M%S')}_run-{run.id}.ar.json"
                ),
                content_type="application/json",
            )
            _upload_to_minio_and_record(
                object_storage=object_storage,
                artifact_repo=artifact_repo,
                file_id=None,
                artifact_type="final_report_en_txt",
                local_path=final_report_text_path_en,
                object_key=(
                    f"artifacts/final_reports/"
                    f"{datetime.now().strftime('%Y%m%d-%H%M%S')}_run-{run.id}.en.txt"
                ),
                content_type="text/plain",
            )
            _upload_to_minio_and_record(
                object_storage=object_storage,
                artifact_repo=artifact_repo,
                file_id=None,
                artifact_type="final_report_ar_txt",
                local_path=final_report_text_path_ar,
                object_key=(
                    f"artifacts/final_reports/"
                    f"{datetime.now().strftime('%Y%m%d-%H%M%S')}_run-{run.id}.ar.txt"
                ),
                content_type="text/plain",
            )
            _upload_to_minio_and_record(
                object_storage=object_storage,
                artifact_repo=artifact_repo,
                file_id=None,
                artifact_type="final_report_en_pdf",
                local_path=final_report_pdf_path_en,
                object_key=(
                    f"artifacts/final_reports/"
                    f"{datetime.now().strftime('%Y%m%d-%H%M%S')}_run-{run.id}.en.pdf"
                ),
                content_type="application/pdf",
            )
            _upload_to_minio_and_record(
                object_storage=object_storage,
                artifact_repo=artifact_repo,
                file_id=None,
                artifact_type="final_report_ar_pdf",
                local_path=final_report_pdf_path_ar,
                object_key=(
                    f"artifacts/final_reports/"
                    f"{datetime.now().strftime('%Y%m%d-%H%M%S')}_run-{run.id}.ar.pdf"
                ),
                content_type="application/pdf",
            )

            run_repo.mark_completed(run.id, minio_artifact_id_en or local_final_artifact_en.id)
            job_repo.transition(final_report_job.id, "completed")
            _log("INFO", f"DONE final report: EN={final_report_path_en}, AR={final_report_path_ar}")
        except Exception as exc:
            run_repo.mark_failed(run.id, str(exc))
            job_repo.mark_failed(final_report_job.id, str(exc))
            _log("ERROR", f"Final report generation failed: {exc}")
            raise
    else:
        _log("INFO", "SKIP final report: missing EN/AR single reports")

    run_elapsed = perf_counter() - run_start
    success_count = sum(1 for r in results if r.get("status") == "success")
    failed_count = sum(1 for r in results if r.get("status") == "failed")
    _log(
        "INFO",
        (
            f"SUMMARY run: processed={len(results)}, success={success_count}, "
            f"failed={failed_count}, skipped={skipped_files}, duration={run_elapsed:.2f}s"
        ),
    )

    return {
        "processed": len(results),
        "skipped": skipped_files,
        "single_reports_generated": len(generated_single_reports),
        "single_reports_generated_en": generated_single_reports_en,
        "single_reports_generated_ar": generated_single_reports_ar,
        "single_reports_used_for_final_en": len(all_single_reports_en),
        "single_reports_used_for_final_ar": len(all_single_reports_ar),
        "final_report_file": str(final_report_path) if final_report_path else None,
        "final_report_file_en": str(final_report_path_en) if final_report_path_en else None,
        "final_report_file_ar": str(final_report_path_ar) if final_report_path_ar else None,
        "final_report_text_file": str(final_report_text_path) if final_report_text_path else None,
        "final_report_text_file_en": str(final_report_text_path_en) if final_report_text_path_en else None,
        "final_report_text_file_ar": str(final_report_text_path_ar) if final_report_text_path_ar else None,
        "final_report_pdf_file": str(final_report_pdf_path) if final_report_pdf_path else None,
        "final_report_pdf_file_en": str(final_report_pdf_path_en) if final_report_pdf_path_en else None,
        "final_report_pdf_file_ar": str(final_report_pdf_path_ar) if final_report_pdf_path_ar else None,
        "final_report_job_id": final_report_job_id,
        "final_report_run_id": final_run_id,
        "metadata_db": str(DB_PATH),
        "results": results,
    }


if __name__ == "__main__":
    summary = process_source_files()
    print(summary)
