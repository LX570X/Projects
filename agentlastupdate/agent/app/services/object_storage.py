"""
Purpose:
- MinIO object storage helper for uploading files/JSON artifacts.

Libraries used:
- os/pathlib/io/json/typing: config handling and payload/file helpers.
- minio (runtime import): S3-compatible object storage client.
"""

import io
import json
import os
from pathlib import Path
from typing import Any, Optional


def _as_bool(value: str, default: bool = False) -> bool:
    if value is None:
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


class ObjectStorageService:
    def __init__(self) -> None:
        self.endpoint = os.getenv("MINIO_ENDPOINT", "").strip()
        self.access_key = os.getenv("MINIO_ACCESS_KEY", "").strip()
        self.secret_key = os.getenv("MINIO_SECRET_KEY", "").strip()
        self.bucket = os.getenv("MINIO_BUCKET", "incident-analysis").strip()
        self.secure = _as_bool(os.getenv("MINIO_SECURE", "false"), default=False)
        self.enabled = _as_bool(os.getenv("MINIO_ENABLED", "false"), default=False)
        self.fail_open = _as_bool(os.getenv("MINIO_FAIL_OPEN", "true"), default=True)
        self.init_error: Optional[str] = None

        self._client: Optional[Any] = None
        if self.enabled:
            try:
                self._validate_required_config()
                try:
                    from minio import Minio  # type: ignore
                except Exception as e:
                    raise RuntimeError(
                        "MinIO is enabled but 'minio' package is not installed. "
                        "Install with: pip install -r requirements.txt"
                    ) from e

                self._client = Minio(
                    self.endpoint,
                    access_key=self.access_key,
                    secret_key=self.secret_key,
                    secure=self.secure,
                )
                self._ensure_bucket()
            except Exception as e:
                if not self.fail_open:
                    raise
                self.init_error = str(e)
                self.enabled = False
                self._client = None

    def _validate_required_config(self) -> None:
        missing = []
        if not self.endpoint:
            missing.append("MINIO_ENDPOINT")
        if not self.access_key:
            missing.append("MINIO_ACCESS_KEY")
        if not self.secret_key:
            missing.append("MINIO_SECRET_KEY")
        if not self.bucket:
            missing.append("MINIO_BUCKET")

        if missing:
            raise ValueError(f"MinIO is enabled but missing required settings: {', '.join(missing)}")

    def _ensure_bucket(self) -> None:
        if not self._client:
            return
        try:
            if not self._client.bucket_exists(self.bucket):
                self._client.make_bucket(self.bucket)
        except Exception as e:
            raise RuntimeError(f"Failed to ensure MinIO bucket '{self.bucket}': {e}") from e

    def upload_file(self, *, local_path: Path, object_key: str, content_type: Optional[str] = None) -> dict:
        if not self.enabled or not self._client:
            raise RuntimeError("MinIO upload requested while object storage is disabled")

        file_size = local_path.stat().st_size
        result = self._client.fput_object(
            bucket_name=self.bucket,
            object_name=object_key,
            file_path=str(local_path),
            content_type=content_type,
        )

        return {
            "bucket": self.bucket,
            "object_key": object_key,
            "etag": result.etag,
            "version_id": result.version_id,
            "size": file_size,
            "content_type": content_type,
        }

    def upload_json(self, *, data: dict, object_key: str) -> dict:
        if not self.enabled or not self._client:
            raise RuntimeError("MinIO upload requested while object storage is disabled")

        payload = json.dumps(data, ensure_ascii=False, indent=2).encode("utf-8")
        stream = io.BytesIO(payload)
        content_type = "application/json"

        result = self._client.put_object(
            bucket_name=self.bucket,
            object_name=object_key,
            data=stream,
            length=len(payload),
            content_type=content_type,
        )

        return {
            "bucket": self.bucket,
            "object_key": object_key,
            "etag": result.etag,
            "version_id": result.version_id,
            "size": len(payload),
            "content_type": content_type,
        }
