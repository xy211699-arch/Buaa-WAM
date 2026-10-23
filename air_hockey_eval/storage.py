"""Single-writer journals with identity-safe resume and interrupted-tail repair."""
from __future__ import annotations

import json
import os
from pathlib import Path

from .schema import canonical_json, fingerprint


def _validate_row(row: dict, expected: set[str]) -> None:
    if not isinstance(row, dict) or row.get("case_id") not in expected:
        raise ValueError("result has an unknown case_id")
    if type(row.get("success")) is not bool:
        raise ValueError("success must be a JSON boolean")
    if not row["success"] and not row.get("failure_reason"):
        raise ValueError("every failed case needs a reason")
    if row["success"] and (row.get("violations") or row.get("failure_reason")):
        raise ValueError("successful result cannot contain violations or a failure")
    canonical_json(row)  # reject NaN/Infinity, including nested telemetry


def _read_rows(path: Path, case_ids: list[str], *, repair: bool = False) -> list[dict]:
    if not path.exists():
        return []
    data = path.read_bytes()
    offset, rows, seen = 0, [], set()
    lines = data.splitlines(keepends=True)
    for index, line in enumerate(lines):
        try:
            row = json.loads(line.decode("utf-8"))
            _validate_row(row, set(case_ids))
        except (ValueError, UnicodeDecodeError, TypeError) as error:
            if repair and index == len(lines) - 1 and not line.endswith(b"\n"):
                with path.open("r+b") as file:
                    file.truncate(offset)
                break
            raise ValueError(f"invalid result journal at byte {offset}") from error
        if row["case_id"] in seen:
            raise ValueError("duplicate completed case in journal")
        seen.add(row["case_id"])
        rows.append(row)
        offset += len(line)
    if repair and rows and path.stat().st_size and not path.read_bytes().endswith(b"\n"):
        with path.open("ab") as file:
            file.write(b"\n"); file.flush(); os.fsync(file.fileno())
    return rows


class RunStore:
    def __init__(self, directory, identity: dict, case_ids: list[str], *, resume: bool = False):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        if not case_ids or len(case_ids) != len(set(case_ids)):
            raise ValueError("expected case ids must be nonempty and unique")
        self.case_ids, self.identity = list(case_ids), dict(identity)
        self._lock, self._journal = None, None
        try:
            self._acquire_lock()
            metadata = {"schema_version": 1, "identity": self.identity, "case_ids": self.case_ids}
            path = self.directory / "run.json"
            if path.exists():
                if not resume:
                    raise FileExistsError("run already exists; use --resume with the same identity")
                if json.loads(path.read_text(encoding="utf-8")) != metadata:
                    raise ValueError("resume identity/protocol/checkpoint or case list mismatch")
            else:
                with path.open("x", encoding="utf-8") as file:
                    file.write(canonical_json(metadata) + "\n")
                    file.flush(); os.fsync(file.fileno())
            rows = _read_rows(self.directory / "episodes.jsonl", self.case_ids, repair=True)
            self.completed = {row["case_id"]: row for row in rows}
            self._journal = (self.directory / "episodes.jsonl").open("a", encoding="utf-8", newline="\n")
        except BaseException:
            self.close()
            raise

    def _acquire_lock(self):
        self._lock = (self.directory / ".run.lock").open("a+b")
        self._lock.seek(0, 2)
        if self._lock.tell() == 0:
            self._lock.write(b"0"); self._lock.flush()
        self._lock.seek(0)
        if os.name == "nt":
            import msvcrt
            msvcrt.locking(self._lock.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(self._lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)

    def append(self, row: dict) -> None:
        _validate_row(row, set(self.case_ids))
        if row["case_id"] in self.completed:
            raise ValueError("case already completed")
        self._journal.write(canonical_json(row) + "\n")
        self._journal.flush()
        os.fsync(self._journal.fileno())
        self.completed[row["case_id"]] = row

    def finalize(self) -> None:
        if set(self.completed) != set(self.case_ids):
            raise ValueError("cannot finalize an incomplete run")
        output = {"complete": True, "cases": len(self.completed),
                  "identity_sha256": fingerprint(self.identity)}
        temporary = self.directory / "complete.json.partial"
        temporary.write_text(canonical_json(output) + "\n", encoding="utf-8")
        os.replace(temporary, self.directory / "complete.json")

    def close(self):
        if self._journal is not None:
            self._journal.close(); self._journal = None
        if self._lock is not None:
            self._lock.close(); self._lock = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def read_run(directory) -> dict:
    directory = Path(directory)
    metadata = json.loads((directory / "run.json").read_text(encoding="utf-8"))
    if metadata.get("schema_version") != 1:
        raise ValueError("unknown run format")
    rows = _read_rows(directory / "episodes.jsonl", metadata["case_ids"])
    return {"identity": metadata["identity"], "case_ids": metadata["case_ids"], "results": rows}
