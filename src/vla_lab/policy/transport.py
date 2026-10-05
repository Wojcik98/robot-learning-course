"""Pack and restore complete ACT reference bundles for another machine."""

from __future__ import annotations

import hashlib
import os
import tempfile
import zipfile
from pathlib import Path

import yaml

from vla_lab.types import JSONValue


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_act_manifest(workspace: Path) -> dict[str, JSONValue]:
    """Read the pinned reference bundle paths and file hashes."""
    return yaml.safe_load((workspace / "manifests" / "act_references.yaml").read_text())


def _entries(workspace: Path, manifest: dict[str, JSONValue]) -> list[tuple[str, Path, str]]:
    entries = []
    for task, reference in manifest["tasks"].items():
        for filename, digest in reference["checkpoint_files_sha256"].items():
            entries.append(
                (f"{task}/{filename}", workspace / reference["checkpoint_path"] / filename, digest)
            )
    return entries


def pack_act_references(workspace: Path, archive: Path) -> dict[str, JSONValue]:
    """Write the seven required files for each verified ACT checkpoint."""
    manifest = read_act_manifest(workspace)
    entries = _entries(workspace, manifest)
    for _, source, expected in entries:
        if not source.is_file() or _sha256(source) != expected:
            raise ValueError(f"ACT reference file is missing or changed: {source}")
    if archive.exists():
        raise FileExistsError(f"Archive already exists: {archive}")
    archive.parent.mkdir(parents=True, exist_ok=True)
    temporary = archive.with_suffix(archive.suffix + ".tmp")
    try:
        with zipfile.ZipFile(
            temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=1
        ) as bundle:
            for name, source, _ in entries:
                bundle.write(source, arcname=name)
        os.replace(temporary, archive)
    finally:
        temporary.unlink(missing_ok=True)
    return {"archive": str(archive.resolve()), "sha256": _sha256(archive), "files": len(entries)}


def install_act_references(workspace: Path, archive: Path) -> dict[str, JSONValue]:
    """Verify every archive member before it installs either checkpoint."""
    manifest = read_act_manifest(workspace)
    entries = _entries(workspace, manifest)
    names = {name for name, _, _ in entries}
    with zipfile.ZipFile(archive) as bundle:
        if len(bundle.namelist()) != len(names) or set(bundle.namelist()) != names:
            raise ValueError("ACT archive member list differs from the pinned manifest")
        for name, _, expected in entries:
            with bundle.open(name) as stream:
                digest = hashlib.sha256()
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(block)
            if digest.hexdigest() != expected:
                raise ValueError(f"ACT archive file hash differs: {name}")
        for _, target, expected in entries:
            if target.exists() and _sha256(target) != expected:
                raise FileExistsError(f"Target has changed: {target}")
        stage_parent = workspace / ".local-data"
        stage_parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="act-import-", dir=stage_parent) as stage_name:
            stage = Path(stage_name)
            for name, target, _ in entries:
                if target.is_file():
                    continue
                staged = stage / name
                staged.parent.mkdir(parents=True, exist_ok=True)
                with bundle.open(name) as source, staged.open("wb") as output:
                    while block := source.read(1024 * 1024):
                        output.write(block)
            for name, target, _ in entries:
                if target.is_file():
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                os.replace(stage / name, target)
    return {"archive": str(archive.resolve()), "sha256": _sha256(archive), "files": len(entries)}
