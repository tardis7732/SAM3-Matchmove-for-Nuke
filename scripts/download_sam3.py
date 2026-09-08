"""Download official SAM3 using hf_token from .env, without persisting credentials."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def read_token(path):
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        stripped = line.strip()
        if stripped.startswith("export "):
            stripped = stripped[7:].lstrip()
        key, separator, value = stripped.partition("=")
        if separator and key.strip().lower() == "hf_token":
            parts = shlex.split(value.strip(), comments=True, posix=True)
            if len(parts) != 1 or not parts[0]:
                raise ValueError("hf_token is empty or malformed in the selected .env file")
            return parts[0]
    raise ValueError("No hf_token entry was found in the selected .env file")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", type=Path, default=ROOT.parent / ".env")
    args = parser.parse_args()
    token = None
    try:
        token = read_token(args.env_file)
        from huggingface_hub import get_hf_file_metadata, hf_hub_url
        metadata = get_hf_file_metadata(hf_hub_url("facebook/sam3", "sam3.pt"), token=token)
        print(json.dumps({"official_access": True, "size_bytes": metadata.size,
                          "revision": metadata.commit_hash}), flush=True)
        destination = ROOT / "checkpoints"
        destination.mkdir(exist_ok=True)
        environment = dict(os.environ, HF_TOKEN=token, HF_HUB_DISABLE_PROGRESS_BARS="1",
                           PYTHONUTF8="1", PYTHONIOENCODING="utf-8")
        command = [str(Path(sys.executable).with_name("hf.exe")), "download", "facebook/sam3",
                   "sam3.pt", "--revision", metadata.commit_hash, "--local-dir", str(destination)]
        process = subprocess.Popen(command, env=environment, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace",
                                   creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        for line in process.stdout:
            print(line.replace(token, "[REDACTED]"), end="", flush=True)
        if process.wait() != 0:
            raise RuntimeError("Official HF download failed; see the sanitized CLI output above")
        checkpoint = destination / "sam3.pt"
        if checkpoint.stat().st_size != metadata.size:
            raise RuntimeError("Checkpoint size differs from the official metadata")
        digest = hashlib.sha256()
        with checkpoint.open("rb") as stream:
            for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
        sha256 = digest.hexdigest()
        if not re.fullmatch(r"[a-fA-F0-9]{64}", metadata.etag or ""):
            raise RuntimeError("Official checkpoint metadata does not provide the expected SHA-256")
        if sha256.lower() != metadata.etag.lower():
            raise RuntimeError("Checkpoint SHA-256 differs from the official metadata")
        result = {"ok": True, "repository": "facebook/sam3", "revision": metadata.commit_hash,
                  "file": str(checkpoint), "size_bytes": metadata.size, "sha256": sha256,
                  "matches_official_sha256": True, "credentials_persisted": False}
        report = ROOT / "validation" / "sam3_runtime" / "checkpoint_download.json"
        report.parent.mkdir(exist_ok=True, parents=True)
        report.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(result, indent=2), flush=True)
        return 0
    except Exception as exc:
        # Avoid raw exception representations: HTTP errors may contain request details.
        response = getattr(exc, "response", None)
        print(json.dumps({"ok": False, "error_type": type(exc).__name__,
                          "http_status": getattr(response, "status_code", None),
                          "message": "Official SAM3 download or verification did not complete."}), flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
