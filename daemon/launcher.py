"""Launch a configured native/WSL engine without a visible console.

The JSON command is an argv array, never a shell string. Windows process ids
are monitored here and never passed into the Linux pid namespace.
"""

import argparse
import ctypes
import json
import os
import subprocess
import time
import traceback
from pathlib import Path

from protocol import PORT, request

ROOT = Path(__file__).resolve().parents[1]


def parent_alive(pid):
    if not pid:
        return True
    if os.name == "nt":
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.OpenProcess.argtypes = [ctypes.c_uint32, ctypes.c_int, ctypes.c_uint32]
        kernel.OpenProcess.restype = ctypes.c_void_p
        kernel.WaitForSingleObject.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
        kernel.CloseHandle.argtypes = [ctypes.c_void_p]
        handle = kernel.OpenProcess(0x00100000, False, pid)
        if not handle:
            return ctypes.get_last_error() == 5  # access denied is not proof of death
        try:
            return kernel.WaitForSingleObject(handle, 0) == 258
        finally:
            kernel.CloseHandle(handle)
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def launch(args, log):
    profile = Path(args.config)
    if not profile.is_file():
        raise RuntimeError(
            "Engine is not configured. Run tools/configure.py; see README.md."
        )
    settings = json.loads(profile.read_text(encoding="utf-8-sig"))
    command = settings.get("command")
    if (
        not isinstance(command, list)
        or not command
        or any(not isinstance(a, str) or not a for a in command)
    ):
        raise ValueError("launcher.json command must be a nonempty string array")
    # Another host may already own this endpoint. Never terminate an engine we did not start.
    try:
        info, _ = request({"cmd": "info"}, port=args.port, timeout=2)
        if info.get("engine") == "SAM3 Mask":
            return
        raise RuntimeError("The configured port belongs to another service")
    except (OSError, ConnectionError):
        pass
    command = command + ["--port", str(args.port)]
    env = dict(os.environ)
    for name in ("PYTHONHOME", "PYTHONPATH", "PYTHONEXECUTABLE", "PYTHONSTARTUP"):
        env.pop(name, None)
    env["PYTHONUNBUFFERED"] = "1"
    for name in ('NUKE_PATH', 'QT_PLUGIN_PATH', 'QT_QPA_PLATFORM_PLUGIN_PATH'):
        env.pop(name, None)
    env['PATH'] = os.pathsep.join(p for p in env.get('PATH', '').split(os.pathsep)
                               if not any(part.lower().startswith('nuke') for part in Path(p).parts))
    env.update(PYTHONUTF8='1', PYTHONIOENCODING='utf-8', PYTHONNOUSERSITE='1')
    print("Starting configured SAM3 Mask engine", file=log, flush=True)
    child = subprocess.Popen(
        command,
        cwd=settings.get("cwd") or str(ROOT),
        env=env,
        stdin=subprocess.DEVNULL,
        stdout=log,
        stderr=subprocess.STDOUT,
        creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
    )
    try:
        while child.poll() is None:
            if not parent_alive(args.parent_pid):
                try:
                    request({"cmd": "shutdown"}, port=args.port, timeout=3)
                except (OSError, RuntimeError, ValueError):
                    pass
                try:
                    child.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    child.terminate()
                return
            time.sleep(1)
        if child.returncode:
            raise RuntimeError(
                f"Engine process exited with code {child.returncode}; see earlier log entries"
            )
    finally:
        if child.poll() is None:
            child.terminate()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default=str(ROOT / "config/launcher.json"))
    p.add_argument("--port", type=int, default=PORT)
    p.add_argument("--parent-pid", type=int, default=0)
    args = p.parse_args()
    (ROOT / "output").mkdir(exist_ok=True)
    with (ROOT / "output/daemon.log").open("a", encoding="utf-8", buffering=1) as log:
        try:
            launch(args, log)
        except Exception:
            traceback.print_exc(file=log)
            raise SystemExit(1)


if __name__ == "__main__":
    main()
