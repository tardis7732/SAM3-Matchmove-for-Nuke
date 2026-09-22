import json
import os

_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_build_default = os.path.join(_root, "ofx", "build")
try:
    with open(os.path.join(_root, "config", "frontend.json"), encoding="utf-8-sig") as _file:
        _build_default = json.load(_file).get("ofx_build_dir", _build_default)
except (OSError, ValueError):
    pass
_bundles = os.environ.get("SAM3_OFX_BUILD_DIR", _build_default)
_current = os.environ.get("OFX_PLUGIN_PATH", "")
if _bundles not in _current.split(os.pathsep):
    os.environ["OFX_PLUGIN_PATH"] = _bundles + (os.pathsep + _current if _current else "")

import sam3_unified
sam3_unified.register_callbacks()
