"""Run this file in the Nuke Script Editor to load without restarting Nuke."""
from pathlib import Path
import sys
import nuke

plugin = str(Path(__file__).resolve().parent / "nuke")
nuke.pluginAddPath(plugin)
if plugin not in sys.path:
    sys.path.insert(0, plugin)
import sam3_matchmove_nuke
sam3_matchmove_nuke.register_menu()
sam3_matchmove_nuke.create_node()
