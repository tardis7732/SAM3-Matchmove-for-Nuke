import nuke
import sam3_ofx_nuke
import sam3_matchmove_nuke

nuke.menu('Nodes').addCommand('AI/SAM3 Mask OFX', 'sam3_ofx_nuke.create()')
# Preserve the original node for existing scripts and video identity tracking.
sam3_matchmove_nuke.register_menu()
