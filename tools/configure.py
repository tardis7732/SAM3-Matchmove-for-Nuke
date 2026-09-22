"""Connect OFX to a new or existing SAM3 environment; never copy models."""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--matchmove-root', type=Path, default=ROOT,
                        help='Read config.local.json from this SAM3 installation (default: this repository)')
    parser.add_argument('--python-exe', type=Path, help='Override the external Python executable')
    parser.add_argument('--checkpoint', type=Path, help='Override the official sam3.pt path')
    args = parser.parse_args()
    matchmove = args.matchmove_root.resolve()
    profile = matchmove / 'config.local.json'
    settings = json.loads(profile.read_text(encoding='utf-8-sig')) if profile.is_file() else {}
    def resolve(value):
        path = Path(value).expanduser()
        return (path if path.is_absolute() else matchmove / path).resolve()
    python_value = args.python_exe or settings.get('python_exe')
    checkpoint_value = args.checkpoint or settings.get('checkpoint')
    if not python_value or not checkpoint_value:
        parser.error('Run scripts/setup_sam3.ps1 and scripts/auth_sam3.ps1 first, or supply --python-exe and --checkpoint.')
    python = resolve(python_value)
    checkpoint = resolve(checkpoint_value)
    if not python.is_file() or not checkpoint.is_file():
        raise FileNotFoundError('Install the SAM3 Matchmove environment and checkpoint first')
    config = ROOT / 'config'
    config.mkdir(exist_ok=True)
    launcher = {'command': [str(python), str(ROOT / 'daemon/sam3_daemon.py'), '--checkpoint', str(checkpoint)], 'cwd': str(ROOT)}
    # Always use the tracking/export helpers shipped with this checkout, even
    # when Python and the checkpoint come from another installation.
    frontend = {'python': str(python), 'ofx_build_dir': str(ROOT / 'ofx/prebuilt'), 'matchmove_root': str(ROOT)}
    for name, data in [('launcher.json', launcher), ('frontend.json', frontend)]:
        (config / name).write_text(json.dumps(data, indent=2), encoding='utf-8')
    print('Configured SAM3 OFX with existing CUDA environment:', python)


if __name__ == '__main__':
    main()
