# SAM3 Mask OFX for Nuke

**SAM3 Matchmove has been updated to a single-node OFX workflow.**

[English](README.en.md) -> [한국어](README.md) -> [Usage guide](docs/USAGE.md)

https://github.com/user-attachments/assets/72ade68a-44d2-4ede-9832-d4da3031d503

*10-second edited demo: originals, SAM3 masks, stabilized crops, and native Nuke nodes for a ball, car, and face.*

> This demo was recorded with an earlier version. Following the OFX update, the UI and node structure shown in the video may differ from the current version. Follow the updated installation and usage instructions below.

**Analyze -> RAM masks -> Solve -> Export native Nuke nodes.** The OFX plugin receives Nuke pixels and plays the completed mask batch directly from RAM. No temporary plate sequence, mask PNGs, or result JSON/CSV files are written by the new workflow.

A prebuilt **Windows x64 OFX** is included. Normal installation does **not** require C++ build tools. Prepare the separate Python environment and official SAM3 checkpoint using the steps below.

## Requirements

- Windows x64; the OFX update has been checked in **Nuke 17.1v1**. Other hosts and versions are unverified.
- External **Python 3.12+**, Git, and an NVIDIA CUDA GPU for SAM3 inference. Do not install SAM3/PyTorch into Nuke's embedded Python.
- Access approval for [facebook/sam3](https://huggingface.co/facebook/sam3) and its official `sam3.pt` checkpoint.
- The setup script pins the official SAM3 source and installs PyTorch 2.10.0 / torchvision 0.25.0 CUDA 12.8 wheels. See [setup_sam3.ps1](scripts/setup_sam3.ps1) and [requirements](requirements-sam3-windows.txt).

## Fresh installation

Run these commands in PowerShell from the repository root. Replace the base Python path with your actual Python 3.12+ executable.

```powershell
git clone https://github.com/tardis7732/SAM3-Matchmove-for-Nuke.git
cd SAM3-Matchmove-for-Nuke
powershell -ExecutionPolicy Bypass -File scripts/setup_sam3.ps1 -PythonExe 'C:\Python312\python.exe'
powershell -ExecutionPolicy Bypass -File scripts/auth_sam3.ps1
.\.venv-sam3\Scripts\python.exe tools/configure.py
powershell -ExecutionPolicy Bypass -File install.ps1
```

Obtain model access before running authentication/download. Fully restart Nuke, then add **Tab -> SAM3 Mask OFX**.

`configure.py` reads `config.local.json` and creates `config/frontend.json` and `config/launcher.json`. The root installer configures the bundled OFX with `sam3mask.cfg` and registers the plugin path in `.nuke/init.py`, backing up an existing init file before modifying it. These machine-specific settings are ignored by Git. No per-node Python or launcher paths are required.

## Workflow

1. Connect `Read -> SAM3 Mask OFX -> Viewer`. Enter a Target such as `ball` or `red car`.
2. Set **Frame range**. Reset reads the connected input range and sets Reference frame to its first frame.
3. Run **Analyze** to generate and store the complete mask batch in RAM.
4. Inspect **View**: Plate, Mask (default), Plate + mask alpha, or Mask overlay. Overlay adds red at 50% inside the mask.
5. Choose **Motion / Reference frame**, then press **Solve**. It calculates tracks from stored masks and grayscale frames in a separate CPU process without SAM3 inference.
6. Change Smoothing or Crop margin and run Solve again. Crop size and Aspect ratio apply on the next Export without another Solve.
7. Select **Output** and **Export**. Native nodes are placed below SAM3.

| Output | Native result |
| --- | --- |
| Tracker | Tracker4 |
| Matchmove | Transform for an inserted image |
| Stabilize | Transform stabilizing the plate |
| Plate Stabilize Crop | Transform + Reformat extracting a stabilized crop |
| Generated Crop Matchmove | Transform + Reformat restoring an edited/generated crop to the plate |

Tracker and plate stabilization outputs connect to the source. Connect your insert to Matchmove or Generated Crop Matchmove. Use **View -> Mask** for the mask output.

## Reviewing and saving results

- Before Analyze, outside the stored range, and after reopening a script, the mask is black. Timeline playback never starts automatic inference.
- RAM pixels are **not serialized into `.nk`**. Analyze again after reopening Nuke or the script. Saved Solve data can still be exported, and exported native nodes work independently.
- Changing the plate or detection settings requires Analyze again. Motion, Reference, Smoothing, or Crop margin changes require only Solve. A range containing uncached frames requires Analyze first.
- Reduce the analysis range or input resolution if the RAM cache reaches capacity.
- BBox measures translation and uniform scale. Features can estimate rotation but may fall back to BBox; this is not a 3D pose, perspective, or camera solver.
- Use square pixels and correctly configured Read colorspace. The default input encoding is linear sRGB / Rec.709. Convert ACEScg before the node.

## Build from source

Optional: install Visual Studio 2022 C++ Build Tools and Windows SDK. CMake/Ninja are taken from Visual Studio or PATH.

```powershell
powershell -ExecutionPolicy Bypass -File install.ps1 -BuildFromSource
# Or build with native regression checks, then register that build:
powershell -ExecutionPolicy Bypass -File ofx/build.ps1 -RunTests
powershell -ExecutionPolicy Bypass -File install.ps1 -SkipBuild -BuildDirectory build
```

## Troubleshooting and checks

- Missing node: configure, run the **root** `install.ps1`, and restart Nuke.
- Unknown OFX plugin: check the bundled `SAM3Mask.ofx` and its adjacent local cfg; rerun installation.
- Wrong Python/model path or a moved checkout: rerun configure and install with valid paths.

Local Nuke checks covered RAM generation/playback, Solve, View/Overlay, and native export. The following repository tests require no sample media or model:

```powershell
.\.venv-sam3\Scripts\python.exe -m unittest discover -s tests -p 'test_*.py'
powershell -ExecutionPolicy Bypass -File ofx/build.ps1 -RunTests
```

Official [Meta SAM3](https://github.com/facebookresearch/sam3) runs in external Python. Its source and models are not redistributed here. OFX integration references MarigoldV2-Nuke / MoGe-nuke; attribution is retained in [licenses](licenses/) and the OpenFX headers.
