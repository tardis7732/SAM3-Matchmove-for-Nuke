# SAM3 Mask OFX for Nuke

**SAM3 Matchmove를 OFX 기반의 단일 노드 워크플로로 업데이트했습니다.**

[한국어](README.md) · [English](README.en.md) · [상세 사용법](docs/USAGE.md)

https://github.com/user-attachments/assets/72ade68a-44d2-4ede-9832-d4da3031d503

*10초 편집 데모: 공·차량·얼굴의 원본, SAM3 마스크, 안정화 Crop과 Nuke 노드 구성.*

> 데모는 이전 버전에서 제작되었습니다. OFX 업데이트로 영상 속 UI와 노드 구조는 현재 버전과 다를 수 있습니다. 설치와 사용 방법은 아래 최신 안내를 참고하세요.

대상을 텍스트로 지정하고 **Analyze로 RAM 마스크 생성 → Solve로 움직임 계산 → Export로 Nuke 기본 노드 생성**을 진행합니다.
마스크를 파일로 저장하고 별도 Read로 불러오는 과정 없이, 같은 SAM3 노드에서 Plate·Mask·Overlay를 확인합니다.

Windows x64용 OFX 바이너리를 포함합니다. **일반 설치에는 C++ 빌드 도구가 필요하지 않습니다.**
모델과 별도 Python 환경은 아래 순서대로 준비합니다.

## 주요 기능

- 네이티브 OFX가 Nuke의 입력 픽셀과 RAM 마스크 재생을 처리합니다.
- **Analyze**로 지정한 구간의 마스크를 생성하고, **Solve**로 저장된 마스크의 움직임을 계산합니다.
- Analyze는 Reference frame에서 선택한 객체 ID를 기준으로 앞뒤 프레임을 영상 추적합니다. Object index는 기준 프레임에서만 면적순으로 대상을 선택합니다.
- 분석 후 프레임 이동·재생은 RAM에 저장된 결과를 사용하며 SAM3를 재추론하지 않습니다.
- View는 **Plate / Mask / Plate + mask alpha / Mask overlay**, 기본은 Mask입니다.
- Motion은 **Translation / Translation + Scale / Translation + Scale + Rotation** 중에서 선택합니다. 크롭 출력도 선택한 움직임을 안정화하며, 출력 해상도는 Crop size를 유지합니다.
- Export는 **Tracker / Matchmove / Stabilize / Plate Stabilize Crop / Generated Crop Matchmove**를 노드 아래에 만듭니다.

## 처음 설치하기

### 1. 준비 사항

| 항목 | 준비할 내용 |
| --- | --- |
| OS / Nuke | Windows x64. 현재 OFX는 **Nuke 17.1v1**에서 검증했습니다. 다른 버전·호스트는 미검증입니다. |
| Python | **Python 3.12 이상**을 별도로 설치합니다. Nuke 내장 Python은 사용하지 않습니다. |
| Git | 아래 저장소와 공식 SAM3 소스를 받는 데 필요합니다. |
| GPU | SAM3 마스크 생성에는 NVIDIA CUDA GPU가 필요합니다. Solve는 CPU에서 실행합니다. |
| 모델 계정 | [공식 SAM3 모델 페이지](https://huggingface.co/facebook/sam3)에서 접근 승인을 받습니다. |
| 저장 공간 | Python 패키지, 공식 SAM3 소스와 모델을 다운로드할 공간이 필요합니다. |

setup 스크립트는 공식 SAM3를 고정 커밋으로 받고 PyTorch 2.10.0 / torchvision 0.25.0의 CUDA 12.8 wheel을 설치합니다.
패키지 목록은 [설치 스크립트](scripts/setup_sam3.ps1)와 [requirements](requirements-sam3-windows.txt)를 참고하세요.

### 2. 저장소 받기

PowerShell에서 실행합니다.

```powershell
git clone https://github.com/tardis7732/SAM3-Matchmove-for-Nuke.git
cd SAM3-Matchmove-for-Nuke
```

이후 명령은 모두 이 **저장소 루트**에서 실행합니다. 설치 후에도 폴더를 유지하세요.

### 3. 외부 Python / CUDA / SAM3 환경 만들기

아래 Python 경로는 실제 설치한 **Python 3.12+ 실행 파일 경로**로 바꾸세요.

```powershell
powershell -ExecutionPolicy Bypass -File scripts/setup_sam3.ps1 -PythonExe 'C:\Python312\python.exe'
```

`.venv-sam3`에 외부 실행 환경, `vendor/sam3`에 공식 SAM3 소스가 준비되고 `config.local.json`에 경로가 설정됩니다.
PyTorch와 SAM3는 외부 환경에서만 실행하며 Nuke 내장 Python에는 설치하지 않습니다.

### 4. 모델 로그인 / 다운로드

공식 모델 페이지에서 접근 승인을 받은 Hugging Face 계정으로 진행합니다.

```powershell
powershell -ExecutionPolicy Bypass -File scripts/auth_sam3.ps1
```

필요하면 로그인 프롬프트가 표시되고, 공식 `sam3.pt`를 `checkpoints` 폴더로 받습니다.
모델 접근 승인이 없으면 다운로드가 완료되지 않습니다.

### 5. OFX와 Python / 모델 연결

새로 만들어진 외부 Python으로 실행합니다.

```powershell
.\.venv-sam3\Scripts\python.exe tools/configure.py
```

이 명령은 `config.local.json`을 읽고 로컬 `config/frontend.json`과 `config/launcher.json`을 생성합니다.
Python 실행 파일이나 launcher 경로를 노드마다 입력할 필요가 없습니다.

### 6. Nuke 등록

```powershell
powershell -ExecutionPolicy Bypass -File install.ps1
```

포함된 Windows OFX 바이너리를 사용하고, 그 옆에 `sam3mask.cfg`를 생성합니다.
`.nuke/init.py`에 플러그인 경로를 등록하며 기존 init 파일은 변경 전에 백업합니다.
설치가 끝나면 **Nuke를 완전히 종료하고 다시 실행 → Tab → SAM3 Mask OFX**를 추가하세요.

환경·모델·설정 파일은 로컬에만 존재하며 Git에 포함되지 않습니다.

## 사용 순서

1. `Read → SAM3 Mask OFX → Viewer`로 연결하고 Target에 `ball`, `red car` 같은 대상을 입력합니다.
2. **Frame range**에서 구간을 정합니다. Reset은 입력 범위를 읽고 Reference frame을 시작 프레임으로 맞춥니다.
3. 대상이 보이는 **Reference frame**과 **Object index**를 정한 뒤 **Analyze**합니다. 같은 객체 ID를 앞뒤로 추적한 마스크를 RAM에 저장합니다.
4. **View**로 결과를 확인합니다. Mask overlay는 마스크 영역에 빨간색을 50%로 겹칩니다.
5. **Motion**을 선택하고 **Solve**를 실행합니다. 움직임은 Analyze에 사용한 Reference frame을 기준으로 계산합니다.
6. Smoothing이나 Crop margin을 바꾸면 Solve를 다시 실행합니다. Crop size / Aspect ratio 변경은 다음 Export에 적용됩니다.
7. **Output**에서 종류를 선택하고 **Export**합니다. 생성 노드는 SAM3 아래에 배치됩니다.

| Output | 생성 결과 |
| --- | --- |
| Tracker | Nuke Tracker4 |
| Matchmove | 삽입 영상용 Transform |
| Stabilize | 원본 안정화 Transform |
| Plate Stabilize Crop | 안정화 크롭용 Transform + Reformat |
| Generated Crop Matchmove | 생성·수정한 크롭을 원래 plate에 되돌리는 Transform + Reformat |

Tracker / Stabilize / Plate Stabilize Crop은 원본 입력에 연결됩니다.
Matchmove와 Generated Crop Matchmove에는 삽입 영상을 연결하세요. 마스크는 **View → Mask** 출력으로 사용합니다.

## 결과 확인과 저장

- Analyze 전과 저장된 범위 밖의 Mask는 검은색입니다. 프레임 이동만으로 자동 추론하지 않습니다.
- RAM 마스크는 `.nk`에 저장되지 않습니다. Nuke 종료·스크립트 재열기 후 마스크를 보려면 Analyze가 필요합니다.
- Solve 결과는 노드 데이터로 저장되므로 `.nk`에 저장한 움직임은 다시 Export할 수 있습니다. Export된 기본 노드는 독립적으로 사용합니다.
- 입력 영상·검출 설정·Reference frame 변경은 Analyze부터, Motion / Smoothing / Crop margin 변경은 Solve만 다시 실행합니다.
- 선택한 객체 ID가 없는 프레임은 빈 마스크로 남깁니다. 다른 대상의 면적이 커졌다는 이유로 선택을 바꾸지 않습니다. 가림이나 복잡한 움직임에 의한 모델의 추적 오류는 결과에서 확인하세요.
- RAM 용량이 부족하면 분석 구간을 줄이거나 입력 해상도를 낮추세요.
- Translation은 이동만, Translation + Scale은 이동과 균일 스케일, Translation + Scale + Rotation은 이동·균일 스케일·회전을 계산합니다. 영상 특징이 부족한 회전 추정 구간은 결과를 확인하세요. 3D pose·카메라·perspective solver는 아닙니다.

## 소스에서 빌드할 경우

일반 설치에서는 필요하지 않습니다. 직접 빌드하려면 **Visual Studio 2022 C++ Build Tools**와 Windows SDK가 필요합니다.
CMake와 Ninja는 Visual Studio에 포함된 도구 또는 PATH에서 찾습니다. 위의 configure 단계까지 완료한 뒤 실행합니다.

```powershell
powershell -ExecutionPolicy Bypass -File install.ps1 -BuildFromSource

# 네이티브 테스트까지 실행하려면
powershell -ExecutionPolicy Bypass -File ofx/build.ps1 -RunTests
powershell -ExecutionPolicy Bypass -File install.ps1 -SkipBuild -BuildDirectory build
```

## 문제 해결 / 검증

- **노드가 보이지 않음**: configure와 루트 `install.ps1`을 실행하고 Nuke를 재시작하세요.
- **Unknown OFX plugin**: `ofx/prebuilt/.../SAM3Mask.ofx`와 그 옆 cfg를 확인하고 루트 install을 다시 실행하세요.
- **Python / 모델 경로 오류**: 외부 Python과 공식 `sam3.pt`의 준비를 확인하고 configure를 다시 실행하세요. 설치 폴더를 이동한 경우도 configure / install이 필요합니다.
- **입력 색상 / 크기 문제**: Read 색공간을 올바르게 지정하세요. 기본 입력은 linear sRGB / Rec.709이며 ACEScg는 노드 앞에서 변환합니다. square pixel 입력을 사용하세요.

Nuke 17.1v1에서 RAM 생성·재생, Solve, View / Overlay, 기본 노드 Export를 검사했습니다.
아래 저장소 테스트에는 예제 영상이나 모델이 필요하지 않습니다.

```powershell
.\.venv-sam3\Scripts\python.exe -m unittest discover -s tests -p 'test_*.py'
powershell -ExecutionPolicy Bypass -File ofx/build.ps1 -RunTests
```

## 출처

공식 [Meta SAM3](https://github.com/facebookresearch/sam3)를 외부 Python에서 사용합니다. 모델과 공식 소스는 재배포하지 않습니다.
OFX 연결 구조는 MarigoldV2-Nuke / MoGe-nuke를 참고했습니다. 관련 고지와 라이선스는 [licenses](licenses/) 및 OpenFX 헤더에 보관합니다.
