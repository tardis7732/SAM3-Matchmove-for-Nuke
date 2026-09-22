# SAM3 Mask OFX for Nuke

**SAM3 Matchmove를 OFX 기반의 단일 노드 워크플로로 업데이트했습니다.**

[한국어](README.md) · [English](README.en.md) · [상세 사용법](docs/USAGE.md)

대상을 텍스트로 지정하고 **Analyze로 RAM 마스크 생성 → Solve로 움직임 계산 → Export로 Nuke 기본 노드 생성**을 진행합니다.
마스크를 파일로 저장하고 별도 Read로 불러오는 과정 없이, 같은 SAM3 노드에서 Plate·Mask·Overlay를 확인합니다.

Windows x64용 OFX 바이너리를 포함합니다. **일반 설치에는 C++ 빌드 도구가 필요하지 않습니다.**
모델과 별도 Python 환경은 아래 순서대로 준비합니다. 예제 영상·이미지·작업 `.nk`·모델은 포함하지 않습니다.

## OFX 업데이트 내용

- 네이티브 OFX가 Nuke의 입력 픽셀과 RAM 마스크 재생을 처리합니다.
- **Analyze / Solve 분리**: Motion이나 Reference frame을 바꾸면 저장된 마스크로 Solve만 다시 합니다.
- 자동 마스크 PNG·분석용 입력 시퀀스·결과 JSON/CSV를 만들지 않습니다.
- 분석 후 프레임 이동·재생은 RAM에 저장된 결과를 사용하며 SAM3를 재추론하지 않습니다.
- View는 **Plate / Mask / Plate + mask alpha / Mask overlay**, 기본은 Mask입니다.
- Export는 **Tracker / Matchmove / Stabilize / Plate Stabilize Crop / Generated Crop Matchmove**를 노드 아래에 만듭니다.
- 내부 그래프는 View에 필요한 노드만 유지하고, 안정화·크롭 노드는 Export할 때 생성합니다.

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
2. **Frame range [시작] [끝] Reset Analyze**에서 구간을 정합니다. Reset은 입력 범위를 읽고 Reference frame을 시작 프레임으로 맞춥니다.
3. **Analyze**로 해당 구간의 마스크를 RAM에 만듭니다. 처음에는 모델 로딩 시간이 추가됩니다.
4. **View**로 결과를 확인합니다. Mask overlay는 마스크 영역에 빨간색을 50%로 겹칩니다.
5. **Motion / Reference frame / Solve**에서 모드와 기준을 정하고 **Solve**를 실행합니다.
6. Smoothing이나 Crop margin을 바꾸면 Solve를 다시 실행합니다. Crop size / Aspect ratio 변경은 다음 Export에 적용됩니다.
7. **Output**에서 종류를 선택하고 **Export**합니다. 생성 노드는 SAM3 아래에 배치됩니다.

Smoothing과 Crop margin은 Motion 아래에 각각 한 줄씩 있습니다. Crop size / Aspect ratio / Lock ratio는 그 아래 한 줄입니다.
기본 크기는 **720 × 720, 1:1**이며 화면비 프리셋은 Height를 유지합니다.
Confidence는 최소 검출 점수, Object index는 면적순 대상 번호입니다. 두 값은 슬라이더 없는 숫자 입력입니다.
Input encoding·Refresh mask·Engine status·Stop engine / free GPU와 별도 Adjustments 탭은 메인 UI에서 숨깁니다.

| Output | 생성 결과 |
| --- | --- |
| Tracker | Nuke Tracker4 |
| Matchmove | 삽입 영상용 Transform |
| Stabilize | 원본 안정화 Transform |
| Plate Stabilize Crop | 안정화 크롭용 Transform + Reformat |
| Generated Crop Matchmove | 생성·수정한 크롭을 원래 plate에 되돌리는 Transform + Reformat |

Tracker / Stabilize / Plate Stabilize Crop은 원본 입력에 연결됩니다.
Matchmove와 Generated Crop Matchmove에는 삽입 영상을 연결하세요. 마스크는 **View → Mask** 출력으로 사용합니다.

## RAM 동작과 제한

- Analyze 전과 저장된 범위 밖의 Mask는 검은색입니다. 프레임 이동만으로 자동 추론하지 않습니다.
- RAM 마스크는 `.nk`에 저장되지 않습니다. Nuke 종료·스크립트 재열기 후 마스크를 보려면 Analyze가 필요합니다.
- Solve 결과는 노드 데이터로 저장되므로 `.nk`에 저장한 움직임은 다시 Export할 수 있습니다. Export된 기본 노드는 독립적으로 사용합니다.
- 입력 영상·검출 설정 변경은 Analyze부터, Motion / Reference / Smoothing / Crop margin 변경은 Solve만 다시 실행합니다.
- GPU 엔진이 종료되어도 같은 Nuke 세션의 RAM 재생과 CPU Solve는 가능합니다. Nuke 자체의 GPU 메모리까지 해제하는 기능은 아닙니다.
- 기본 예산은 엔진 캐시 512 MiB, 비트 압축 재생 마스크 구간당 512 MiB, 압축 Python Solve 데이터 전체 512 MiB입니다. 모델·압축 해제·계산용 임시 메모리는 별도입니다. 한도를 넘으면 오류를 표시하며 디스크로 저장하지 않습니다.
- OFX는 프레임마다 독립 검출합니다. **동일 객체 ID를 영상 전체에 고정하지 않습니다.** 면적 순위가 바뀌면 Object index의 대상도 달라질 수 있습니다.
- BBox는 위치와 균일 스케일을 계산합니다. Features는 회전도 추정하지만 실패하면 BBox로 대체될 수 있습니다. 3D pose·카메라·perspective solver는 아닙니다.

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
- **Properties 오류 이후**: 작업을 저장하고 Nuke를 완전히 종료한 뒤 다시 실행하세요. 현재 UI는 기본 버튼 표시를 사용하고 노브 이관 전에 열린 패널을 닫습니다.

Nuke 17.1v1에서 RAM 생성·재생, 엔진 종료 후 Solve, View / Overlay, 기본 노드 Export, Properties 열기·닫기를 검사했습니다.
아래 저장소 테스트에는 예제 영상이나 모델이 필요하지 않습니다.

```powershell
.\.venv-sam3\Scripts\python.exe -m unittest discover -s tests -p 'test_*.py'
powershell -ExecutionPolicy Bypass -File ofx/build.ps1 -RunTests
```

## 출처

공식 [Meta SAM3](https://github.com/facebookresearch/sam3)를 외부 Python에서 사용합니다. 모델과 공식 소스는 재배포하지 않습니다.
OFX 연결 구조는 MarigoldV2-Nuke / MoGe-nuke를 참고했습니다. 관련 고지와 라이선스는 [licenses](licenses/) 및 OpenFX 헤더에 보관합니다.
