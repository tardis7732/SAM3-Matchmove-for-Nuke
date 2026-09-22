# SAM3 Mask OFX 사용 가이드

[처음 설치하기](../README.md#처음-설치하기) · [English](../README.en.md)

## 설치 구조

처음 내려받은 저장소에서 README의 순서를 따릅니다.

1. `scripts/setup_sam3.ps1`: 외부 `.venv-sam3` Python 환경과 공식 SAM3 소스를 준비합니다.
2. `scripts/auth_sam3.ps1`: Hugging Face 로그인 후 공식 `sam3.pt`를 받습니다. 모델 접근 승인이 먼저 필요합니다.
3. `.venv-sam3\Scripts\python.exe tools/configure.py`: Python / 모델과 OFX를 연결합니다.
4. 루트 `install.ps1`: 포함된 Windows OFX 바이너리를 Nuke에 등록합니다. 마지막에 Nuke를 재시작합니다.

Nuke 내장 Python과 `.venv-sam3`는 다른 환경입니다. PyTorch와 SAM3는 외부 환경에서만 실행합니다.

| 파일 / 폴더 | 역할 |
| --- | --- |
| `.venv-sam3/` | 외부 Python / CUDA 패키지 |
| `vendor/sam3/` | setup이 받는 공식 SAM3 소스 |
| `checkpoints/sam3.pt` | auth가 받는 모델 |
| `config.local.json` | setup이 생성하는 로컬 Python / 모델 경로 |
| `config/frontend.json` | Nuke UI가 사용할 경로 |
| `config/launcher.json` | 외부 GPU 엔진 실행 명령 |
| `ofx/prebuilt/` | 포함된 Windows x64 OFX 바이너리 |
| OFX 옆 `sam3mask.cfg` | install이 생성하는 실행 설정 |
| `.nuke/init.py` | Nuke의 플러그인 경로 등록 |

설치 폴더를 유지하세요. 생성된 설정들은 컴퓨터별 파일이며 Git에서 제외합니다.
일반 설치는 Visual Studio가 필요하지 않습니다. 직접 컴파일할 경우만 README의 소스 빌드를 사용합니다.

## 기본 연결과 입력

`Read → SAM3 Mask OFX → Viewer`로 연결합니다. 기존 합성 노드의 결과도 입력으로 연결할 수 있습니다.
Read가 직접 읽지 못하는 파일은 Nuke에서 읽을 수 있는 형식으로 먼저 준비해야 합니다.
OFX는 Nuke가 제공하는 픽셀로 Analyze하므로 분석용 Write·입력 시퀀스를 자동 생성하지 않습니다.

기본 입력은 linear sRGB / Rec.709입니다. Read 색공간을 올바르게 지정하고 ACEScg는 SAM3 앞에서 변환하세요.
입력 RGB는 검출 전에 0–1로 제한합니다. square pixel 입력이 필요합니다.

## 마스크 생성

- **Target**: 찾을 대상을 텍스트로 입력합니다. 예: `ball`, `red car`.
- **Confidence**: 검출을 채택하는 최소 점수입니다. 높이면 오검출이 줄지만 대상이 빠지는 프레임이 생길 수 있습니다.
- **Object index**: 면적순 검출 영역 번호입니다. 0은 가장 큰 영역입니다. 합집합이나 -1 모드는 없습니다.
- **Frame range**: Analyze 구간입니다. Reset은 입력 범위를 읽고 Reference frame을 시작 프레임으로 맞춥니다.
- **Analyze**: 마스크와 Solve용 흑백 영상을 RAM에 생성합니다. 입력과 검출 설정이 일치하는 엔진 캐시는 재사용할 수 있습니다.

Analyze 전에는 Mask가 검은색입니다. 완료 후 프레임 이동·재생은 저장된 결과를 사용합니다.
입력 영상이나 검출 설정을 바꾸면 Analyze로 갱신하세요. 원본 픽셀 변경을 매 프레임 자동 검사하지 않습니다.

## View

| 선택 | 출력 |
| --- | --- |
| Plate | 원본 영상 |
| Mask | 마스크. 새 노드 기본값 |
| Plate + mask alpha | 원본 RGB에 마스크 alpha 적용 |
| Mask overlay | 마스크 영역에 빨간색을 50%로 겹친 원본 |

Plate + mask alpha는 RGB를 premultiply하지 않습니다. 필요한 경우 뒤에 Premult를 연결하세요.
Mask overlay는 원본 alpha를 유지하며 alpha 없는 RGB 입력도 지원합니다.
View에 안정화 항목은 없습니다. 안정화와 크롭은 Export로 만듭니다.

## Solve와 크롭

**Motion / Reference frame / Solve**가 한 줄에 있습니다. Analyze 완료 후 Solve를 실행하세요.
Solve는 저장된 마스크와 흑백 영상만 CPU로 처리하며 SAM3를 다시 호출하지 않습니다.

- **BBox position + scale**: 마스크 경계로 이동과 균일 스케일을 계산합니다. 회전은 측정하지 않습니다.
- **Features**: 마스크 내부 영상 특징으로 이동·스케일·회전을 계산합니다. 실패 시 BBox로 대체될 수 있습니다.
- **Reference frame**: 움직임의 기준 프레임이며 대상 마스크가 있어야 합니다.
- **Smoothing**: 양의 홀수 프레임 윈도 크기입니다. 1은 추가 평활화 없음입니다.
- **Crop margin**: 마스크 경계에 대한 여유 배율이며 1 이상을 사용합니다.

Smoothing과 Crop margin은 각각 한 줄입니다. 값을 바꾸면 Solve를 다시 실행합니다.

**Crop size [가로] × [세로] / Aspect ratio / Lock ratio**는 한 줄입니다.
기본은 720 × 720 / 1:1이며 화면비 프리셋 변경 시 Height를 유지합니다.
화면비는 1:1, 16:9, 9:16, 4:3, 3:4, 3:2, 2:3, 2.39:1, Custom을 지원합니다.
크기와 화면비는 다음 Export에 적용되므로 Solve를 다시 실행할 필요가 없습니다.

Input encoding / Refresh mask / Engine status / Stop engine 버튼과 Adjustments 탭은 메인 Properties에서 숨깁니다.

## Export

Solve 후 Output에서 종류를 선택하고 Export합니다.

- Tracker: Nuke Tracker4.
- Matchmove: 삽입 영상용 Transform.
- Stabilize: 원본 안정화 Transform.
- Plate Stabilize Crop: 안정화 크롭을 만드는 Transform + Reformat.
- Generated Crop Matchmove: 생성·수정한 크롭을 원래 plate 좌표로 되돌리는 Transform + Reformat.

생성 노드는 SAM3 아래에 배치하며, 이전에 내보낸 노드와 겹치지 않게 추가됩니다.
Tracker / Stabilize / Plate Stabilize Crop은 원본 입력에 연결됩니다.
Matchmove와 Generated Crop Matchmove의 비어 있는 입력에는 삽입 영상을 연결합니다.
크롭 복원용 영상은 Export 당시 Crop size와 같은 크기를 사용하세요.

Export된 곡선은 그 시점의 복사본입니다. SAM3 설정 변경을 자동으로 따라가지 않으므로 변경 후에는 새로 Export하세요.
마스크는 별도 Export 없이 View의 Mask 출력을 사용합니다.

## 저장과 재시작

자동 마스크 파일·분석용 입력 시퀀스·결과 JSON/CSV를 만들지 않습니다. 마스크 픽셀은 RAM에만 있습니다.
스크립트를 저장하면 Solve 데이터는 노드 안에 포함되지만 마스크 픽셀은 포함되지 않습니다.
Nuke를 종료하거나 스크립트를 다시 열면 마스크를 위해 Analyze가 필요합니다.
이미 저장한 Solve 데이터는 Export할 수 있으며, 내보낸 Nuke 기본 노드는 별도로 동작합니다.

외부 GPU 엔진은 유휴 상태에서 종료될 수 있습니다. 같은 Nuke 세션의 RAM 재생과 CPU Solve는 계속 사용할 수 있고,
다음 Analyze가 엔진을 다시 시작합니다. RAM 한도에 도달하면 더 짧은 구간이나 작은 입력을 사용하세요.

## 제한

프레임마다 독립 검출하므로 객체 ID를 고정하지 않습니다. 여러 대상의 크기 순위가 바뀌면 index의 대상도 달라질 수 있습니다.
없는 마스크나 특징 추적 실패 구간은 보간·유지·BBox 대체를 포함할 수 있으므로 Export 전에 확인하세요.
이 도구는 2D 대상 매치무브이며 3D 카메라·pose·perspective solve가 아닙니다.
