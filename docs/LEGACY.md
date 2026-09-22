# 상세 사용 가이드

[프로젝트 소개로 돌아가기](../README.md)

## 설치 보충

### PowerShell 실행 정책

스크립트 실행이 차단될 때만 현재 PowerShell 창의 실행 정책을 변경하고 설치 명령을 다시 실행합니다.

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

Nuke에는 노드 UI를 등록하고, SAM3는 별도 Worker Python에 설치합니다. 검증한 조합은 Nuke 17.0v3의 Python 3.11.11과 외부 Python 3.12.9입니다. SAM3 text에는 CUDA GPU와 공식 `sam3.pt`가 필요합니다.

### 이미 SAM3 환경이 있다면

`scripts/install_nuke.ps1`로 Nuke에 등록한 뒤 노드의 **Environment → Worker Python / SAM3 checkpoint**에 기존 환경과 가중치를 지정합니다. Worker 환경에 [requirements-worker.txt](../requirements-worker.txt)의 패키지도 필요합니다. `Check Worker Environment`로 의존성을 확인하세요.

프로젝트 공통 기본값은 `config.example.json`을 `config.local.json`으로 복사해 설정할 수 있습니다. `python_exe`, `checkpoint`, `output_root`에 자신의 경로를 넣으며, 이 로컬 파일은 Git에서 제외됩니다. 설치 스크립트는 기존 로컬 설정을 유지합니다.

SAM3 환경 설치 스크립트는 공식 소스를 특정 커밋으로 고정하고 PyTorch 2.10.0 / torchvision 0.25.0의 CUDA 12.8 wheel과 Windows용 의존성을 설치합니다. 새 GPU나 다른 드라이버 조합에서의 호환성을 보장하는 목록은 아닙니다. SAM3.1용 모델 가중치를 SAM3용 `sam3.pt` 대신 사용하지 마세요.

### 모델 인증

`scripts/auth_sam3.ps1`은 외부 환경의 Hugging Face CLI로 로그인하고 모델을 받습니다. 먼저 [공식 모델 페이지](https://huggingface.co/facebook/sam3)에서 접근 승인이 필요합니다.

기존 `.env`를 사용하는 로컬 작업자는 `scripts/download_sam3.py`에 `--env-file`로 파일을 지정할 수도 있습니다. 토큰은 문서·Nuke knob·공유 설정에 넣지 않습니다. 다운로드 스크립트는 가중치의 SHA-256을 공식 메타데이터와 대조합니다.

### 설치 폴더를 옮겼다면

Nuke는 등록한 플러그인 폴더를 직접 참조합니다. `install_nuke.ps1`은 이미 등록된 항목을 유지하므로 다시 실행해도 기존 경로를 자동 변경하지 않습니다. 폴더를 옮겼다면 `.nuke/init.py`의 SAM3 등록 블록과 `config.local.json`의 경로를 새 위치에 맞춰 갱신해야 합니다.

## 입력과 프레임

**Plate(0)**에는 최종적으로 분석할 원본 또는 처리된 마지막 노드를 연결합니다. **Mask(1)**는 기존 전체 프레임 마스크를 입력하는 포트입니다.

`First / Last`는 Nuke 타임라인 기준입니다. `Reset`은 Plate 입력이 제공하는 출력 범위를 읽으며, Reference가 범위 밖에 있으면 안으로 조정합니다. `Use Current Frame as Reference`는 현재 프레임을 기준으로 설정합니다.

직접 지원하는 Read 입력은 파일을 전달합니다. 지원되지 않는 포맷이나 중간 처리 노드가 있으면 `SAM3_Input_Write`를 만들고 **Input requires conversion; Write created**를 표시합니다. 자동 렌더하지 않으므로 해당 Write를 렌더한 뒤 Analyze를 다시 누르세요. Mask도 변환이 필요하면 별도 Write를 준비합니다.

렌더된 입력이 준비돼 있으면 재사용합니다. **앞단을 수정했다면 Write를 다시 렌더해야 변경 내용이 반영됩니다.** 원본 Plate 연결은 유지합니다. Retime 뒤에 연결했다면 결과 키도 리타임 이후 타임라인에 맞춰 생성됩니다.

직접 파일 입력은 파일의 RGB 값을 읽으며 Nuke의 Read 색변환이나 Viewer LUT를 재현하지 않습니다. Nuke에서 처리한 픽셀을 분석하려면 해당 처리 노드를 연결하고 준비된 PNG Write를 사용하세요. 분석용 PNG는 sRGB 8비트 RGB이며, 합성용 원본 Plate는 기존 채널과 색공간을 유지합니다.

## 대상과 움직임 설정

| 설정 | 의미 |
| --- | --- |
| Target text | 기준 프레임에서 찾을 대상. 기본 `face`는 실제 얼굴을 찾는 문구이며 자동 대상 선택이 아닙니다. |
| Object index | 기준 프레임 검출 크기 순서. `0`이 가장 큰 대상이며 선택된 객체 ID를 유지합니다. |
| Detection threshold | 검출 점수의 최소값. 매 프레임 트래킹 정확도를 나타내는 수치는 아닙니다. |
| BBox position + scale | 마스크 경계의 중심과 크기로 이동·균일 스케일을 계산합니다. 회전은 0입니다. |
| Features: position + scale + rotation | 마스크 내부의 영상 특징점을 추적하고 2D 이동·균일 스케일·회전을 추정합니다. |
| Smoothing window | 대칭적인 시간 평균 창. `1`은 끔이며 홀수를 사용합니다. 값을 키우면 움직임도 완만해집니다. |
| Crop margin | 추적 영역 주변의 여백. `1.2`는 20% 여백이며 변경 후 재분석이 필요합니다. |
| Fixed crop size | 전체 구간에서 같은 Crop 크기 사용. 끄면 대상 크기의 변화를 따릅니다. 변경 후 재분석이 필요합니다. |

Features는 optical flow로 대응점을 찾고 RANSAC으로 변환을 계산합니다. SAM3가 직접 회전값을 출력하는 구조가 아닙니다. 추정 실패 시 bbox 보조 추정을 사용하고 결과에 기록합니다. 기준 프레임에서 대상이 검출되지 않으면 안내 후 분석을 중단합니다.

## 기존 마스크 분석

1. 전체 프레임 마스크를 Mask 입력에 연결합니다.
2. `Mask source = Input mask`를 선택합니다.
3. RGBA/Roto는 `alpha`, 흑백 RGB 마스크는 `red` 등 실제 값이 들어 있는 채널을 선택합니다.
4. Analyze로 움직임을 계산하고 Result JSON을 만든 뒤 Export합니다.

이 경로는 SAM3 추론을 실행하지 않습니다. NumPy·OpenCV·Pillow가 있는 외부 Python이 필요합니다. Plate와 마스크의 해상도·픽셀 좌표·프레임이 일치해야 하며, anamorphic 입력은 먼저 정사각형 픽셀로 맞춥니다.

분석 결과 마스크는 전체 Plate 크기의 RGBA로 저장되고 Mask 입력에 연결됩니다. **Mask source 자체는 바뀌지 않습니다.** 반면 **Output → Mask Read → Export**는 Read만 별도로 생성하며 기존 연결을 바꾸지 않습니다.

## Crop과 생성 결과 복원

`Plate Stabilize Crop`으로 원본의 대상 주변을 추출하고, 같은 크기의 생성·수정 결과를 `Generated Crop Matchmove`의 첫 Transform에 연결합니다. 뒤의 Reformat은 원래 Plate 해상도를 복원합니다.

- 기본 크기: `720×720`, `1:1`, Lock ratio 켜짐.
- 화면비: `1:1`, `16:9`, `9:16`, `4:3`, `3:4`, `3:2`, `2:3`, `2.39:1`, `Custom`.
- 비율 잠금: Width를 바꾸면 Height가, Height를 바꾸면 Width가 같이 조정됩니다.
- 잠금 해제: 폭과 높이를 독립적으로 입력할 수 있습니다.

직사각형 출력은 대상 중심 주변을 화면비에 맞춰 넓히고 균일하게 스케일합니다. Width/Height는 Crop 출력에만 적용되며 마스크의 전체 해상도는 바꾸지 않습니다. **크기만 변경했다면 다시 Export하면 됩니다.** 분석에 쓰이는 Motion·Crop margin·Fixed crop size를 바꿨다면 재분석하세요.

## 결과 저장과 재사용

`Output folder` 아래 실행별 폴더에 다음 파일이 저장됩니다.

```text
run-folder/
  job.json
  result.json
  tracks.csv
  progress.json
  worker.log
  masks/
    mask.001001.png
    ...
```

`Result JSON`에 기존 `result.json`을 지정하면 해당 데이터를 사용해 다시 Export할 수 있습니다. 움직임 출력은 JSON을 사용하며 마스크를 다시 분석하지 않습니다. Mask Read를 복구하려면 JSON이 가리키는 마스크 시퀀스도 남아 있어야 합니다.

각 Export는 선택한 노드만 담은 `.nk`도 결과 폴더에 저장합니다. 기존 파일이 있으면 번호를 붙여 보존합니다. 생성한 Tracker와 Transform은 기본 Nuke 노드이며, Tracker 연동 Transform은 해당 Tracker도 같이 있어야 합니다.

Result JSON과 작업용 `.nk`에는 로컬 파일 경로가 들어갈 수 있습니다. 이 저장소는 로컬 결과·예제·로그·환경 설정을 Git에서 제외합니다. 다른 사람에게 분석 결과를 전달할 때는 필요한 파일과 경로를 별도로 정리하세요.
