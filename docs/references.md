# 참고한 프로젝트와 검증 범위

출처 확인일: 2026-09-07. 아래 외부 프로젝트에 관한 내용은 각 저자 저장소의 문서 및 소스 검토 결과이며,
해당 플러그인을 직접 설치해 실행한 검증 기록은 아닙니다.
이 패키지는 공식 SAM 3 Python API를 호출하는 별도 어댑터를 제공합니다.

## 기반 워크플로우

- [tardis7732/wan-animate-sam3-head-for-nuke](https://github.com/tardis7732/wan-animate-sam3-head-for-nuke)
- 확인한 체크아웃: `83aea32926097833c8328382ee32d3f5f194d650`
- [해당 버전 README](https://github.com/tardis7732/wan-animate-sam3-head-for-nuke/blob/83aea32926097833c8328382ee32d3f5f194d650/README.md)
- [ComfyUI 워크플로우](https://github.com/tardis7732/wan-animate-sam3-head-for-nuke/blob/83aea32926097833c8328382ee32d3f5f194d650/workflows/wan-animate-sam3-head-for-nuke.json)
- [Nuke 템플릿](https://github.com/tardis7732/wan-animate-sam3-head-for-nuke/blob/83aea32926097833c8328382ee32d3f5f194d650/nuke/wan-animate-sam3-head-for-nuke.nk)

같은 저자가 만든 위 워크플로우에서 SAM 3으로 머리 대상을 찾고 bbox 중심/크기를 안정화하여 정방형 Face/Mask 소스를
원본 플레이트 위치에 배치하는 흐름을 참고했습니다. 기존 템플릿은 Transform을 이용한
2D 배치이며, 이 프로젝트도 2D matchmove 데이터를 다룹니다.

## 기존 Nuke SAM 프로젝트

| 프로젝트 | 확인한 내용 | 참고한 부분 |
| --- | --- | --- |
| [Code2Collapse/Nuke-Sam3-Gizmo](https://github.com/Code2Collapse/Nuke-Sam3-Gizmo) | 공개 소스가 존재합니다. README는 Nuke 16+/Python 3.11을 명시하며 Windows에서는 별도 `python_packages`를 Nuke Python 경로에 넣습니다. | 연결된 Nuke 입력을 임시 PNG로 렌더하고, 추론한 마스크를 내부 Read에 넣어 alpha로 반환하는 구조. |
| [smert999/NukeOnyxSam3](https://github.com/smert999/NukeOnyxSam3) | 확인 당시 저장소와 raw README 주소가 404였습니다. | 현재 설치 가능 여부 및 코드를 검증하지 못해 구현 기반으로 사용하지 않았습니다. |
| [MaikiOS/NukeSamurai-Windows](https://github.com/MaikiOS/NukeSamurai-Windows) | README가 SAM 2.1 기반 프로젝트이며 NukeOnyxSam3으로 이전했다고 설명합니다. 외부 subprocess 사용도 설명합니다. | Nuke/PyTorch 환경을 분리한다는 구조상 참고. |

Code2Collapse 저장소에서 확인한 `main` SHA는 `10787328981c88d6b550beb17420c4b6268e599e`입니다.
[실제 inference.py](https://github.com/Code2Collapse/Nuke-Sam3-Gizmo/blob/10787328981c88d6b550beb17420c4b6268e599e/inference.py)는
SAM 3 텍스트 추론에 자체 `Sam3Processor`를 사용하고, SAM 2 모드에서만 Grounding DINO를 사용합니다.
현행 코드의 refinement는 MatAnyone2이며 README의 ViTMatte 설명과 차이가 있습니다.
확인한 주요 경로는 프레임별 이미지 추론과 alpha 출력으로, 본 프로젝트의 공식 video predictor와는 다릅니다.
라이선스는 [Apache-2.0](https://github.com/Code2Collapse/Nuke-Sam3-Gizmo/blob/10787328981c88d6b550beb17420c4b6268e599e/LICENSE)입니다.

NukeOnyxSam3의 과거 공개 사실은 [저자가 남긴 Hugging Face 토론](https://huggingface.co/facebook/sam3/discussions/46)에서도 확인됩니다.
이는 현재 다운로드/실행 가능하다는 근거는 아닙니다.

## 공식 Meta SAM 3 API

저장소: [facebookresearch/sam3](https://github.com/facebookresearch/sam3)

확인한 `main` SHA: `660a5e9e1b8b4c02c0ad97229b88a09a6e4ff5b7` (2026-08-26).
GitHub 커밋 API로 SHA를 확인하고 동일 버전의 세션 디스패치 및 출력 소스도 대조했습니다.

| 소스 | 확인한 계약 |
| --- | --- |
| [README](https://github.com/facebookresearch/sam3/blob/660a5e9e1b8b4c02c0ad97229b88a09a6e4ff5b7/README.md) | Python 3.12+, PyTorch 2.7+, CUDA 12.6+ 및 체크포인트 접근 승인/인증. |
| [model_builder.py](https://github.com/facebookresearch/sam3/blob/660a5e9e1b8b4c02c0ad97229b88a09a6e4ff5b7/sam3/model_builder.py) | `build_sam3_video_predictor`와 SAM 3 `sam3.pt`를 사용. SAM 3.1은 별도 모델 계열. |
| [sam3_video_predictor.py](https://github.com/facebookresearch/sam3/blob/660a5e9e1b8b4c02c0ad97229b88a09a6e4ff5b7/sam3/model/sam3_video_predictor.py) | 생성자는 `checkpoint_path`, `gpus_to_use`, `compile`, `async_loading_frames`를 받음. 내부 모델이 CUDA로 이동함. |
| [sam3_base_predictor.py](https://github.com/facebookresearch/sam3/blob/660a5e9e1b8b4c02c0ad97229b88a09a6e4ff5b7/sam3/model/sam3_base_predictor.py) | `start_session` → `add_prompt` → `propagate_in_video` → `close_session` 및 `shutdown`. 스트림 요청 키는 `start_frame_index`. |
| [sam3_video_inference.py](https://github.com/facebookresearch/sam3/blob/660a5e9e1b8b4c02c0ad97229b88a09a6e4ff5b7/sam3/model/sam3_video_inference.py) | 출력 ID/점수/mask 및 정규화 XYWH box 구성. `out_probs`는 객체의 최초 탐지 점수에서 나옴. |
| [io_utils.py](https://github.com/facebookresearch/sam3/blob/660a5e9e1b8b4c02c0ad97229b88a09a6e4ff5b7/sam3/model/io_utils.py) | 이미지 폴더 입력과 숫자 stem 정렬, PNG 지원. |
| [sam3_image_processor.py](https://github.com/facebookresearch/sam3/blob/660a5e9e1b8b4c02c0ad97229b88a09a6e4ff5b7/sam3/model/sam3_image_processor.py) | 이미지 API의 bbox는 픽셀 XYXY로 video API와 다름. |

모델과 런타임 패키지는 이 배포에 포함하지 않습니다. `Checkpoint`가 비어 있으면 공식
빌더가 [facebook/sam3](https://huggingface.co/facebook/sam3)의 승인된 로컬 캐시를 조회하고,
필요하면 Hugging Face에서 다운로드합니다. 이미 받은 공식 `sam3.pt`의 로컬 경로도 지정할 수 있습니다.
접근 승인은 모델 페이지에서, 인증은 **외부 worker Python 환경**의 `hf auth login`으로 합니다.
Meta 코드/가중치는 [SAM License](https://github.com/facebookresearch/sam3/blob/660a5e9e1b8b4c02c0ad97229b88a09a6e4ff5b7/LICENSE)를 따릅니다.

## 어댑터의 프레임/좌표 처리

- 입력 `job.frames` 순서와 원래 Nuke 프레임 번호의 매핑을 보관합니다. 직접 읽을 수 있는 이미지 시퀀스는 원본 인코딩을 유지한 채 임시 폴더에 링크하거나 복사하고, 파일명을 `00000000`부터 순서대로 지정합니다. 확장자는 원본을 유지하되 `.tif`는 `.tiff`로 정규화합니다.
- 동영상 입력은 지정된 프레임을 디코딩해 PIL 이미지 목록으로 공식 predictor에 전달합니다. 이 경로에서는 중간 PNG 시퀀스를 만들지 않습니다. Nuke에서 입력 변환이 필요한 경우에는 사용자가 생성된 Write를 렌더한 뒤 그 이미지 시퀀스를 분석합니다.
- SAM의 `frame_index`는 0부터 시작합니다. Nuke의 시작 프레임이 1001이거나 음수여도 결과 키/파일명에는 원래 번호를 사용합니다.
- `out_binary_masks`는 좌상단 기준 `[객체, 높이, 너비]` boolean 데이터입니다. 입력 전체 해상도와 일치하는지 검증한 뒤 각 채널에 동일한 0/255 마스크 값을 넣은 RGBA PNG로 저장합니다.
- video `out_boxes_xywh`는 정규화된 좌상단 `x,y,width,height`입니다. 이미지 API의 `boxes`는 픽셀 `x0,y0,x1,y1`이므로 서로 대체하지 않습니다.
- 기준 프레임의 유효 마스크를 면적 내림차순으로 정렬해 `Object index`를 적용합니다. 동률은 객체 ID 오름차순으로 정렬합니다.
- 선택한 객체 ID를 양방향 전파 내내 유지합니다. 다른 객체가 더 크거나 점수가 높아져도 바꾸지 않습니다.
- 기준 마스크는 선택 때의 결과를 유지합니다. 대상 누락, 낮은 탐지 점수, 반환되지 않은 프레임은 빈 마스크와 `detected=false`로 남깁니다.
- `score`는 SAM 탐지 점수입니다. 공식 video 출력에서는 최초 탐지 점수를 재사용하므로 매 프레임의 추적 정확도나 가림 신뢰도라고 해석하면 안 됩니다.
- 모델 추론 오류나 취소 때도 스트림/세션 종료와 `shutdown()`을 시도하며, 추론용 임시 링크/복사본만 정리합니다.
