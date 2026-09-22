# 참고 자료

## 공식 SAM3

- [Meta SAM3 저장소](https://github.com/facebookresearch/sam3)
- [공식 모델 및 접근 신청](https://huggingface.co/facebook/sam3)
- [설치 스크립트가 사용하는 소스 버전](https://github.com/facebookresearch/sam3/tree/660a5e9e1b8b4c02c0ad97229b88a09a6e4ff5b7)
- [SAM3 모델 빌더](https://github.com/facebookresearch/sam3/blob/660a5e9e1b8b4c02c0ad97229b88a09a6e4ff5b7/sam3/model_builder.py)
- [SAM3 이미지 프로세서](https://github.com/facebookresearch/sam3/blob/660a5e9e1b8b4c02c0ad97229b88a09a6e4ff5b7/sam3/model/sam3_image_processor.py)
- [SAM 라이선스](https://github.com/facebookresearch/sam3/blob/660a5e9e1b8b4c02c0ad97229b88a09a6e4ff5b7/LICENSE)

환경과 모델 준비 방법은 [설치 안내](../README.md#처음-설치하기)를 참고하세요.

## Nuke / OFX

대상 중심의 크롭과 원본 위치 복원 흐름은 [wan-animate-sam3-head-for-nuke](https://github.com/tardis7732/wan-animate-sam3-head-for-nuke)를 참고했습니다.
OFX 연결 구조는 MarigoldV2-Nuke / MoGe-nuke를 참고했으며, 관련 고지와 라이선스는 [licenses](../licenses/)에 보관합니다.
사용하는 OpenFX 헤더의 라이선스는 [여기](../ofx/include/openfx/LICENSE.md)에 있습니다.

## 구현과 사용법

- [Nuke 노드와 Export 연결](../nuke/sam3_unified.py)
- [RAM Analyze / Solve](../nuke/sam3_memory.py)
- [SAM3 마스크 생성](../daemon/backend.py)
- [마스크 기반 움직임 계산](../daemon/memory_tracking.py)
- [상세 사용법](USAGE.md)