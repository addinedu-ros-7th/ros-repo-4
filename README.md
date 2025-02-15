<p align="center">
  <img width="916" alt="image" src="https://github.com/user-attachments/assets/93aa6310-9032-4851-b83d-fbc23b61d00a" />
</p>

# 00. 통합 영상 및 발표 자료
## 01. 통합 영상
[![ ](https://img.youtube.com/vi/GnMYARfWPEs/0.jpg)](https://youtube.com/shorts/GnMYARfWPEs?si=HJs3EaWiIBXYDwoN)
## 02. 발표 자료
[파이토끼 발표 자료](https://docs.google.com/presentation/d/13RBxSe_4TvYX2x6u4SbVQMm9CwZP5-0HAGWoL_2vUaw/edit#slide=id.g33465be63bc_0_94)

# 01. 프로젝트 소개
## 01-1. 프로젝트 목표
- 화재 및 재난 현장에서 **장애물 및 위험도 분석**, **실시간 구조 지도** **생성**, **구조 대상 탐지 및 대피 유도**를 수행하는 자율주행 로봇 시스템 개발

## 01-2. 주제 선정 배경
- **구조 대원의 안전** : 연기, 화염, 건물 붕괴의 위험 속에서 직접 진입.
- **정보 부족** : 재난 현장의 구조 정보와 위험 요소 실시간 상황 파악의 어려움.
- **대피 경로 부재** : 대피 경로에 대해 명확한 가이드 제공이 어려움.

## 01-3. 팀원 및 역할
| **이름** | **역할** |
|---|---|
| **김소영** <br> **(팀장)** | • 기획 및 설계 <br> • 통신 시스템 설계 <br> • Control GUI 디자인 및 개발 |
| **김재현** | • 딥러닝 모델 개발 <br> • 모델 배포 및 운영 <br> • AI Manager 개발 |
| **임주원** | • SLAM 및 Nav2 기반 주행 구현 <br> • ROS PKG 생성 및 통신 구현 <br> • 로봇 경로 생성 알고리즘 개발 |
| **함동균** | • Control Server 개발 <br> • 로봇 경로 생성 알고리즘 개발 <br> • 다중 로봇 제어 구현 |


## 01-4. 기술 스택
| **항목** | **내용** |
|:--:|---|
| **개발 환경** | ![Ubuntu](https://img.shields.io/badge/Ubuntu-24.04-E95420?logo=ubuntu&logoColor=white) |
| **개발 언어** | ![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white) ![SQL](https://img.shields.io/badge/SQL-4479A1?logo=postgresql&logoColor=white) |
| **개발 도구** | ![ROS2](https://img.shields.io/badge/ROS2-Jazzy-22314E?logo=ros&logoColor=white) ![PyQt](https://img.shields.io/badge/PyQt-41CD52?logo=qt&logoColor=white) ![MySQL](https://img.shields.io/badge/MySQL-4479A1?logo=mysql&logoColor=white) |
| **협업 도구** | ![Jira](https://img.shields.io/badge/Jira-0052CC?logo=jira&logoColor=white) ![Confluence](https://img.shields.io/badge/Confluence-172B4D?logo=confluence&logoColor=white) ![Git](https://img.shields.io/badge/Git-F05032?logo=git&logoColor=white) ![GitHub](https://img.shields.io/badge/GitHub-181717?logo=github&logoColor=white) |
| **MLOps** | ![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white) ![TorchServe](https://img.shields.io/badge/TorchServe-FF6F00?logo=pytorch&logoColor=white) ![Docker](https://img.shields.io/badge/Docker-2496ED?logo=docker&logoColor=white) |
| **통신** | ![ROS2](https://img.shields.io/badge/ROS2-DDS-22314E?logo=ros&logoColor=white) ![UDP](https://img.shields.io/badge/UDP-0052CC?logo=wikipedia&logoColor=white) ![HTTP](https://img.shields.io/badge/HTTP-000000?logo=GoogleChrome&logoColor=white) ![TCP](https://img.shields.io/badge/TCP-004880?logo=cisco&logoColor=white) |




# 02. 프로젝트 설계
## 02-1. 주요 기능
<img width="660" alt="image" src="https://github.com/user-attachments/assets/3c218ff5-192f-4879-87e9-4d87851f8050" />

## 02-2. 시스템 구성도
<img width="660" alt="image" src="https://github.com/user-attachments/assets/22245218-3bad-403a-ad0f-79057b9ef289" />

## 02-3. Pyri 로봇 상태 정의
<img width="660" alt="image" src="https://github.com/user-attachments/assets/2e3471e1-56fe-44ad-8428-a18380eda540" />

## 02-4. 구조 작업 시퀀스
<img width="660" alt="image" src="https://github.com/user-attachments/assets/a4dd0f1a-4b79-449c-9edd-1b9be381ce85" />

## 02-5. Behavior Tree
<img width="660" alt="image" src="https://github.com/user-attachments/assets/0f5f6665-1225-47db-83d3-1cf0c468ec36" />

## 02-6. ERD
<img width="660" alt="image" src="https://github.com/user-attachments/assets/bbe280f6-760b-4240-8a45-b7d9cf7c0b58" />

## 02-7. 상황실 화면 구성도
<img width="660" alt="image" src="https://github.com/user-attachments/assets/670a8014-cb19-4742-af86-51d46e1ff583" />


# 03. 주요 기능
## 03-1. 화재 탐지
## 03-2. 구조 대상 탐지
## 03-3. 장애물 대응(1)
## 03-4. 장애물 대응(2)
## 03-5. 화재 확산 차단

# 04. 핵심 기술
## 04-1. 최적 경로 생성
## 04-2. ROS2 :: Domain Bridge
## 04-3. 딥러닝

# 05. 결론
## 05-1. 통합 영상
## 05-2. 테스트 결과
## 05-3. 프로젝트 리뷰 및 개선 사항
