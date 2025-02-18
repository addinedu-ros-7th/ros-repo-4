<p align="center">
  <img width="916" alt="image" src="https://github.com/user-attachments/assets/93aa6310-9032-4851-b83d-fbc23b61d00a" />
</p>

# 00. 통합 영상 및 발표 자료
## 01-1. 통합 영상
[![ ](https://img.youtube.com/vi/GnMYARfWPEs/0.jpg)](https://youtube.com/shorts/GnMYARfWPEs?si=HJs3EaWiIBXYDwoN)
## 02-1. 발표 자료
[파이토끼 발표 자료](https://docs.google.com/presentation/d/13RBxSe_4TvYX2x6u4SbVQMm9CwZP5-0HAGWoL_2vUaw/edit?usp=sharing)
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
## 03-1. 전체 경로 탐색
### Kruskal & A* 알고리즘을 이용한 최적 경로 탐색
<img src="https://github.com/user-attachments/assets/a57346d5-68b2-45bf-a4e3-e78e6a432175" width="600" height="200">
<img src="https://github.com/user-attachments/assets/65aa3d6a-c0a1-42ca-9a46-aeace1e3f1e8" width="600" height="200">
<img src="https://github.com/user-attachments/assets/d5c56928-a161-4de3-bc50-c2d40ceda8b3" width="600" height="200">

## 03-2. 위험도 분석
### 화재 및 연기 탐지 / 위험도 분석
<img src="https://github.com/user-attachments/assets/11efa4a3-fdf5-486e-b7ab-eeb7548c979f" width="300" height="400">
<img src="https://github.com/user-attachments/assets/1fe0e000-ab33-407f-b42a-471f08af9396" width="300" height="400">

## 03-3. 구조 대상 탐지
### 구조 대상 탐지 및 구조 대상 자세, 나이, 성별 추정
<img src="https://github.com/user-attachments/assets/0a39ba48-0afb-43ca-971e-d0123a52c3b1" width="300" height="400">
<img src="https://github.com/user-attachments/assets/e7d03ff8-2ba3-4baa-b951-e5d4996fccf6" width="300" height="400">

## 03-4. 장애물 대응
<img src="https://github.com/user-attachments/assets/f1bdb5ef-c359-4982-bc69-d9159d780491" width="300" height="200">
<img src="https://github.com/user-attachments/assets/ec4802f4-3ab4-42fe-ad4a-57080ea4488e" width="300" height="200">
<img src="https://github.com/user-attachments/assets/e85b5ff4-65ba-47b7-96a9-80e9f4b3af32" width="300" height="200">

## 03-5. 화재 확산 차단

# 04. 핵심 기술
## 04-1. 최적 경로 생성

| <img src="https://github.com/user-attachments/assets/18dedd00-2364-4508-a5d5-acdaa0e7c9b0" width="200"> | <img src="https://github.com/user-attachments/assets/476f25c2-b1ca-405e-98e0-6b544f87c4b0" width="200"> | <img src="https://github.com/user-attachments/assets/8a957cbe-4db8-443d-8a39-b95b668db48a" width="200"> |
|:--:|:--:|:--:|
| 도면 이미지에서 셀 추출 및 분석 | 통로 중앙 지점 <br>1차 Waypoint로 선택 | 군집화된 점들 중 중심점 추출<br> 최종 Waypoint로 선택 |

| <img src="https://github.com/user-attachments/assets/2177ce4e-bcf6-4762-9497-51ccade4c9cc" width="200"> | <img src="https://github.com/user-attachments/assets/47434bf6-c2be-4d65-8009-01c059d6dc50" width="200"> | <img src="https://github.com/user-attachments/assets/0cba918d-7b47-42a3-9389-879fdb6eb2f0" width="200"> |
|:--:|:--:|:--:|
| 각 좌표에 넘버링, 그래프화 | 노드 사이의 거리를 cost로 사용 | 모든 노드를 최소 비용 연결하는<br> 하나의 그래프 생성 |

## 04-2. ROS2 :: Domain Bridge
- 다중 로봇 제어
<img width="660" alt="image" src="https://github.com/user-attachments/assets/94ea6444-b20b-4293-bc3f-4e110c9409d9" />

## 04-3. 딥러닝

# 05. 결론
## 05-1. 테스트 결과
| Category         | Test Case | Priority | Result |
|-----------------|------------------------------------------------|:----------:|:--------:|
| **위험도 분석 및 대피 경로 생성** | 도면 정보를 바탕으로 현장 탐색 경로 생성 | 1 | PASS |
| | 경로를 배분하여 두 대의 로봇이 자율적으로 탐색 | 1 | PASS |
| | 딥러닝 데이터와 센서 데이터로 위치별 위험도 분석 | 2 | PASS |
| | 위험 구역과 안전 구역을 색상으로 구분하여 상황실 UI에 시각화 | 1 | PASS |
| **장애물 대응** | 장애물 감지 시 후진을 통해 회피 | 2 | PASS |
| | 장애물 감지 시 주행 가능한 경로로 재설정 | 3 | PASS |
| **화재 확산 차단** | 방화문 지점 도착 시 방화문 상태를 파악하여 열려 있을 경우 폐문 | 2 | PASS |
| **상황실 관제** | 구조 대상자와 상황실과의 실시간 통신 | 2 | FAIL |
| **구조 대상 탐지 및 구조 지원** | 구조 대상자 감지 및 위치와 거리 추정 | 1 | PASS |
| | 구조 대상자의 성별, 나이, 자세 추정 | 1 | PASS |
| | 구조 대상자 감지 시 맵 위에 시각화 | 1 | PASS |
| | 위험도와 현재 로봇/구조대상자 위치를 기반으로 대피 경로 생성 | 1 | PASS |
| | 상황실의 선택에 따라 구조 혹은 대피 안내 | 1 | PASS |
| | 들 것으로 구조 | 4 | FAIL |
| **구호 물품 관리 및 전달** | 구조 대상자 감지 시 접근 후 구호 물품 전달 | 2 | PASS |
| | 구호 물품 잔여량 모니터링 | 3 | PASS |

## 05-2. 프로젝트 리뷰 및 개선 사항

| **Category**        | **문제점**                                      | **해결 시도**                                  | **결과**                               | **개선 방안**                                      |
|---------------------|----------------------------------------------|----------------------------------------------|--------------------------------------|--------------------------------------------------|
| **Localization**    | 랜드마크 없는 단순 환경에서 위치 오류 빈번    | 딥러닝을 활용한 위치 보정                      | 보정되었지만 위치 오류 재발생          | Feature-based Localization 적용 및 데이터 융합을 통한 위치 보정 |
|                     | 좁은 주행 공간에서 로봇 제어 문제 발생       | Cost map, 속도 파라미터 수정                   | 비교적 안정된 주행                  | 고성능 IMU 및 LiDAR 센서 사용                     |
|                     | 바퀴 미끄러짐 및 회전 오차                   | 최소한의 회전 및 방향 정보 전달                | Localization 정확도 일부 개선         | 엔코더 모터 사용                                  |
| **Deep Learning**   | 복잡한 환경에서 적용 가능한 모델 필요        | 데이터 증강을 통해 개선                        | 모델 성능 일부 개선                   | 적외선 카메라 및 다양한 센서 융합 멀티 모달 학습  |
|                     | 엣지 AI 도입을 위한 모델 양자화               | 라즈베리파이5에서 적용 테스트                  | 리소스 한계로 도입 어려움             | 최적화된 통신과 양자화 모델 사용                  |
