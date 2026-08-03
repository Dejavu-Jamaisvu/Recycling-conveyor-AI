# Recycling-conveyor-AI

컨베이어 벨트로 들어오는 재활용품을 카메라로 촬영해 **YOLO + CNN 하이브리드 AI**로 metal / plastic / paper를 실시간 분류하고, 서보 게이트로 자동 배출하며, 전화번호 기반 포인트를 적립하는 임베디드 분리수거 시스템입니다.

Jetson Nano(AI 추론 · GUI · DB) · STM32F411(실시간 상태머신 · 서보 제어) · Arduino Uno(스텝모터 구동) 3개 보드가 각자 역할을 나눠 맡고, UART/GPIO로 통신하는 구조입니다. 이 저장소는 [Jetson_Recycling-conveyor-belt](https://github.com/kcci-AI/Jetson_Recycling-conveyor-belt)와 [STM32_Recycling-conveyor-belt](https://github.com/kcci-AI/STM32_Recycling-conveyor-belt) 두 팀 저장소를 하나로 모은 통합 저장소입니다.

## 시스템 아키텍처

```
[카메라] --Jetson Nano-->  YOLO 탐지 → CNN 정밀분류 → 최종 클래스(metal/plastic/paper)
                                  │
                     UART 115200bps (TRIGGER / CLASS:xxx)
                                  │
                         [STM32F411] 상태머신
              IR 센서 감지 → 벨트 정지 → 결과 대기 → 서보 게이트 세팅 → 벨트 재가동
                                  │
                        GPIO 2선 (가동/정지 · 방향)
                                  │
                          [Arduino Uno] 스텝모터 구동
                       (컨베이어 벨트 정/역회전 펄스 생성)
```

| 보드 | 역할 | 담당 |
|---|---|---|
| **Jetson Nano** | 카메라 촬영, YOLO+CNN 하이브리드 추론, GUI 대시보드, DB/포인트 | `Jetson/` — 이 프로젝트의 핵심(AI) 파트 |
| **STM32F411** | IR 센서 인터럽트, 4단계 상태머신(MOVING→WAIT_RESULT→SORTING→JOG_BACK), 서보 게이트 2개 PWM 제어 | `STM32/recycle_stm/` |
| **Arduino Uno** | STM32의 가동/정지·방향 신호를 받아 스텝모터 펄스 생성(컨베이어 벨트 구동) | `STM32/recycle_stepmotor_ard/` |

## 전체 프로젝트 구조

```
.
├── Jetson/                              # ★ AI 추론 · GUI · DB (중점 구현 파트)
│   ├── run/
│   │   ├── gui_dashboard_hybrid.py      # 메인: YOLO+CNN 하이브리드 GUI (STM32 연동 + DB/포인트)
│   │   ├── gui_dashboard.py             # CNN 단독 GUI (다중 모델 자동 재시도)
│   │   ├── gui_dashboard_yolo_only.py   # YOLO 단독 GUI
│   │   ├── model_select.py              # 모델 폴더 선택 공용 모듈
│   │   └── 4-1 ~ 4-5_*.py               # 개발 단계별 CLI 버전(GPIO 직접제어 → STM32 UART 전환 등)
│   ├── models/
│   │   ├── first_test/                  # 최초 YOLO/CNN 학습 실험
│   │   ├── 3_class/                     # 자체 촬영 데이터셋(402장)
│   │   └── trash_line_3class/           # Roboflow 공개 데이터셋(2,724장) — 최종 채택
│   └── db/                              # sorting_log.db 초기화/조회/포인트 로직
│
└── STM32/
    ├── recycle_stm/                     # STM32CubeIDE 프로젝트 (HAL, C)
    │   └── Core/
    │       ├── Src/MyApp/
    │       │   ├── conveyor_sequence.c  # 4단계 상태머신(SEQ_MOVING/WAIT_RESULT/SORTING/JOG_BACK)
    │       │   ├── servo.c              # 서보 게이트 2개 각도 제어(PWM)
    │       │   └── ap.c                 # 애플리케이션 진입점
    │       └── Src/usart.c              # Jetson과의 UART 통신(TRIGGER 송신 / CLASS 수신)
    └── recycle_stepmotor_ard/
        └── recycle_stepmotor.ino        # Arduino: STM32 신호 → 스텝모터 펄스 변환
```

## Jetson · AI 파이프라인 (중점 구현 파트)

- **YOLO** (YOLOv8n, Ultralytics): 프레임에서 물체 위치를 찾아 크롭. 신뢰도가 충분히 높으면(기본 0.5 이상) 이 결과를 바로 최종 분류로 채택하고 CNN 단계를 건너뜁니다.
- **CNN** (MobileNetV2 기반, TFLite): YOLO 결과가 애매할 때 크롭된 이미지를 정밀 재분류. 신뢰도가 기준(기본 0.6) 미만이면 다른 모델로 재시도하고, 그래도 부족하면 `unknown` 처리합니다.

### 데이터셋 · 모델 비교

두 데이터셋과 두 학습 전략(Freeze / 파인튜닝)을 모두 직접 학습해 비교했습니다.

| 구분 | 3_class (402장) | trash_line_3class (2,724장, 채택) |
|---|---|---|
| mAP50-95 (YOLO) | 52.0% | 87.0% |
| CNN 정확도 (Freeze → 파인튜닝) | 93.10% → 96.55% | 91.70% → 93.17% |

### 실측 검증 결과 (Jetson 실기기)

| 항목 | 결과 |
|---|---|
| 추론 시간 | YOLO 단독 평균 74ms · CNN 개입 평균 203ms · 전체 평균 111.8ms |
| 판정 경로 분포 | YOLO 1회 확정 78.6% · CNN 개입(기본) 14.3% · 둘 다 확신 부족 7.1% |
| 실환경 정분류 (5개씩) | metal 4/5 · plastic 5/5 · paper 5/5 → 전체 14/15 (93.3%) |

### 실행 방법

```bash
pip3 install opencv-python ultralytics pyserial PyQt5 matplotlib tflite_runtime --break-system-packages
cd Jetson/run
python3 gui_dashboard_hybrid.py
```

### 주요 트러블슈팅

| 문제 | 원인 | 해결 |
|---|---|---|
| STM32 3.3V 신호로 5V 스텝모터 드라이버 무반응 | 로직 레벨 불일치 | Arduino가 5V 펄스 생성을 전담, STM32는 신호 전달만 담당하는 구조로 재설계 |
| unknown 재촬영 시 동일 결과로 중복 트리거 | 같은 장면 재촬영, 센서 미확인 재감지 | 벨트 후진 재검출(최대 2회) + armed 로직으로 재감지 차단 |
| 검증 정확도 90%인데 실전엔 항상 같은 클래스 | 모델 내장 Rescaling + 코드 수동 정규화 이중 적용 | 코드의 수동 정규화를 제거하고 모델 내장 레이어만 사용 |
| ultralytics 설치 후 numpy 충돌로 CNN까지 에러 | TF/ultralytics/matplotlib의 numpy 요구 버전 상충 | CNN 추론 경로를 tflite_runtime으로 전환해 TensorFlow 의존성 자체를 우회 |

## STM32 · 실시간 제어

`conveyor_sequence.c`가 4단계 상태머신으로 시퀀스를 관리합니다.

```
SEQ_MOVING (벨트 ON, IR 감지 대기)
   → IR 감지 → 벨트 정지 → SEQ_WAIT_RESULT (Jetson 추론 결과 대기, UART "TRIGGER" 송신)
   → 결과 수신("CLASS:xxx") → SEQ_SORTING (서보 게이트 세팅 후 벨트 재가동)
   → 결과 미수신/확신부족 → SEQ_JOG_BACK (벨트 후진 후 재검출, 최대 2회) → SEQ_MOVING
```

서보 게이트 2개(`servo.c`)를 조합해 metal/plastic/paper 3개 수거함으로 분기하며, 모든 대기는 `HAL_GetTick` 기반 비차단 방식이라 메인 루프가 멈추지 않습니다.

## Arduino · 컨베이어 구동

STM32가 GPIO 2선(`SIGNAL_PIN`: 가동/정지, `DIR_PIN`: 정/역방향)으로 보내는 신호를 받아 스텝모터 드라이버에 `PUL`/`DIR`/`ENA` 펄스를 직접 생성합니다. STM32가 3.3V, 스텝모터 드라이버가 5V 로직이라 발생했던 무반응 문제를 Arduino가 5V 펄스 생성을 전담하는 구조로 해결했습니다.

## 향후 개선 계획

- YOLO + 로봇팔 기반 다중 분류
- 물성 센서 결합으로 재질 판별 정확도 향상
- 클래스 확장 (유리 · 비닐)
- 참여형 수거함 리워드 서비스

## 팀

심하림 · 임은선
