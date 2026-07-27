# =====================================================================
# 4단계: Jetson Nano 실시간 추론 + 컨베이어/서보 제어 통합 스크립트
#        (CNN 단독 버전 — YOLO 위치 검출 없음)
# =====================================================================
# 카메라 위치가 고정되어 있고, IR 센서로 한 번에 하나씩만 투입되는 구조라서
# YOLO로 위치를 찾는 과정 없이 촬영한 이미지를 바로 CNN으로 분류합니다.
# (1번 스크립트/YOLO 학습은 이제 필요 없습니다. 2번 크롭 스크립트는
#  3번 CNN 학습용 데이터를 만드는 데는 계속 써도 되고, 통째로 찍은 사진을
#  그대로 학습시켜도 됩니다.)
#
# 사전 준비물 (같은 폴더에 넣어두기)
#   - classifier.tflite   (3번 스크립트 결과물: CNN 분류 모델)
#   - class_names.txt     (3번 스크립트 결과물: 클래스 순서, 예: can/glass/plastic)
#
# Jetson Nano에 설치 필요 (최초 1회)
#   pip install opencv-python
#   pip install tflite-runtime          # tensorflow 전체 설치보다 가벼움
#   sudo pip install Jetson.GPIO        # 보통 JetPack에 기본 포함되어 있음
#
# 아래 "설정값" 구간은 전부 실제 하드웨어에 맞게 실측해서 채워야 합니다.
# =====================================================================

import time
import sqlite3
from datetime import datetime

import numpy as np
import cv2
import Jetson.GPIO as GPIO

try:
    from tflite_runtime.interpreter import Interpreter
except ImportError:
    from tensorflow.lite.python.interpreter import Interpreter


# ---------------------------------------------------------------------
# 설정값 (실측 후 수정 필요)
# ---------------------------------------------------------------------
CNN_MODEL_PATH = "classifier.tflite"
CLASS_NAMES_PATH = "class_names.txt"

IR_SENSOR_PIN = 7          # IR 센서 신호 핀 (BOARD 번호 기준, 배선에 맞게 수정)
CONVEYOR_RELAY_PIN = 11    # 컨베이어 모터 ON/OFF 제어 핀
SERVO_PINS = {             # 클래스별 분기 서보가 연결된 핀
    "plastic": 12,
    "glass": 13,
    "can": 15,
}

BELT_SPEED_MPS = 0.15                 # 벨트 속도 (m/s) - 실측 필요
DIVERT_DISTANCES_M = {                # 촬영 지점 -> 각 분기 지점까지 거리(m)
    "can": 0.20,
    "glass": 0.35,
    "plastic": 0.50,                  # 가장 마지막 지점 (기본 낙하 경로)
}

# 고정 ROI (선택): 물체가 항상 대략 같은 자리에 온다면 좌표를 지정해서
# 배경을 잘라내면 분류 정확도가 올라갑니다. (x1, y1, x2, y2) 픽셀 좌표.
# 모르겠으면 None으로 두고 전체 프레임을 그대로 씁니다.
FIXED_ROI = None   # 예: (150, 80, 450, 380)

CNN_CONF_THRESHOLD = 0.6              # CNN 분류 확정 최소 신뢰도 (이하면 미확정 처리)
STOP_SETTLE_SEC = 0.3                 # 벨트 정지 후 진동 가라앉는 대기 시간
CAMERA_INDEX = 0
DB_PATH = "sorting_log.db"


# ---------------------------------------------------------------------
# 초기화
# ---------------------------------------------------------------------
def init_gpio():
    GPIO.setmode(GPIO.BOARD)
    GPIO.setup(IR_SENSOR_PIN, GPIO.IN)
    GPIO.setup(CONVEYOR_RELAY_PIN, GPIO.OUT, initial=GPIO.HIGH)  # 평소엔 벨트 작동 중
    for pin in SERVO_PINS.values():
        GPIO.setup(pin, GPIO.OUT)


def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS sorting_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            cnn_class TEXT,
            cnn_confidence REAL,
            final_class TEXT,
            servo_used TEXT
        )
        """
    )
    conn.commit()
    return conn


def load_class_names():
    with open(CLASS_NAMES_PATH, encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


# ---------------------------------------------------------------------
# 벨트 / 서보 제어
# ---------------------------------------------------------------------
def stop_belt():
    GPIO.output(CONVEYOR_RELAY_PIN, GPIO.LOW)


def start_belt():
    GPIO.output(CONVEYOR_RELAY_PIN, GPIO.HIGH)


def actuate_servo(pin):
    # 서보 하나 작동시키고 원위치. 듀티사이클 값(2.5/7.5)은 사용하는 서보
    # 스펙에 맞춰 조정하세요 (보통 SG90 기준 대략적인 값입니다).
    pwm = GPIO.PWM(pin, 50)  # 50Hz
    pwm.start(0)
    pwm.ChangeDutyCycle(7.5)   # 열림 각도
    time.sleep(0.5)
    pwm.ChangeDutyCycle(2.5)   # 닫힘 각도로 복귀
    time.sleep(0.3)
    pwm.stop()


# ---------------------------------------------------------------------
# 고정 ROI 크롭 (설정 안 하면 원본 그대로 반환)
# ---------------------------------------------------------------------
def apply_roi(frame):
    if FIXED_ROI is None:
        return frame
    x1, y1, x2, y2 = FIXED_ROI
    return frame[y1:y2, x1:x2]


# ---------------------------------------------------------------------
# CNN 분류
# ---------------------------------------------------------------------
def classify(interpreter, image, class_names):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    height, width = input_details[0]["shape"][1:3]

    img = cv2.resize(image, (width, height))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
    img = (img / 127.5) - 1.0  # 3번 스크립트의 MobileNetV2 전처리와 동일하게 맞춤
    img = np.expand_dims(img, axis=0)

    interpreter.set_tensor(input_details[0]["index"], img)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]["index"])[0]

    idx = int(np.argmax(output))
    return class_names[idx], float(output[idx])


# ---------------------------------------------------------------------
# 메인 루프
# ---------------------------------------------------------------------
def main():
    init_gpio()
    conn = init_db()
    class_names = load_class_names()

    print("CNN 모델 로딩...")
    cnn_interpreter = Interpreter(model_path=CNN_MODEL_PATH)
    cnn_interpreter.allocate_tensors()

    cap = cv2.VideoCapture(CAMERA_INDEX)

    print("대기 중... (IR 센서 감지 대기, Ctrl+C로 종료)")
    try:
        while True:
            # IR 센서 반응 여부는 배선에 따라 HIGH/LOW가 반대일 수 있으니
            # 실제 테스트하면서 조건을 맞춰야 합니다.
            if GPIO.input(IR_SENSOR_PIN) == GPIO.HIGH:

                # 1) 벨트 정지
                stop_belt()
                time.sleep(STOP_SETTLE_SEC)

                # 2) 촬영
                ret, frame = cap.read()
                if not ret:
                    print("카메라 촬영 실패, 벨트 재가동 후 재시도")
                    start_belt()
                    continue

                # 3) 고정 ROI 적용 (설정한 경우) 후 CNN으로 바로 분류
                image = apply_roi(frame)
                cnn_class, cnn_conf = classify(cnn_interpreter, image, class_names)
                print(f"CNN class={cnn_class} conf={cnn_conf:.2f}")

                # 4) 신뢰도 확인 -> 최종 클래스 확정
                if cnn_conf < CNN_CONF_THRESHOLD:
                    final_class = "unknown"
                    servo_used = "none"
                    print("판단 불확실 -> 서보 작동 없이 기본 경로로 흘려보냄")
                else:
                    final_class = cnn_class
                    servo_used = final_class

                # 5) 벨트 재가동 + 계산된 대기시간 후 서보 작동
                start_belt()
                if servo_used in SERVO_PINS:
                    distance = DIVERT_DISTANCES_M[servo_used]
                    wait_time = distance / BELT_SPEED_MPS
                    time.sleep(wait_time)
                    actuate_servo(SERVO_PINS[servo_used])

                # 6) DB 기록
                conn.execute(
                    """
                    INSERT INTO sorting_log
                        (timestamp, cnn_class, cnn_confidence, final_class, servo_used)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        datetime.now().isoformat(),
                        cnn_class,
                        cnn_conf,
                        final_class,
                        servo_used,
                    ),
                )
                conn.commit()

            time.sleep(0.05)  # IR 센서 폴링 간격

    except KeyboardInterrupt:
        print("\n종료 중...")
    finally:
        cap.release()
        GPIO.cleanup()
        conn.close()


if __name__ == "__main__":
    main()

# =====================================================================
# 실측/튜닝이 필요한 값 정리
#   - IR_SENSOR_PIN, CONVEYOR_RELAY_PIN, SERVO_PINS: 실제 배선한 GPIO 번호
#   - BELT_SPEED_MPS: 벨트에 표시 그어놓고 이동 시간 재서 계산 (거리 ÷ 시간)
#   - DIVERT_DISTANCES_M: 줄자로 촬영 지점~각 분기 지점 거리 측정
#   - FIXED_ROI: 물체가 항상 비슷한 자리에 온다면 좌표 지정 (선택 사항)
#   - 서보 듀티사이클(2.5 / 7.5): 사용하는 서보 스펙시트 참고해서 조정
#   - CNN_CONF_THRESHOLD: 실제 테스트하면서 오분류율 보고 조정
# =====================================================================
