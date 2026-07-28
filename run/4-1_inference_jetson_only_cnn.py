# =====================================================================
# 4-1단계: Jetson Nano 실시간 추론 + 컨베이어/서보 제어 통합 스크립트
#        (CNN 단독 버전 — YOLO 위치 검출 없음)
# =====================================================================
# 카메라 위치가 고정되어 있고, IR 센서로 한 번에 하나씩만 투입되는 구조라서
# YOLO로 위치를 찾는 과정 없이 촬영한 이미지를 바로 CNN으로 분류합니다.
# (1번 스크립트/YOLO 학습은 이제 필요 없습니다. 2번 크롭 스크립트는
#  3번 CNN 학습용 데이터를 만드는 데는 계속 써도 되고, 통째로 찍은 사진을
#  그대로 학습시켜도 됩니다.)
#
# 폴더 구조 (3개 폴더로 분리, models/는 모델별 하위 폴더로 구성)
#   project/
#   ├── models/
#   │   ├── model_freeze/     classifier.tflite, class_names.txt
#   │   └── model_finetune/   classifier.tflite, class_names.txt
#   ├── run/     이 스크립트 + 4-2 계열 스크립트들 + model_select.py
#   └── db/      points.py, db_setup.py, db_view.py, *.db
#
# 실행할 때 어떤 모델을 쓸지 고를 수 있습니다.
#   python3 4-1_inference_jetson_only_cnn.py                 실행 중 번호로 선택
#   python3 4-1_inference_jetson_only_cnn.py model_finetune   바로 지정
#
# Jetson Nano에 설치 필요 (최초 1회)
#   pip install opencv-python
#   pip install tflite-runtime          # tensorflow 전체 설치보다 가벼움
#   sudo pip install Jetson.GPIO        # 보통 JetPack에 기본 포함되어 있음
#
# 아래 "설정값" 구간은 전부 실제 하드웨어에 맞게 실측해서 채워야 합니다.
# =====================================================================

import os
import sys
import time
import sqlite3
import threading
from datetime import datetime

import numpy as np
import cv2
import Jetson.GPIO as GPIO

# 이 스크립트(run/) 기준으로 형제 폴더(models/, db/) 경로를 계산합니다.
# 어느 위치에서 실행하든(python3 4-1_....py 든, python3 run/4-1_....py 든)
# __file__ 기준이라 항상 올바른 경로를 찾습니다.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.normpath(os.path.join(BASE_DIR, "..", "models"))
DB_DIR = os.path.normpath(os.path.join(BASE_DIR, "..", "db"))

sys.path.append(DB_DIR)  # points.py가 db 폴더에 있어서 import하려면 경로 추가 필요
from points import init_users_table, award_points, is_valid_phone
from model_select import select_model  # 같은 run/ 폴더에 있으므로 바로 import 가능

try:
    from tflite_runtime.interpreter import Interpreter
except ImportError:
    from tensorflow.lite.python.interpreter import Interpreter


# ---------------------------------------------------------------------
# 설정값 (실측 후 수정 필요)
# ---------------------------------------------------------------------
# CNN_MODEL_PATH, CLASS_NAMES_PATH는 고정값이 아니라 main()에서
# select_model()로 실행 시점에 결정됩니다 (models/ 하위 폴더 중 선택).

IR_SENSOR_PIN = 7          # IR 센서 신호 핀 (BOARD 번호 기준, 배선에 맞게 수정)
CONVEYOR_RELAY_PIN = 11    # 컨베이어 모터 ON/OFF 제어 핀
SERVO_PINS = {             # 클래스별 분기 서보가 연결된 핀 (metal/plastic/paper 3종)
    "metal": 12,
    "plastic": 13,
    "paper": 15,
}

BELT_SPEED_MPS = 0.15                 # 벨트 속도 (m/s) - 실측 필요
DIVERT_DISTANCES_M = {                # 촬영 지점 -> 각 분기 지점까지 거리(m)
    "metal": 0.20,
    "plastic": 0.35,
    "paper": 0.50,                    # 가장 마지막 지점 (기본 낙하 경로)
}

# 고정 ROI (선택): 물체가 항상 대략 같은 자리에 온다면 좌표를 지정해서
# 배경을 잘라내면 분류 정확도가 올라갑니다. (x1, y1, x2, y2) 픽셀 좌표.
# 모르겠으면 None으로 두고 전체 프레임을 그대로 씁니다.
FIXED_ROI = None   # 예: (150, 80, 450, 380)

CNN_CONF_THRESHOLD = 0.6              # CNN 분류 확정 최소 신뢰도 (이하면 미확정 처리)
STOP_SETTLE_SEC = 0.3                 # 벨트 정지 후 진동 가라앉는 대기 시간
CAMERA_INDEX = 0
DB_PATH = os.path.join(DB_DIR, "sorting_log.db")

# 현재 반납 세션의 사용자 전화번호. 키보드 입력 스레드가 갱신하고,
# 메인 루프가 분류할 때마다 이 값을 읽어서 포인트를 적립합니다.
current_user = {"phone": None}


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
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS sorting_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            phone TEXT,
            cnn_class TEXT,
            cnn_confidence REAL,
            final_class TEXT,
            servo_used TEXT,
            points_awarded INTEGER NOT NULL DEFAULT 0
        )
        """
    )
    init_users_table(conn)  # points.py: 사용자별 누적 포인트 테이블
    conn.commit()
    return conn


def load_class_names(class_names_path):
    with open(class_names_path, encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


# ---------------------------------------------------------------------
# 키보드로 전화번호 입력받는 스레드
# (메인 루프는 센서/카메라 감시로 바쁘기 때문에, 입력은 별도 스레드에서
#  받아서 current_user만 갱신합니다. 별도 키패드 없이 컴퓨터 키보드로
#  터미널에 그냥 입력하면 됩니다.)
# ---------------------------------------------------------------------
def input_thread_func():
    print("\n[사용자 입력] 전화번호(4자리 이상 숫자)를 입력하고 Enter 누르세요.")
    print("[사용자 입력] 그냥 Enter만 누르면 '게스트'로 진행합니다.\n")
    while True:
        text = input("전화번호 입력 > ").strip()
        if text == "":
            current_user["phone"] = None
            print("-> 게스트로 진행 (포인트 적립 안 됨)")
        elif is_valid_phone(text):
            current_user["phone"] = text
            print(f"-> 사용자 전환: {text} (이후 반납 건은 이 번호로 적립됩니다)")
        else:
            print("전화번호는 숫자만, 4자리 이상 입력해주세요.")


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
    # 주의: 정규화(/127.5 - 1)는 여기서 하지 않습니다.
    # 3번 스크립트의 Keras 모델 안에 Rescaling 레이어가 이미 포함되어 있어서
    # .tflite로 변환된 모델은 0~255 원본 픽셀값을 그대로 입력받습니다.
    # 여기서 또 정규화하면 이중 정규화가 되어 입력이 뭉개집니다.
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

    # models/ 하위 폴더 중 어떤 모델을 쓸지 선택 (인자로 지정 안 하면 번호로 선택)
    model_path, class_names_path = select_model(MODELS_DIR)
    class_names = load_class_names(class_names_path)

    print("CNN 모델 로딩...")
    cnn_interpreter = Interpreter(model_path=model_path)
    cnn_interpreter.allocate_tensors()

    cap = cv2.VideoCapture(CAMERA_INDEX)

    # 전화번호 입력용 스레드 시작 (데몬으로 실행 -> 메인 종료되면 같이 종료)
    threading.Thread(target=input_thread_func, daemon=True).start()

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

                # 6) 포인트 적립 (전화번호가 입력되어 있을 때만)
                phone = current_user["phone"]
                points = 0
                if phone:
                    points = award_points(conn, phone, final_class)
                    if points > 0:
                        print(f"[포인트] {phone}님 +{points}점 적립!")

                # 7) DB 기록
                conn.execute(
                    """
                    INSERT INTO sorting_log
                        (timestamp, phone, cnn_class, cnn_confidence, final_class, servo_used, points_awarded)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        datetime.now().isoformat(),
                        phone,
                        cnn_class,
                        cnn_conf,
                        final_class,
                        servo_used,
                        points,
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
