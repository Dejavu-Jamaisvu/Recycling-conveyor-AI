# =====================================================================
# 4-2단계: Jetson Nano 실시간 추론 스크립트 (STM32와 시리얼 통신 버전)
# =====================================================================
# 하드웨어 역할 분담
#   - STM32  : IR 센서 읽기, 컨베이어 벨트 정지/재가동, 서보 작동 (실시간 제어)
#   - Jetson : 카메라 촬영 + CNN 분류만 담당 (이 스크립트)
#
# STM32 <-> Jetson 시리얼 통신 프로토콜 (예시, STM32 펌웨어에 동일하게 구현 필요)
#   STM32  -> Jetson : "TRIGGER\n"        IR센서 감지 + 벨트 정지 완료 후 전송
#   Jetson -> STM32  : "CLASS:plastic\n"  분류 결과 전송
#                       (STM32가 이 값을 받아서 벨트 재가동 타이밍 계산 +
#                        해당 서보 작동까지 전부 처리)
#
# 이 방식으로 바뀌면서 Jetson.GPIO는 더 이상 쓰지 않습니다. 벨트/서보/센서는
# 전부 STM32 담당이고, Jetson은 "찍고 분류해서 결과만 알려주는" 역할입니다.
#
# 폴더 구조 (3개 폴더로 분리, models/는 모델별 하위 폴더로 구성)
#   project/
#   ├── models/
#   │   ├── model_freeze/     classifier.tflite, class_names.txt
#   │   └── model_finetune/   classifier.tflite, class_names.txt
#   ├── run/     이 스크립트 + 4-1 계열 스크립트들 + model_select.py
#   └── db/      points.py, db_setup.py, db_view.py, *.db
#
# 실행할 때 어떤 모델을 쓸지 고를 수 있습니다.
#   python3 4-2_inference_jetson_serial_stm32.py                 실행 중 번호로 선택
#   python3 4-2_inference_jetson_serial_stm32.py model_finetune   바로 지정
#
# 사전 준비물
#   - Jetson Nano <-> STM32 를 USB 케이블로 연결
#
# Jetson Nano에 설치 필요 (최초 1회)
#   pip install opencv-python pyserial
#   pip install tflite-runtime
# =====================================================================

import os
import sys
import time
import sqlite3
import threading
from datetime import datetime

import numpy as np
import cv2
import serial

# 이 스크립트(run/) 기준으로 형제 폴더(models/, db/) 경로를 계산합니다.
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
# 설정값 (확인/수정 필요)
# ---------------------------------------------------------------------
# CNN_MODEL_PATH, CLASS_NAMES_PATH는 고정값이 아니라 main()에서
# select_model()로 실행 시점에 결정됩니다 (models/ 하위 폴더 중 선택).

# Jetson Nano에서 터미널에 `ls /dev/tty*` 입력해서 STM32 연결된 포트 확인
# (보통 /dev/ttyACM0 또는 /dev/ttyUSB0 로 잡힘)
SERIAL_PORT = "/dev/ttyACM0"
SERIAL_BAUDRATE = 115200   # STM32 펌웨어의 Baudrate 설정과 반드시 동일해야 함

# 고정 ROI (선택): 물체가 항상 비슷한 자리에 온다면 좌표 지정, 모르면 None
FIXED_ROI = None   # 예: (150, 80, 450, 380)

CNN_CONF_THRESHOLD = 0.6   # 이 이하면 "unknown" 처리
CAMERA_INDEX = 0
DB_PATH = os.path.join(DB_DIR, "sorting_log.db")

# 현재 반납 세션의 사용자 전화번호. 키보드 입력 스레드가 갱신하고,
# 메인 루프가 분류할 때마다 이 값을 읽어서 포인트를 적립합니다.
current_user = {"phone": None}


# ---------------------------------------------------------------------
# 초기화
# ---------------------------------------------------------------------
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
# 키보드로 전화번호 입력받는 스레드 (별도 키패드 없이 컴퓨터 키보드로 입력)
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
    conn = init_db()

    # models/ 하위 폴더 중 어떤 모델을 쓸지 선택 (인자로 지정 안 하면 번호로 선택)
    model_path, class_names_path = select_model(MODELS_DIR)
    class_names = load_class_names(class_names_path)

    print("CNN 모델 로딩...")
    cnn_interpreter = Interpreter(model_path=model_path)
    cnn_interpreter.allocate_tensors()

    print(f"STM32 시리얼 연결 시도... ({SERIAL_PORT}, {SERIAL_BAUDRATE}bps)")
    ser = serial.Serial(SERIAL_PORT, SERIAL_BAUDRATE, timeout=1)
    time.sleep(2)  # 보드 리셋 후 안정화될 때까지 대기

    cap = cv2.VideoCapture(CAMERA_INDEX)

    # 전화번호 입력용 스레드 시작 (데몬으로 실행 -> 메인 종료되면 같이 종료)
    threading.Thread(target=input_thread_func, daemon=True).start()

    print("대기 중... (STM32로부터 TRIGGER 신호 대기, Ctrl+C로 종료)")
    try:
        while True:
            line = ser.readline().decode("utf-8", errors="ignore").strip()
            if not line:
                continue

            print("수신:", line)

            if line == "TRIGGER":
                # 1) 촬영 (STM32가 이미 벨트를 정지시킨 상태에서 신호를 보냄)
                ret, frame = cap.read()
                if not ret:
                    print("카메라 촬영 실패 -> unknown으로 응답")
                    ser.write(b"CLASS:unknown\n")
                    continue

                # 2) 고정 ROI 적용 후 CNN 분류
                image = apply_roi(frame)
                cnn_class, cnn_conf = classify(cnn_interpreter, image, class_names)
                print(f"CNN class={cnn_class} conf={cnn_conf:.2f}")

                final_class = cnn_class if cnn_conf >= CNN_CONF_THRESHOLD else "unknown"

                # 3) 결과를 STM32로 전송 -> 벨트 재가동 + 서보 작동은 STM32가 처리
                ser.write(f"CLASS:{final_class}\n".encode("utf-8"))

                # 4) 포인트 적립 (전화번호가 입력되어 있을 때만)
                phone = current_user["phone"]
                points = 0
                if phone:
                    points = award_points(conn, phone, final_class)
                    if points > 0:
                        print(f"[포인트] {phone}님 +{points}점 적립!")

                # 5) DB 기록
                conn.execute(
                    """
                    INSERT INTO sorting_log
                        (timestamp, phone, cnn_class, cnn_confidence, final_class, points_awarded)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (datetime.now().isoformat(), phone, cnn_class, cnn_conf, final_class, points),
                )
                conn.commit()

    except KeyboardInterrupt:
        print("\n종료 중...")
    finally:
        cap.release()
        ser.close()
        conn.close()


if __name__ == "__main__":
    main()

# =====================================================================
# 확인/맞춰야 할 것
#   - SERIAL_PORT: Jetson에서 `ls /dev/tty*` 로 STM32 연결 포트 확인 후 수정
#   - SERIAL_BAUDRATE: STM32 펌웨어의 Serial.begin(baudrate) 값과 반드시 일치
#   - "TRIGGER" / "CLASS:xxx" 문자열 프로토콜은 STM32 펌웨어 쪽에도
#     동일하게 구현되어야 함 (STM32가 CLASS: 뒤 값을 보고 벨트 재가동
#     타이밍 계산 + 해당 서보를 작동시켜야 함)
#   - FIXED_ROI, CNN_CONF_THRESHOLD: 실제 테스트하며 조정
# =====================================================================
