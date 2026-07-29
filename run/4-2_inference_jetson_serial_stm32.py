# =====================================================================
# 4-2단계 수정본: Jetson Nano 실시간 추론 (STM32와 UART 핀헤더 연결 버전)
# =====================================================================
# 원본(4-2_inference_jetson_serial_stm32.py)에서 바뀐 곳 — 딱 3군데:
#   [수정1] SERIAL_PORT: /dev/ttyACM0 → /dev/ttyTHS1 (J41 핀헤더 UART)
#   [수정2] 카메라 버퍼 크기 1로 설정 (최신 프레임 유지)
#   [수정3] TRIGGER 수신 시 촬영 전에 묵은 프레임 버리기
#           (버퍼에 쌓인 "물체 도착 전" 장면이 찍히는 문제 방지)
#
# 배선: Jetson 핀8(TXD)→STM32 PA10, 핀10(RXD)→PA9, 핀6 GND→GND
#
# 사전 준비 (최초 1회):
#   sudo systemctl stop nvgetty && sudo systemctl disable nvgetty
#   sudo usermod -aG dialout $USER
#   sudo reboot
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

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.normpath(os.path.join(BASE_DIR, "..", "models"))
DB_DIR = os.path.normpath(os.path.join(BASE_DIR, "..", "db"))

sys.path.append(DB_DIR)
from points import init_users_table, award_points, is_valid_phone
from model_select import select_model

try:
    from tflite_runtime.interpreter import Interpreter
except ImportError:
    from tensorflow.lite.python.interpreter import Interpreter


# ---------------------------------------------------------------------
# 설정값
# ---------------------------------------------------------------------
SERIAL_PORT = "/dev/ttyTHS1"   # [수정1] J41 핀헤더 UART (기존: /dev/ttyACM0)
SERIAL_BAUDRATE = 115200       # STM32 USART1 설정과 동일

FIXED_ROI = None               # 예: (150, 80, 450, 380) — last_capture.jpg 보고 조정
CNN_CONF_THRESHOLD = 0.6
CAMERA_INDEX = 0
DB_PATH = os.path.join(DB_DIR, "sorting_log.db")

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
    init_users_table(conn)
    conn.commit()
    return conn


def load_class_names(class_names_path):
    with open(class_names_path, encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


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
# CNN 분류 (원본 그대로 — 모델에 Rescaling 포함이라 정규화 안 함)
# ---------------------------------------------------------------------
def classify(interpreter, image, class_names):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    height, width = input_details[0]["shape"][1:3]

    img = cv2.resize(image, (width, height))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
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

    model_path, class_names_path = select_model(MODELS_DIR)
    class_names = load_class_names(class_names_path)

    print("CNN 모델 로딩...")
    cnn_interpreter = Interpreter(model_path=model_path)
    cnn_interpreter.allocate_tensors()

    print(f"STM32 시리얼 연결 시도... ({SERIAL_PORT}, {SERIAL_BAUDRATE}bps)")
    ser = serial.Serial(SERIAL_PORT, SERIAL_BAUDRATE, timeout=1)
    ser.reset_input_buffer()
    time.sleep(2)

    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)   # [수정2] 프레임 버퍼 최소화
    for _ in range(10):                   # 카메라 워밍업 (노출 안정화)
        cap.read()

    threading.Thread(target=input_thread_func, daemon=True).start()

    print("대기 중... (STM32로부터 TRIGGER 신호 대기, Ctrl+C로 종료)")
    try:
        while True:
            line = ser.readline().decode("utf-8", errors="ignore").strip()
            if not line:
                continue

            print("수신:", line)

            if line == "TRIGGER":
                # [수정3] 버퍼에 남은 묵은 프레임 버리고 최신 프레임 확보
                for _ in range(4):
                    cap.read()

                # 1) 촬영
                ret, frame = cap.read()
                if not ret:
                    print("카메라 촬영 실패 -> unknown으로 응답")
                    ser.write(b"CLASS:unknown\n")
                    continue

                cv2.imwrite("last_capture.jpg", frame)   # ROI 조정용 디버그 저장

                # 2) 고정 ROI 적용 후 CNN 분류
                image = apply_roi(frame)
                cnn_class, cnn_conf = classify(cnn_interpreter, image, class_names)
                print(f"CNN class={cnn_class} conf={cnn_conf:.2f}")

                final_class = cnn_class if cnn_conf >= CNN_CONF_THRESHOLD else "unknown"

                # 3) 결과를 STM32로 전송
                ser.write(f"CLASS:{final_class}\n".encode("utf-8"))

                # 4) 포인트 적립
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