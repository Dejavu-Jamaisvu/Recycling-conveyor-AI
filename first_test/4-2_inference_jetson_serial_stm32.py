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
# 사전 준비물
#   - classifier.tflite, class_names.txt (3번 스크립트 결과물)
#   - Jetson Nano <-> STM32 를 USB 케이블로 연결
#
# Jetson Nano에 설치 필요 (최초 1회)
#   pip install opencv-python pyserial
#   pip install tflite-runtime
# =====================================================================

import time
import sqlite3
from datetime import datetime

import numpy as np
import cv2
import serial

try:
    from tflite_runtime.interpreter import Interpreter
except ImportError:
    from tensorflow.lite.python.interpreter import Interpreter


# ---------------------------------------------------------------------
# 설정값 (확인/수정 필요)
# ---------------------------------------------------------------------
CNN_MODEL_PATH = "classifier.tflite"
CLASS_NAMES_PATH = "class_names.txt"

# Jetson Nano에서 터미널에 `ls /dev/tty*` 입력해서 STM32 연결된 포트 확인
# (보통 /dev/ttyACM0 또는 /dev/ttyUSB0 로 잡힘)
SERIAL_PORT = "/dev/ttyACM0"
SERIAL_BAUDRATE = 115200   # STM32 펌웨어의 Baudrate 설정과 반드시 동일해야 함

# 고정 ROI (선택): 물체가 항상 비슷한 자리에 온다면 좌표 지정, 모르면 None
FIXED_ROI = None   # 예: (150, 80, 450, 380)

CNN_CONF_THRESHOLD = 0.6   # 이 이하면 "unknown" 처리
CAMERA_INDEX = 0
DB_PATH = "sorting_log.db"


# ---------------------------------------------------------------------
# 초기화
# ---------------------------------------------------------------------
def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS sorting_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            cnn_class TEXT,
            cnn_confidence REAL,
            final_class TEXT
        )
        """
    )
    conn.commit()
    return conn


def load_class_names():
    with open(CLASS_NAMES_PATH, encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


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
    # 정규화(1/127.5, offset=-1)는 3번 스크립트에서 모델 안에 레이어로 이미
    # 포함되어 tflite로 변환됨 -> 여기서 또 나누면 이중 정규화가 되어 입력값이
    # 전부 -1.01~-0.99 사이로 뭉개짐 (항상 같은 클래스만 나오는 원인)
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
    class_names = load_class_names()

    print("CNN 모델 로딩...")
    cnn_interpreter = Interpreter(model_path=CNN_MODEL_PATH)
    cnn_interpreter.allocate_tensors()

    print(f"STM32 시리얼 연결 시도... ({SERIAL_PORT}, {SERIAL_BAUDRATE}bps)")
    ser = serial.Serial(SERIAL_PORT, SERIAL_BAUDRATE, timeout=1)
    time.sleep(2)  # 보드 리셋 후 안정화될 때까지 대기

    cap = cv2.VideoCapture(CAMERA_INDEX)

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

                # 4) DB 기록
                conn.execute(
                    """
                    INSERT INTO sorting_log (timestamp, cnn_class, cnn_confidence, final_class)
                    VALUES (?, ?, ?, ?)
                    """,
                    (datetime.now().isoformat(), cnn_class, cnn_conf, final_class),
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
