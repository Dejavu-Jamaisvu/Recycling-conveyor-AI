# =====================================================================
# 4-2단계 수정본: Jetson Nano 실시간 추론 (STM32와 UART 핀헤더 연결 버전)
# =====================================================================
# 하드웨어 수정 사항 (본인이 실측/테스트해서 반영한 것, 그대로 유지):
#   [수정1] SERIAL_PORT: /dev/ttyACM0 → /dev/ttyTHS1 (J41 핀헤더 UART)
#   [수정2] 카메라 버퍼 크기 1로 설정 (최신 프레임 유지)
#   [수정3] 카메라 워밍업 10프레임 (노출 안정화)
#
# 이번에 추가한 것:
#   [수정4] 카메라 미리보기 창 추가 (cv2.imshow) + 분류 결과 화면 오버레이
#   [수정5] 시리얼 수신을 별도 스레드로 분리
#           (원래 ser.readline()이 메인 스레드를 막고 있어서 미리보기를
#            같이 띄울 수 없었음. 스레드에서 "TRIGGER" 감지만 하고,
#            실제 촬영/분류는 메인 스레드의 미리보기 루프에서 처리)
#
# [수정3']에 대한 참고: 원래 있던 "TRIGGER 수신 시 묵은 프레임 4장 버리기"는
# 뺐습니다. 이제 미리보기 루프가 매 프레임 cap.read()를 계속 호출하고 있어서
# (기존엔 대기 중엔 read()를 아예 안 불러서 버퍼에 묵은 프레임이 쌓였던 것),
# TRIGGER가 왔을 때 쓰는 frame은 항상 그 순간 막 읽은 최신 프레임입니다.
# 오히려 4번 더 읽는 건 트리거 순간에 불필요한 지연만 추가하는 셈이라 제거했고,
# BUFFERSIZE=1 설정은 그대로 유지했습니다 (안전장치로 유지하는 게 좋음).
#
# 배선: Jetson 핀8(TXD)→STM32 PA10, 핀10(RXD)→PA9, 핀6 GND→GND
#
# 사전 준비 (최초 1회):
#   sudo systemctl stop nvgetty && sudo systemctl disable nvgetty
#   sudo usermod -aG dialout $USER
#   sudo reboot
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
# 주의: 카메라 미리보기 창(cv2.imshow)을 띄우므로 모니터가 연결된 상태에서
# (SSH 원격 접속이 아니라 Jetson 본체 데스크톱에서, 또는 VNC로) 실행해야 합니다.
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
SERIAL_PORT = "/dev/ttyTHS1"   # [수정1] J41 핀헤더 UART
SERIAL_BAUDRATE = 115200       # STM32 USART1 설정과 동일

FIXED_ROI = None               # 예: (150, 80, 450, 380) — last_capture.jpg 보고 조정
CNN_CONF_THRESHOLD = 0.6
CAMERA_INDEX = 0
DB_PATH = os.path.join(DB_DIR, "sorting_log.db")

LAST_CAPTURE_PATH = os.path.join(BASE_DIR, "last_capture.jpg")  # ROI 조정용 디버그 저장

current_user = {"phone": None}

# STM32로부터 "TRIGGER"를 받으면 이 이벤트를 세팅합니다.
# (시리얼 읽기는 별도 스레드, 카메라 미리보기+분류는 메인 스레드에서 동시에 처리)
trigger_event = threading.Event()


# ---------------------------------------------------------------------
# 초기화
# ---------------------------------------------------------------------
def _ensure_column(conn, table, column, coltype):
    """예전 실행으로 이미 만들어진 sorting_log.db를 이어서 쓸 때 새로 추가된
    컬럼이 없으면 채워넣습니다."""
    cols = [row[1] for row in conn.execute(f"PRAGMA table_info({table})")]
    if column not in cols:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {coltype}")


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
    _ensure_column(conn, "sorting_log", "infer_ms", "REAL")  # CNN 추론 1회 소요시간(ms)
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


# ---------------------------------------------------------------------
# [수정5] STM32 시리얼 수신 스레드 ("TRIGGER" 감지 -> trigger_event만 세팅)
# 실제 촬영/분류/응답은 메인 스레드(카메라 미리보기 루프)에서 처리합니다.
# ---------------------------------------------------------------------
def serial_listener_func(ser):
    while True:
        try:
            line = ser.readline().decode("utf-8", errors="ignore").strip()
        except serial.SerialException:
            print("[시리얼] 연결이 끊겼습니다.")
            return
        if not line:
            continue
        print("[시리얼 수신]", line)
        if line == "TRIGGER":
            trigger_event.set()


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
# 메인 루프 (카메라 미리보기 + TRIGGER 처리를 함께 수행)
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
    if not cap.isOpened():
        print(f"카메라(index={CAMERA_INDEX})를 열 수 없습니다. CAMERA_INDEX 값을 확인하세요.")
        return
    for _ in range(10):                   # [수정3] 카메라 워밍업 (노출 안정화)
        cap.read()

    threading.Thread(target=input_thread_func, daemon=True).start()
    threading.Thread(target=serial_listener_func, args=(ser,), daemon=True).start()

    window_name = "Camera Preview (STM32 TRIGGER 자동 촬영) - Q: Quit"
    last_result_text = ""
    last_result_until = 0.0

    print("대기 중... (STM32로부터 TRIGGER 신호 대기, 미리보기 창에서 Q로 종료)")
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("카메라 촬영 실패")
                break

            # --- 미리보기 화면 구성 ---
            display = frame.copy()
            if FIXED_ROI is not None:
                x1, y1, x2, y2 = FIXED_ROI
                cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2)

            user_label = current_user["phone"] if current_user["phone"] else "게스트"
            cv2.putText(
                display, f"user: {user_label}", (10, display.shape[0] - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2,
            )

            if last_result_text and time.time() < last_result_until:
                cv2.putText(
                    display, last_result_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2,
                )
            cv2.imshow(window_name, display)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27):  # 27 = ESC
                break

            # --- STM32 TRIGGER 처리 ---
            # frame은 이 루프 이번 회차에서 방금 막 읽은 최신 프레임이라
            # 따로 "묵은 프레임 버리기"가 필요 없습니다.
            if trigger_event.is_set():
                trigger_event.clear()

                cv2.imwrite(LAST_CAPTURE_PATH, frame)  # ROI 조정용 디버그 저장

                # 1) 고정 ROI 적용 후 CNN 분류
                image = apply_roi(frame)
                t0 = time.perf_counter()
                cnn_class, cnn_conf = classify(cnn_interpreter, image, class_names)
                infer_ms = (time.perf_counter() - t0) * 1000
                print(f"CNN class={cnn_class} conf={cnn_conf:.2f} ({infer_ms:.1f} ms)")

                final_class = cnn_class if cnn_conf >= CNN_CONF_THRESHOLD else "unknown"

                # 화면에 결과 3초간 표시
                last_result_text = f"{final_class} ({cnn_conf:.2f})"
                last_result_until = time.time() + 3

                # 2) 결과를 STM32로 전송
                ser.write(f"CLASS:{final_class}\n".encode("utf-8"))

                # 3) 포인트 적립
                phone = current_user["phone"]
                points = 0
                if phone:
                    points = award_points(conn, phone, final_class)
                    if points > 0:
                        print(f"  [포인트] {phone}님 +{points}점 적립!")

                # 4) DB 기록
                conn.execute(
                    """
                    INSERT INTO sorting_log
                        (timestamp, phone, cnn_class, cnn_confidence, final_class, points_awarded, infer_ms)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (datetime.now().isoformat(), phone, cnn_class, cnn_conf, final_class, points, infer_ms),
                )
                conn.commit()

    except KeyboardInterrupt:
        print("\n종료 중...")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        ser.close()
        conn.close()


if __name__ == "__main__":
    main()

# =====================================================================
# 확인/맞춰야 할 것
#   - SERIAL_PORT: 핀헤더 UART가 아니라면 /dev/ttyACM0 등으로 다시 바꿀 것
#   - SERIAL_BAUDRATE: STM32 펌웨어의 Serial.begin(baudrate) 값과 반드시 일치
#   - "TRIGGER" / "CLASS:xxx" 문자열 프로토콜은 STM32 펌웨어 쪽에도 동일하게 구현
#   - FIXED_ROI: last_capture.jpg 확인하면서 좌표 조정
#   - 모니터가 연결된 Jetson 본체 데스크톱(또는 VNC)에서 실행해야 미리보기 창이 뜸
# =====================================================================