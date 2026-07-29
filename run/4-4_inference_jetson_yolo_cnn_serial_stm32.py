# =====================================================================
# 4-4단계: Jetson Nano 실시간 추론 (STM32 UART 연결 + YOLO 위치검출 + CNN 분류)
# =====================================================================
# 4-2(STM32 시리얼 + 카메라 미리보기)에 YOLO 위치검출을 추가한 버전입니다.
#
# *** 실제 하드웨어 구조 (중요) ***
#   Jetson --- UART(1:1) --- STM32 --- (IR센서, 서보 3개, 아두이노에 컨베이어
#   제어 중계)
#   즉 Jetson은 STM32 하나하고만 UART로 통신합니다. GPIO로 뭔가를 직접
#   제어하지 않습니다 (그건 4-1/4-3처럼 STM32 없이 GPIO를 직접 쓰는 구조일 때
#   얘기고, 지금 하드웨어는 그게 아닙니다). 프로토콜은 4-2와 동일합니다.
#     STM32 -> Jetson : "TRIGGER"        (IR센서가 물체 감지 + 벨트 정지 완료)
#     Jetson -> STM32 : "CLASS:metal" 등  (분류 결과 - STM32가 서보 작동시킴)
#
# YOLO 위치검출은 두 가지 방식을 지원하고, 아래 우선순위로 자동 선택합니다
# (4-3과 동일한 방식).
#   1) best.pt   - ultralytics로 바로 로딩 (torch/ultralytics 설치 필요)
#   2) detector.tflite - tflite_runtime으로 로딩 (가볍고 설치 간단, Colab에서
#      best.pt를 미리 변환해둬야 함 - 1_train_yolo_colab.py 참고)
#   파일은 models/detector/ 안에 있어도 되고, 지금 선택한 CNN 모델 폴더
#   (models/모델이름/) 안에 같이 넣어도 인식됩니다.
#   둘 다 없으면: 위치검출 없이 4-2처럼 전체 프레임(또는 FIXED_ROI)을 그대로
#   CNN에 넣습니다 - 즉 이 스크립트는 YOLO 없이도 그냥 실행됩니다.
#
# 하드웨어 수정 사항 (4-2에서 이어받음, 그대로 유지)
#   - SERIAL_PORT: /dev/ttyTHS1 (J41 핀헤더 UART, 배선: 핀8(TXD)→STM32 PA10,
#     핀10(RXD)→PA9, 핀6 GND→GND)
#   - 카메라 버퍼 크기 1 + 워밍업 10프레임
#   - 시리얼 수신은 별도 스레드 (trigger_event) + 메인 스레드는 미리보기 루프
#
# 사전 준비 (최초 1회):
#   sudo systemctl stop nvgetty && sudo systemctl disable nvgetty
#   sudo usermod -aG dialout $USER
#   sudo reboot
#
# 폴더 구조 (3개 폴더, models/는 모델별 하위 폴더 + detector 폴더로 구성)
#   project/
#   ├── models/
#   │   ├── model_freeze/     classifier.tflite, class_names.txt   (CNN)
#   │   ├── model_finetune/   classifier.tflite, class_names.txt   (CNN)
#   │   └── detector/         best.pt 또는 detector.tflite           (YOLO, 선택)
#   ├── run/     이 스크립트 + 4-x 계열 스크립트들 + model_select.py
#   └── db/      points.py, db_setup.py, db_view.py, *.db
#
# 실행할 때 어떤 CNN 모델을 쓸지 고를 수 있습니다.
#   python3 4-4_inference_jetson_yolo_cnn_serial_stm32.py                 번호로 선택
#   python3 4-4_inference_jetson_yolo_cnn_serial_stm32.py model_finetune  바로 지정
#
# 설치 (필요한 것만 설치하면 됨)
#   pip3 install opencv-python pyserial
#   pip3 install tflite-runtime          # detector.tflite 쓸 경우
#   pip3 install ultralytics             # best.pt 직접 쓸 경우 (torch 필요)
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
SERIAL_PORT = "/dev/ttyTHS1"   # J41 핀헤더 UART
SERIAL_BAUDRATE = 115200       # STM32 USART 설정과 동일

FIXED_ROI = None               # YOLO가 없거나 못 찾았을 때 대신 쓸 좌표. 예: (150, 80, 450, 380)
CNN_CONF_THRESHOLD = 0.6
CAMERA_INDEX = 0
DB_PATH = os.path.join(DB_DIR, "sorting_log.db")

LAST_CAPTURE_PATH = os.path.join(BASE_DIR, "last_capture.jpg")  # ROI 조정용 디버그 저장

DETECTOR_DIR = os.path.join(MODELS_DIR, "detector")
DETECTOR_CONF_THRESHOLD = 0.4  # YOLO 검출 최소 신뢰도

current_user = {"phone": None}

# STM32로부터 "TRIGGER"를 받으면 이 이벤트를 세팅합니다.
# (시리얼 읽기는 별도 스레드, 카메라 미리보기+분류는 메인 스레드에서 동시에 처리)
trigger_event = threading.Event()


# ---------------------------------------------------------------------
# 초기화
# ---------------------------------------------------------------------
def _ensure_column(conn, table, column, coltype):
    """예전 스크립트(4-2 등)로 이미 만들어진 sorting_log.db를 이어서 쓸 때
    새로 추가된 컬럼이 없으면 채워넣습니다. (CREATE TABLE IF NOT EXISTS는
    테이블이 이미 있으면 그냥 넘어가서 새 컬럼이 반영 안 됨)"""
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
            yolo_confidence REAL,
            cnn_class TEXT,
            cnn_confidence REAL,
            final_class TEXT,
            points_awarded INTEGER NOT NULL DEFAULT 0
        )
        """
    )
    _ensure_column(conn, "sorting_log", "yolo_confidence", "REAL")
    _ensure_column(conn, "sorting_log", "points_awarded", "INTEGER NOT NULL DEFAULT 0")
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
# STM32 시리얼 수신 스레드 ("TRIGGER" 감지 -> trigger_event만 세팅)
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
# YOLO 검출기 (best.pt / detector.tflite 둘 다 지원, 4-3과 동일한 방식)
# ---------------------------------------------------------------------
class TFLiteDetector:
    """detector.tflite(ultralytics tflite export) 기반 검출기."""

    def __init__(self, path):
        self.interpreter = Interpreter(model_path=path)
        self.interpreter.allocate_tensors()

    def detect_and_crop(self, frame, conf_threshold=DETECTOR_CONF_THRESHOLD):
        input_details = self.interpreter.get_input_details()
        output_details = self.interpreter.get_output_details()
        in_h, in_w = input_details[0]["shape"][1:3]

        img = cv2.resize(frame, (in_w, in_h))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        img = np.expand_dims(img, axis=0)

        self.interpreter.set_tensor(input_details[0]["index"], img)
        self.interpreter.invoke()
        output = self.interpreter.get_tensor(output_details[0]["index"])[0]

        if output.shape[0] < output.shape[1]:
            output = output.T  # -> (박스개수, 4+클래스수)

        boxes = output[:, :4]
        scores = output[:, 4:]
        if scores.size == 0:
            return None, 0.0, None
        confidences = np.max(scores, axis=1)

        best_idx = int(np.argmax(confidences))
        best_conf = float(confidences[best_idx])
        if best_conf < conf_threshold:
            return None, best_conf, None

        cx, cy, w, h = boxes[best_idx]
        frame_h, frame_w = frame.shape[:2]

        # ultralytics export 버전에 따라 좌표가 0~1 정규화값이거나 모델 입력크기
        # 기준 픽셀값이거나 다를 수 있어서, 값 범위를 보고 자동으로 판단합니다.
        if max(cx, cy, w, h) <= 1.5:
            cx, cy, w, h = cx * frame_w, cy * frame_h, w * frame_w, h * frame_h
        else:
            cx = cx * (frame_w / in_w)
            cy = cy * (frame_h / in_h)
            w = w * (frame_w / in_w)
            h = h * (frame_h / in_h)

        x1 = max(0, int(cx - w / 2))
        y1 = max(0, int(cy - h / 2))
        x2 = min(frame_w, int(cx + w / 2))
        y2 = min(frame_h, int(cy + h / 2))

        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return None, best_conf, None
        return crop, best_conf, (x1, y1, x2, y2)


class UltralyticsDetector:
    """best.pt를 ultralytics로 바로 로딩하는 검출기 (torch 설치 필요)."""

    def __init__(self, path):
        from ultralytics import YOLO  # 여기서만 import (torch 없으면 여기서 실패해서 폴백됨)
        self.model = YOLO(path)

    def detect_and_crop(self, frame, conf_threshold=DETECTOR_CONF_THRESHOLD):
        results = self.model.predict(frame, conf=conf_threshold, verbose=False)
        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            return None, 0.0, None

        best_idx = int(boxes.conf.argmax().item())
        conf = float(boxes.conf[best_idx])
        x1, y1, x2, y2 = boxes.xyxy[best_idx].tolist()

        frame_h, frame_w = frame.shape[:2]
        x1 = max(0, int(x1))
        y1 = max(0, int(y1))
        x2 = min(frame_w, int(x2))
        y2 = min(frame_h, int(y2))

        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return None, conf, None
        return crop, conf, (x1, y1, x2, y2)


def load_detector(extra_dir=None):
    """best.pt(ultralytics) 우선 시도 -> 실패/없으면 detector.tflite -> 그것도
    없으면 None (위치검출 없이 동작).

    파일은 models/detector/ 안에 있어도 되고, 지금 선택한 CNN 모델 폴더
    (models/모델이름/, extra_dir로 전달됨) 안에 같이 넣어도 인식됩니다.
    """
    search_dirs = [d for d in [extra_dir, DETECTOR_DIR] if d]

    for d in search_dirs:
        pt_path = os.path.join(d, "best.pt")
        if os.path.exists(pt_path):
            print(f"YOLO 검출기(.pt, ultralytics) 로딩 시도... ({pt_path})")
            try:
                return UltralyticsDetector(pt_path)
            except Exception as e:
                print(f"best.pt 로딩 실패 ({e})")
                print("-> torch/ultralytics가 설치돼 있는지 확인하세요:")
                print('   python3 -c "import torch, ultralytics"')
                print("-> 다른 위치/파일이 있으면 이어서 찾아봅니다.")

    for d in search_dirs:
        tflite_path = os.path.join(d, "detector.tflite")
        if os.path.exists(tflite_path):
            print(f"YOLO 검출기(.tflite) 로딩... ({tflite_path})")
            try:
                return TFLiteDetector(tflite_path)
            except Exception as e:
                print(f"detector.tflite 로딩 실패: {e}")

    print(f"YOLO 검출기 없음 (찾아본 위치: {', '.join(search_dirs)})")
    print("-> 위치검출 없이 진행 (전체 프레임/FIXED_ROI로 분류)")
    return None


# ---------------------------------------------------------------------
# CNN 분류 (모델에 Rescaling 포함이라 여기서 정규화 안 함)
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

    # 선택된 CNN 모델 폴더 안도 같이 뒤져봄 (models/detector/ 뿐 아니라
    # models/모델이름/ 안에 best.pt를 넣어도 인식되게)
    detector = load_detector(extra_dir=os.path.dirname(model_path))

    print(f"STM32 시리얼 연결 시도... ({SERIAL_PORT}, {SERIAL_BAUDRATE}bps)")
    ser = serial.Serial(SERIAL_PORT, SERIAL_BAUDRATE, timeout=1)
    ser.reset_input_buffer()
    time.sleep(2)

    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)   # 프레임 버퍼 최소화
    if not cap.isOpened():
        print(f"카메라(index={CAMERA_INDEX})를 열 수 없습니다. CAMERA_INDEX 값을 확인하세요.")
        return
    for _ in range(10):                   # 카메라 워밍업 (노출 안정화)
        cap.read()

    threading.Thread(target=input_thread_func, daemon=True).start()
    threading.Thread(target=serial_listener_func, args=(ser,), daemon=True).start()

    window_name = "Camera Preview (STM32 TRIGGER 자동 촬영, YOLO+CNN) - Q: Quit"
    last_result_text = ""
    last_result_until = 0.0
    last_box = None
    last_box_until = 0.0

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

            if last_box is not None and time.time() < last_box_until:
                bx1, by1, bx2, by2 = last_box
                cv2.rectangle(display, (bx1, by1), (bx2, by2), (255, 220, 0), 2)

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

                # 1) YOLO로 위치 검출 + 크롭 (detector 없으면 고정ROI/전체프레임 사용)
                yolo_conf = 0.0
                image = None
                box = None
                if detector is not None:
                    image, yolo_conf, box = detector.detect_and_crop(frame)
                if image is None:
                    image = apply_roi(frame)
                last_box = box
                last_box_until = time.time() + 3

                # 2) CNN으로 최종 분류
                cnn_class, cnn_conf = classify(cnn_interpreter, image, class_names)
                print(f"YOLO conf={yolo_conf:.2f} / CNN class={cnn_class} conf={cnn_conf:.2f}")

                final_class = cnn_class if cnn_conf >= CNN_CONF_THRESHOLD else "unknown"

                # 화면에 결과 3초간 표시
                last_result_text = f"{final_class} ({cnn_conf:.2f})"
                last_result_until = time.time() + 3

                # 3) 결과를 STM32로 전송 (STM32가 서보 작동시킴)
                ser.write(f"CLASS:{final_class}\n".encode("utf-8"))

                # 4) 포인트 적립
                phone = current_user["phone"]
                points = 0
                if phone:
                    points = award_points(conn, phone, final_class)
                    if points > 0:
                        print(f"  [포인트] {phone}님 +{points}점 적립!")

                # 5) DB 기록
                conn.execute(
                    """
                    INSERT INTO sorting_log
                        (timestamp, phone, yolo_confidence, cnn_class, cnn_confidence,
                         final_class, points_awarded)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        datetime.now().isoformat(),
                        phone,
                        yolo_conf,
                        cnn_class,
                        cnn_conf,
                        final_class,
                        points,
                    ),
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
#   - models/detector/best.pt 또는 detector.tflite (혹은 CNN 모델 폴더 안):
#     넣어두면 위치검출이 활성화됨 (없어도 그냥 동작함)
#   - FIXED_ROI: last_capture.jpg 확인하면서 좌표 조정 (YOLO 없을 때 대비용)
#   - 모니터가 연결된 Jetson 본체 데스크톱(또는 VNC)에서 실행해야 미리보기 창이 뜸
# =====================================================================
