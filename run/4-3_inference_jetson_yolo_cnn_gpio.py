# =====================================================================
# 4-3단계: Jetson Nano 실시간 추론 + 컨베이어/서보 제어 통합 스크립트
#        (YOLO 위치검출 + CNN 분류 두 단계 버전, GPIO 직접 제어, STM32 없음)
# =====================================================================
# 4-1(CNN 단독)과 거의 같은 구조인데, CNN에 넣기 전에 YOLO로 물체 위치를 먼저
# 찾아서 그 부분만 크롭한 뒤 분류합니다. 배경이 섞여 들어가는 걸 줄여서
# 분류 정확도를 올리려는 목적입니다.
#
# YOLO 위치검출은 두 가지 방식을 모두 지원하고, 아래 우선순위로 자동 선택합니다.
#   1) models/detector/best.pt      - ultralytics로 바로 로딩 (torch/ultralytics가
#      Jetson에 이미 설치돼 있으면 변환 없이 이걸 그대로 씀. 설치 안 돼있으면 실패하고
#      2번으로 넘어감)
#   2) models/detector/detector.tflite - tflite_runtime으로 로딩 (가볍고 설치가
#      간단하지만, best.pt를 미리 Colab 등에서 tflite로 변환해둬야 함.
#      1_train_yolo_colab.py에 변환 단계가 있습니다.)
#   둘 다 없으면: 위치검출 없이 4-1처럼 전체 프레임(또는 FIXED_ROI)을 그대로 CNN에
#   넣습니다 - 즉 이 스크립트는 YOLO 없이도 그냥 실행됩니다.
#
# 폴더 구조 (3개 폴더로 분리, models/는 모델별 하위 폴더 + detector 폴더로 구성)
#   project/
#   ├── models/
#   │   ├── model_freeze/     classifier.tflite, class_names.txt   (CNN)
#   │   ├── model_finetune/   classifier.tflite, class_names.txt   (CNN)
#   │   └── detector/         best.pt 또는 detector.tflite           (YOLO, 선택)
#   ├── run/     이 스크립트 + 4-x 계열 스크립트들 + model_select.py
#   └── db/      points.py, db_setup.py, db_view.py, *.db
#
# YOLO 파일(best.pt / detector.tflite) 위치는 위 detector/ 폴더가 기본이지만,
# 지금 실행 시 고른 CNN 모델 폴더(예: models/3class/) 안에 같이 넣어둬도
# 자동으로 인식합니다. 두 곳 다 있으면 CNN 모델 폴더 쪽을 먼저 씁니다.
#
# 실행할 때 어떤 CNN 모델을 쓸지 고를 수 있습니다.
#   python3 4-3_inference_jetson_yolo_cnn_gpio.py                 실행 중 번호로 선택
#   python3 4-3_inference_jetson_yolo_cnn_gpio.py model_finetune   바로 지정
#
# Jetson Nano에 설치 필요 (최초 1회)
#   pip install opencv-python
#   pip install tflite-runtime          # detector.tflite 쓸 경우 (가벼움)
#   pip install ultralytics             # best.pt를 직접 쓸 경우 (torch도 필요,
#                                        # JetPack용 PyTorch wheel 별도 설치 필요)
#   sudo pip install Jetson.GPIO        # 보통 JetPack에 기본 포함되어 있음
#
# 설치 확인 (best.pt를 바로 쓰고 싶다면 이게 에러 없이 실행되는지 먼저 확인):
#   python3 -c "import torch, ultralytics; print(torch.__version__, ultralytics.__version__)"
#
# 아래 "설정값" 구간은 전부 실제 하드웨어에 맞게 실측해서 채워야 합니다.
# (GPIO 핀 번호, 벨트 속도, 분기 지점까지 거리, 서보 각도 등)
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
# 설정값 (실측 후 수정 필요)
# ---------------------------------------------------------------------
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

# 고정 ROI (선택): YOLO가 물체를 못 찾았을 때(또는 detector.tflite가 없을 때)
# 대신 쓰는 영역. (x1, y1, x2, y2) 픽셀 좌표. 모르면 None으로 두면 전체 프레임.
FIXED_ROI = None   # 예: (150, 80, 450, 380)

DETECTOR_DIR = os.path.join(MODELS_DIR, "detector")
DETECTOR_PT_PATH = os.path.join(DETECTOR_DIR, "best.pt")            # ultralytics 직접 로딩용
DETECTOR_TFLITE_PATH = os.path.join(DETECTOR_DIR, "detector.tflite")  # tflite 변환본
DETECTOR_CONF_THRESHOLD = 0.4         # YOLO 검출 최소 신뢰도
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


def _ensure_column(conn, table, column, coltype):
    """테이블이 예전 버전 스크립트로 이미 만들어져 있어서 컬럼이 없을 경우 추가.
    (CREATE TABLE IF NOT EXISTS는 테이블이 이미 있으면 그냥 넘어가버려서
    새로 추가된 컬럼이 반영이 안 됨 - 4-1/4-2로 먼저 만든 DB를 4-3에서
    이어서 쓸 때 필요한 처리)"""
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
            servo_used TEXT,
            points_awarded INTEGER NOT NULL DEFAULT 0
        )
        """
    )
    # 예전 스크립트(4-1 등)로 이미 만들어진 sorting_log.db를 이어서 쓰는
    # 경우를 대비한 마이그레이션 (기존 행은 이 컬럼들이 NULL/0으로 채워짐)
    _ensure_column(conn, "sorting_log", "yolo_confidence", "REAL")
    _ensure_column(conn, "sorting_log", "servo_used", "TEXT")
    _ensure_column(conn, "sorting_log", "points_awarded", "INTEGER NOT NULL DEFAULT 0")

    init_users_table(conn)  # points.py: 사용자별 누적 포인트 테이블
    conn.commit()
    return conn


def load_class_names(class_names_path):
    with open(class_names_path, encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


class TFLiteDetector:
    """detector.tflite(ultralytics tflite export) 기반 검출기."""

    def __init__(self, path):
        self.interpreter = Interpreter(model_path=path)
        self.interpreter.allocate_tensors()

    def detect_and_crop(self, frame, conf_threshold=DETECTOR_CONF_THRESHOLD):
        return detect_and_crop_tflite(self.interpreter, frame, conf_threshold)


class UltralyticsDetector:
    """best.pt를 ultralytics로 바로 로딩하는 검출기 (torch 설치 필요)."""

    def __init__(self, path):
        from ultralytics import YOLO  # 여기서만 import (torch 없으면 여기서 실패해서 폴백됨)
        self.model = YOLO(path)

    def detect_and_crop(self, frame, conf_threshold=DETECTOR_CONF_THRESHOLD):
        results = self.model.predict(frame, conf=conf_threshold, verbose=False)
        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            return None, 0.0

        # 한 번에 하나씩 투입된다는 가정으로, 신뢰도 가장 높은 박스 하나만 사용
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
            return None, conf
        return crop, conf


def load_detector(extra_dir=None):
    """best.pt(ultralytics) 우선 시도 -> 실패/없으면 detector.tflite -> 그것도
    없으면 None (위치검출 없이 4-1처럼 동작).

    파일은 models/detector/ 안에 있어도 되고, 지금 선택한 CNN 모델 폴더
    (models/모델이름/, extra_dir로 전달됨) 안에 같이 넣어도 인식됩니다.
    두 곳 다 있으면 CNN 모델 폴더 쪽을 먼저 확인합니다.
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
    print("-> 위치검출 없이 진행 (4-1과 동일 동작)")
    return None


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
# YOLO 위치 검출 (tflite 버전) -> 가장 신뢰도 높은 박스 하나만 크롭
# (한 번에 하나씩 투입되는 구조라는 가정. detector가 없으면 호출 안 됨)
# ---------------------------------------------------------------------
def detect_and_crop_tflite(interpreter, frame, conf_threshold=DETECTOR_CONF_THRESHOLD):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    in_h, in_w = input_details[0]["shape"][1:3]

    img = cv2.resize(frame, (in_w, in_h))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=0)

    interpreter.set_tensor(input_details[0]["index"], img)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]["index"])[0]

    # ultralytics tflite 출력은 보통 (4+클래스수, 박스개수) 형태.
    # 박스개수(수천개)가 훨씬 크므로, 작은 축이 앞에 오도록 필요하면 전치.
    if output.shape[0] < output.shape[1]:
        output = output.T  # -> (박스개수, 4+클래스수)

    boxes = output[:, :4]
    scores = output[:, 4:]
    if scores.size == 0:
        return None, 0.0
    confidences = np.max(scores, axis=1)

    best_idx = int(np.argmax(confidences))
    best_conf = float(confidences[best_idx])
    if best_conf < conf_threshold:
        return None, best_conf

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
        return None, best_conf
    return crop, best_conf


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

    # models/ 하위 폴더 중 어떤 CNN 모델을 쓸지 선택 (인자로 지정 안 하면 번호로 선택)
    model_path, class_names_path = select_model(MODELS_DIR)
    class_names = load_class_names(class_names_path)

    print("CNN 모델 로딩...")
    cnn_interpreter = Interpreter(model_path=model_path)
    cnn_interpreter.allocate_tensors()

    # 선택된 CNN 모델 폴더 안도 같이 뒤져봄 (models/detector/ 뿐 아니라
    # models/모델이름/ 안에 best.pt를 넣어도 인식되게)
    detector = load_detector(extra_dir=os.path.dirname(model_path))

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

                # 3) YOLO로 위치 검출 + 크롭 (detector 없으면 고정ROI/전체프레임 사용)
                yolo_conf = 0.0
                image = None
                if detector is not None:
                    image, yolo_conf = detector.detect_and_crop(frame)
                if image is None:
                    image = apply_roi(frame)

                # 4) CNN으로 최종 분류
                cnn_class, cnn_conf = classify(cnn_interpreter, image, class_names)
                print(f"YOLO conf={yolo_conf:.2f} / CNN class={cnn_class} conf={cnn_conf:.2f}")

                # 5) 신뢰도 확인 -> 최종 클래스 확정
                if cnn_conf < CNN_CONF_THRESHOLD:
                    final_class = "unknown"
                    servo_used = "none"
                    print("판단 불확실 -> 서보 작동 없이 기본 경로로 흘려보냄")
                else:
                    final_class = cnn_class
                    servo_used = final_class

                # 6) 벨트 재가동 + 계산된 대기시간 후 서보 작동
                start_belt()
                if servo_used in SERVO_PINS:
                    distance = DIVERT_DISTANCES_M[servo_used]
                    wait_time = distance / BELT_SPEED_MPS
                    time.sleep(wait_time)
                    actuate_servo(SERVO_PINS[servo_used])

                # 7) 포인트 적립 (전화번호가 입력되어 있을 때만)
                phone = current_user["phone"]
                points = 0
                if phone:
                    points = award_points(conn, phone, final_class)
                    if points > 0:
                        print(f"[포인트] {phone}님 +{points}점 적립!")

                # 8) DB 기록
                conn.execute(
                    """
                    INSERT INTO sorting_log
                        (timestamp, phone, yolo_confidence, cnn_class, cnn_confidence,
                         final_class, servo_used, points_awarded)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        datetime.now().isoformat(),
                        phone,
                        yolo_conf,
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
#   - FIXED_ROI: YOLO가 없거나 못 찾았을 때 대신 쓸 좌표 (선택 사항)
#   - 서보 듀티사이클(2.5 / 7.5): 사용하는 서보 스펙시트 참고해서 조정
#   - CNN_CONF_THRESHOLD / DETECTOR_CONF_THRESHOLD: 실제 테스트하면서 조정
#   - models/detector/best.pt 또는 models/detector/detector.tflite 를 넣어두면
#     위치검출이 활성화됩니다 (best.pt가 있으면 그걸 우선 시도, 안 되면 tflite로
#     폴백, 둘 다 없으면 4-1처럼 위치검출 없이 동작)
# =====================================================================
