# =====================================================================
# 재활용 분류 시스템 - 통합 GUI 대시보드 (PyQt5, 하이브리드 버전)
# =====================================================================
# YOLO 먼저 시도하고, 확신도가 낮아 unknown이면 CNN으로 재시도합니다.
# (CNN 쪽은 models/ 안에 모델이 여러 개면 그것도 순서대로 재시도합니다 -
#  gui_dashboard.py의 다중모델 재시도 기능과 동일)
#
# 판단 순서: YOLO 시도 -> 확신도 충분하면 그 결과로 확정
#            -> 부족하면(unknown) CNN 시도(모델 여러 개면 순서대로) -> 확정
#            -> 그것도 다 부족하면 최종 unknown
#
# 화면에는 최종 결정뿐 아니라 YOLO/CNN 각각 뭐라고 판단했는지도 같이
# 표시해서, 두 모델이 얼마나 일치/불일치하는지 비교해볼 수 있습니다.
#
# YOLO는 선택 사항입니다 - models/detector/ (또는 models/아무폴더/)에
# best.pt나 detector.tflite가 없으면 자동으로 건너뛰고 CNN만 사용합니다.
# CNN 모델은 필수입니다 (models/모델이름/classifier.tflite + class_names.txt).
#
# 클래스 이름은 정확히 "metal"/"plastic"/"paper"(소문자)여야 포인트가
# 정상 적립됩니다 (db/points.py의 POINTS_PER_CLASS 참고).
#
# 트리거(촬영) 방식은 두 가지를 모두 지원합니다.
#   1) STM32가 SERIAL_PORT로 연결돼 있으면: "TRIGGER" 신호가 오면 자동 촬영
#   2) STM32가 없거나 연결에 실패해도: 화면의 "지금 촬영" 버튼으로 수동 촬영
#
# 설치 (Jetson Nano에서는 pip보다 apt 패키지가 훨씬 빠르게 설치됩니다):
#   sudo apt install python3-pyqt5 python3-matplotlib fonts-nanum
#   pip3 install pyserial opencv-python --break-system-packages
#   pip3 install tflite-runtime          # CNN + detector.tflite 쓸 경우
#   pip3 install ultralytics             # best.pt 직접 쓸 경우 (torch 필요)
#
# 실행 (run/ 폴더 안에서):
#   python3 gui_dashboard_hybrid.py                 CNN 모델 번호로 선택
#   python3 gui_dashboard_hybrid.py model_finetune   CNN 모델 바로 지정
#
# 폴더 구조
#   project/
#   ├── models/
#   │   ├── model_xxx/    classifier.tflite, class_names.txt   (CNN, 필수)
#   │   └── detector/      best.pt 또는 detector.tflite          (YOLO, 선택)
#   ├── run/     이 스크립트 + 4-x 스크립트들 + model_select.py
#   └── db/      points.py, db_setup.py, db_view.py, *.db
# =====================================================================

import os
import sys
import time
import sqlite3
from datetime import datetime

import numpy as np
import cv2

from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QLineEdit, QPushButton,
    QVBoxLayout, QHBoxLayout, QTableWidget, QTableWidgetItem, QGroupBox,
    QHeaderView, QMessageBox, QSplitter, QSizePolicy,
)

import matplotlib
matplotlib.use("Qt5Agg")
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure

# 한글 폰트 설정 (Jetson에 나눔고딕 등이 없으면 그래프 글자가 네모(□)로 보일 수 있음.
# 그럴 땐 `sudo apt install fonts-nanum` 설치 후 아래 폰트명이 맞는지 확인하세요.)
matplotlib.rcParams["font.family"] = "NanumGothic"
matplotlib.rcParams["axes.unicode_minus"] = False

try:
    import serial
except ImportError:
    serial = None

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.normpath(os.path.join(BASE_DIR, "..", "models"))
DB_DIR = os.path.normpath(os.path.join(BASE_DIR, "..", "db"))

sys.path.append(DB_DIR)
from points import init_users_table, award_points, is_valid_phone, get_leaderboard, get_points
from model_select import select_model, list_available_models

try:
    from tflite_runtime.interpreter import Interpreter
except ImportError:
    from tensorflow.lite.python.interpreter import Interpreter


# ---------------------------------------------------------------------
# 설정값
# ---------------------------------------------------------------------
SERIAL_PORT = "/dev/ttyTHS1"     # STM32 핀헤더 UART. USB 연결이면 /dev/ttyACM0 등으로 변경
SERIAL_BAUDRATE = 115200
FIXED_ROI = None                  # YOLO가 없거나 못 찾았을 때 CNN에 쓸 좌표. 예: (150, 80, 450, 380)
CNN_CONF_THRESHOLD = 0.6
CAMERA_INDEX = 0
DB_PATH = os.path.join(DB_DIR, "sorting_log.db")
LAST_CAPTURE_PATH = os.path.join(BASE_DIR, "last_capture.jpg")
RECENT_LOG_ROWS = 15

DETECTOR_DIR = os.path.join(MODELS_DIR, "detector")
DETECTOR_CONF_THRESHOLD = 0.5    # 이 이상이면 YOLO 결과를 바로 최종으로 사용
DETECTOR_INTERVAL_MS = 250       # 미리보기용 박스 갱신 주기


def _ensure_column(conn, table, column, coltype):
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
            yolo_class TEXT,
            yolo_confidence REAL,
            final_class TEXT,
            points_awarded INTEGER NOT NULL DEFAULT 0
        )
        """
    )
    _ensure_column(conn, "sorting_log", "yolo_class", "TEXT")
    _ensure_column(conn, "sorting_log", "yolo_confidence", "REAL")
    _ensure_column(conn, "sorting_log", "points_awarded", "INTEGER NOT NULL DEFAULT 0")
    init_users_table(conn)
    conn.commit()
    return conn


def load_class_names(class_names_path):
    with open(class_names_path, encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def classify(interpreter, image, class_names):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    height, width = input_details[0]["shape"][1:3]

    img = cv2.resize(image, (width, height))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
    img = np.expand_dims(img, axis=0)

    # 모델(.tflite) 안에 Rescaling 레이어가 이미 포함되어 있으므로
    # 여기서 픽셀값을 추가로 정규화하면 안 됨 (이중정규화 버그 주의)
    interpreter.set_tensor(input_details[0]["index"], img)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]["index"])[0]

    idx = int(np.argmax(output))
    return class_names[idx], float(output[idx])


def apply_roi(frame):
    if FIXED_ROI is None:
        return frame
    x1, y1, x2, y2 = FIXED_ROI
    return frame[y1:y2, x1:x2]


def fetch_recent_log(conn, limit=RECENT_LOG_ROWS):
    cursor = conn.execute(
        "SELECT id, timestamp, phone, cnn_class, cnn_confidence, yolo_class, "
        "yolo_confidence, final_class, points_awarded "
        "FROM sorting_log ORDER BY id DESC LIMIT ?",
        (limit,),
    )
    return cursor.fetchall()


def fetch_class_counts(conn, phone=None):
    if phone:
        cursor = conn.execute(
            "SELECT final_class, COUNT(*) FROM sorting_log WHERE phone = ? GROUP BY final_class",
            (phone,),
        )
    else:
        cursor = conn.execute(
            "SELECT final_class, COUNT(*) FROM sorting_log GROUP BY final_class"
        )
    return dict(cursor.fetchall())


# ---------------------------------------------------------------------
# YOLO 검출기 (위치 + 분류를 한 번에 반환) - 4-5/gui_dashboard_yolo_only와 동일
# ---------------------------------------------------------------------
def load_labels(path):
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


class TFLiteDetector:
    def __init__(self, path, labels_path):
        self.interpreter = Interpreter(model_path=path)
        self.interpreter.allocate_tensors()
        self.labels = load_labels(labels_path)
        if self.labels is None:
            print(f"[경고] {labels_path} 없음 - 클래스 이름 대신 class_0, class_1...로 표시됩니다")

    def _class_name(self, class_id):
        if self.labels and 0 <= class_id < len(self.labels):
            return self.labels[class_id]
        return f"class_{class_id}"

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
            output = output.T

        boxes = output[:, :4]
        scores = output[:, 4:]
        if scores.size == 0:
            return None, 0.0, None, None

        class_ids = np.argmax(scores, axis=1)
        confidences = np.max(scores, axis=1)

        best_idx = int(np.argmax(confidences))
        best_conf = float(confidences[best_idx])
        best_class_id = int(class_ids[best_idx])
        if best_conf < conf_threshold:
            return None, best_conf, None, self._class_name(best_class_id)

        cx, cy, w, h = boxes[best_idx]
        frame_h, frame_w = frame.shape[:2]

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
        class_name = self._class_name(best_class_id)
        if crop.size == 0:
            return None, best_conf, None, class_name
        return crop, best_conf, (x1, y1, x2, y2), class_name


class UltralyticsDetector:
    def __init__(self, path):
        from ultralytics import YOLO
        self.model = YOLO(path)

    def detect_and_crop(self, frame, conf_threshold=DETECTOR_CONF_THRESHOLD):
        results = self.model.predict(frame, conf=conf_threshold, verbose=False)
        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            return None, 0.0, None, None

        best_idx = int(boxes.conf.argmax().item())
        conf = float(boxes.conf[best_idx])
        class_id = int(boxes.cls[best_idx].item())
        class_name = self.model.names.get(class_id, f"class_{class_id}")
        x1, y1, x2, y2 = boxes.xyxy[best_idx].tolist()

        frame_h, frame_w = frame.shape[:2]
        x1 = max(0, int(x1))
        y1 = max(0, int(y1))
        x2 = min(frame_w, int(x2))
        y2 = min(frame_h, int(y2))

        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return None, conf, None, class_name
        return crop, conf, (x1, y1, x2, y2), class_name


def _find_candidate_dirs():
    dirs = [DETECTOR_DIR]
    if os.path.isdir(MODELS_DIR):
        for name in sorted(os.listdir(MODELS_DIR)):
            folder = os.path.join(MODELS_DIR, name)
            if os.path.isdir(folder) and folder not in dirs:
                dirs.append(folder)
    return dirs


def find_detector_candidates():
    """best.pt 또는 detector.tflite가 있는 폴더를 전부 찾아서
    (폴더경로, 종류) 목록으로 반환. 종류는 'pt' 또는 'tflite'."""
    candidates = []
    for d in _find_candidate_dirs():
        if os.path.exists(os.path.join(d, "best.pt")):
            candidates.append((d, "pt"))
        if os.path.exists(os.path.join(d, "detector.tflite")):
            candidates.append((d, "tflite"))
    return candidates


def _load_candidate(folder, kind):
    if kind == "pt":
        path = os.path.join(folder, "best.pt")
        print(f"YOLO(.pt, ultralytics) 로딩... ({path})")
        return UltralyticsDetector(path)
    else:
        path = os.path.join(folder, "detector.tflite")
        labels_path = os.path.join(folder, "detector_labels.txt")
        print(f"YOLO(.tflite) 로딩... ({path})")
        return TFLiteDetector(path, labels_path)


def load_detector():
    """YOLO는 선택 사항 - 없으면 None을 반환하고 CNN만 사용합니다. 후보가
    여러 개면 번호로 고르게 합니다. (CNN 모델 선택은 이미 sys.argv[1]을 쓰고
    있어서, YOLO 쪽은 커맨드라인 인자 없이 항상 번호 입력으로만 고릅니다.)"""
    candidates = find_detector_candidates()
    if not candidates:
        print("YOLO 모델 없음 - CNN만 사용합니다 (그래도 정상 동작합니다)")
        return None

    if len(candidates) == 1:
        chosen = candidates[0]
    else:
        print("\n사용 가능한 YOLO 모델:")
        for i, (d, kind) in enumerate(candidates, 1):
            print(f"  {i}. {os.path.basename(d)} ({kind})")
        while True:
            sel = input("사용할 YOLO 모델 번호 입력 > ").strip()
            if sel.isdigit() and 1 <= int(sel) <= len(candidates):
                chosen = candidates[int(sel) - 1]
                break
            print("올바른 번호를 입력하세요.")

    folder, kind = chosen
    print(f"[YOLO 모델 선택] {os.path.basename(folder)} ({kind})")
    try:
        return _load_candidate(folder, kind)
    except Exception as e:
        print(f"로딩 실패 (CNN만 사용합니다): {e}")
        return None


# ---------------------------------------------------------------------
# STM32 시리얼 리스너
# ---------------------------------------------------------------------
class SerialListener(QThread):
    triggered = pyqtSignal()
    status = pyqtSignal(str)

    def __init__(self, port, baudrate):
        super().__init__()
        self.port = port
        self.baudrate = baudrate
        self._running = True
        self.ser = None

    def run(self):
        if serial is None:
            self.status.emit("pyserial 미설치 - 자동 트리거 비활성화 (지금 촬영 버튼 사용)")
            return
        try:
            self.ser = serial.Serial(self.port, self.baudrate, timeout=1)
            self.ser.reset_input_buffer()
            time.sleep(2)
            self.status.emit(f"STM32 연결됨 ({self.port})")
        except serial.SerialException as e:
            self.status.emit(f"STM32 연결 실패({e}) - 지금 촬영 버튼으로 진행하세요")
            return

        while self._running:
            try:
                line = self.ser.readline().decode("utf-8", errors="ignore").strip()
            except serial.SerialException:
                self.status.emit("STM32 연결이 끊겼습니다.")
                return
            if line == "TRIGGER":
                self.triggered.emit()

    def send_class(self, final_class):
        if self.ser and self.ser.is_open:
            try:
                self.ser.write(f"CLASS:{final_class}\n".encode("utf-8"))
            except serial.SerialException:
                pass

    def stop(self):
        self._running = False
        if self.ser and self.ser.is_open:
            self.ser.close()


# ---------------------------------------------------------------------
# 메인 대시보드 창
# ---------------------------------------------------------------------
class Dashboard(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("재활용 분류 시스템 - 대시보드 (YOLO→CNN 하이브리드)")
        self.resize(1400, 800)

        self.conn = init_db()

        # --- CNN 모델(들) 로딩 (필수, 여러 개면 순서대로 재시도 대상) ---
        model_path, class_names_path = select_model(MODELS_DIR)
        print("CNN 모델 로딩...")
        self.cnn_models = [self._load_cnn_model("기본", model_path, class_names_path)]
        for name in list_available_models(MODELS_DIR):
            folder = os.path.join(MODELS_DIR, name)
            other_model_path = os.path.join(folder, "classifier.tflite")
            other_class_names_path = os.path.join(folder, "class_names.txt")
            if os.path.normpath(other_model_path) == os.path.normpath(model_path):
                continue
            print(f"대체 CNN 모델 로딩: {name}")
            self.cnn_models.append(self._load_cnn_model(name, other_model_path, other_class_names_path))

        # --- YOLO 검출기 (선택 사항) ---
        print("YOLO 모델 로딩 시도...")
        self.detector = load_detector()

        self.current_phone = None
        self.last_frame = None
        self.detected_box = None  # (x1, y1, x2, y2, class_name, conf) - 미리보기용
        self.serial_listener = None

        self.cap = cv2.VideoCapture(CAMERA_INDEX)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not self.cap.isOpened():
            QMessageBox.critical(self, "오류", f"카메라(index={CAMERA_INDEX})를 열 수 없습니다.")
        for _ in range(10):
            self.cap.read()

        self._build_ui()

        self.preview_timer = QTimer(self)
        self.preview_timer.timeout.connect(self._update_preview)
        self.preview_timer.start(30)

        if self.detector is not None:
            self.detect_timer = QTimer(self)
            self.detect_timer.timeout.connect(self._run_detection)
            self.detect_timer.start(DETECTOR_INTERVAL_MS)

        self.db_timer = QTimer(self)
        self.db_timer.timeout.connect(self._refresh_db_views)
        self.db_timer.start(3000)
        self._refresh_db_views()

        self.serial_listener = SerialListener(SERIAL_PORT, SERIAL_BAUDRATE)
        self.serial_listener.triggered.connect(self._on_trigger)
        self.serial_listener.status.connect(self._set_status)
        self.serial_listener.start()

    # ---------------- UI 구성 ----------------
    def _build_ui(self):
        self.setStyleSheet(
            """
            QGroupBox {
                font-weight: bold;
                border: 1px solid #d0d0d0;
                border-radius: 8px;
                margin-top: 14px;
                padding: 10px;
                background: #fafafa;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 6px;
                color: #333;
            }
            QPushButton {
                padding: 6px 10px;
                border-radius: 6px;
            }
            """
        )

        outer = QVBoxLayout(self)
        outer.setContentsMargins(16, 16, 16, 16)
        outer.setSpacing(12)

        title = QLabel("재활용 분류 시스템 대시보드 (YOLO → CNN 하이브리드)")
        title.setStyleSheet("font-size:20px; font-weight:bold;")
        outer.addWidget(title)

        splitter = QSplitter(Qt.Horizontal)
        splitter.setChildrenCollapsible(False)
        outer.addWidget(splitter, 1)

        # ---------------- 왼쪽: 카메라 미리보기 ----------------
        left_widget = QWidget()
        left = QVBoxLayout(left_widget)
        left.setSpacing(10)

        preview_box = QGroupBox("카메라 미리보기")
        preview_layout = QVBoxLayout()
        self.preview_label = QLabel("카메라 로딩 중...")
        self.preview_label.setMinimumSize(480, 360)
        self.preview_label.setStyleSheet("background:#222; color:white; border-radius:6px;")
        self.preview_label.setAlignment(Qt.AlignCenter)
        self.preview_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        preview_layout.addWidget(self.preview_label)
        preview_box.setLayout(preview_layout)
        left.addWidget(preview_box, 1)

        self.status_label = QLabel("상태: 대기 중")
        self.status_label.setStyleSheet("color:#555; padding:4px 2px;")
        left.addWidget(self.status_label)

        splitter.addWidget(left_widget)

        # ---------------- 가운데: 사용자 조작 + 결과 + 나의 포인트 ----------------
        mid_widget = QWidget()
        mid = QVBoxLayout(mid_widget)
        mid.setSpacing(12)

        control_box = QGroupBox("사용자 조작")
        control_layout = QVBoxLayout()
        control_layout.setSpacing(8)

        phone_row = QHBoxLayout()
        self.phone_input = QLineEdit()
        self.phone_input.setPlaceholderText("전화번호 입력 (숫자만, 4자리 이상)")
        self.phone_input.returnPressed.connect(self._apply_phone)
        apply_btn = QPushButton("적용")
        apply_btn.clicked.connect(self._apply_phone)
        guest_btn = QPushButton("게스트로")
        guest_btn.clicked.connect(self._set_guest)
        phone_row.addWidget(self.phone_input)
        phone_row.addWidget(apply_btn)
        phone_row.addWidget(guest_btn)
        control_layout.addLayout(phone_row)

        self.current_user_label = QLabel("현재 사용자: 게스트")
        self.current_user_label.setStyleSheet("color:#555;")
        control_layout.addWidget(self.current_user_label)

        capture_btn = QPushButton("지금 촬영 (수동 트리거)")
        capture_btn.setStyleSheet(
            "font-weight:bold; padding:10px; background:#2e7d32; color:white;"
        )
        capture_btn.clicked.connect(self._on_trigger)
        control_layout.addWidget(capture_btn)

        control_box.setLayout(control_layout)
        mid.addWidget(control_box)

        # 최근 분류 결과 (최종 결정 + YOLO/CNN 각각의 판단 비교)
        result_box = QGroupBox("최근 분류 결과")
        result_layout = QVBoxLayout()
        self.result_label = QLabel("-")
        self.result_label.setStyleSheet("font-size:20px; font-weight:bold;")
        self.result_label.setAlignment(Qt.AlignCenter)
        self.detail_label = QLabel("")
        self.detail_label.setStyleSheet("color:#666; font-size:12px;")
        self.detail_label.setWordWrap(True)
        result_layout.addWidget(self.result_label)
        result_layout.addWidget(self.detail_label)
        result_box.setLayout(result_layout)
        mid.addWidget(result_box)

        my_box = QGroupBox("나의 포인트 현황")
        my_layout = QVBoxLayout()
        self.my_points_label = QLabel("게스트 - 전화번호를 입력하면 포인트가 적립/조회됩니다")
        self.my_points_label.setStyleSheet("font-size:16px; font-weight:bold; color:#2e7d32;")
        self.my_points_label.setWordWrap(True)
        self.my_breakdown_label = QLabel("")
        self.my_breakdown_label.setStyleSheet("color:#555;")
        self.my_breakdown_label.setWordWrap(True)
        my_layout.addWidget(self.my_points_label)
        my_layout.addWidget(self.my_breakdown_label)
        my_box.setLayout(my_layout)
        mid.addWidget(my_box, 1)

        splitter.addWidget(mid_widget)

        # ---------------- 오른쪽: DB 결과 대시보드 ----------------
        right_widget = QWidget()
        right = QVBoxLayout(right_widget)
        right.setSpacing(12)

        log_box = QGroupBox("최근 분류 기록 (DB)")
        log_layout = QVBoxLayout()
        self.log_table = QTableWidget(0, 7)
        self.log_table.setHorizontalHeaderLabels(
            ["시각", "전화번호", "YOLO", "CNN", "최종", "포인트", "판정"]
        )
        self.log_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.log_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.log_table.setAlternatingRowColors(True)
        log_layout.addWidget(self.log_table)
        log_box.setLayout(log_layout)
        right.addWidget(log_box, 3)

        rank_box = QGroupBox("포인트 랭킹 (상위 10명, DB)")
        rank_layout = QVBoxLayout()
        self.rank_table = QTableWidget(0, 2)
        self.rank_table.setHorizontalHeaderLabels(["전화번호", "포인트"])
        self.rank_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.rank_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.rank_table.setAlternatingRowColors(True)
        rank_layout.addWidget(self.rank_table)
        rank_box.setLayout(rank_layout)
        right.addWidget(rank_box, 2)

        stats_box = QGroupBox("통계 그래프 (전체 배출 건수)")
        stats_layout = QVBoxLayout()
        self.stats_figure = Figure(figsize=(4, 2.6))
        self.stats_canvas = FigureCanvasQTAgg(self.stats_figure)
        self.stats_ax = self.stats_figure.add_subplot(111)
        stats_layout.addWidget(self.stats_canvas)
        stats_box.setLayout(stats_layout)
        right.addWidget(stats_box, 3)

        splitter.addWidget(right_widget)

        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 3)
        splitter.setStretchFactor(2, 4)

    # ---------------- 모델 로딩 / CNN 다중모델 재시도 ----------------
    def _load_cnn_model(self, name, model_path, class_names_path):
        interpreter = Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        return {
            "name": name,
            "interpreter": interpreter,
            "class_names": load_class_names(class_names_path),
        }

    def _classify_with_cnn_fallback(self, image):
        """CNN 모델을 1순위부터 시도. 확신도 기준 미달이면 다음 모델로 재시도.
        반환: (model_name, cnn_class, cnn_conf) - 전부 미달이면 그중 최고값."""
        best = None
        for model in self.cnn_models:
            cnn_class, cnn_conf = classify(model["interpreter"], image, model["class_names"])
            print(f"  [CNN:{model['name']}] class={cnn_class} conf={cnn_conf:.2f}")
            if cnn_conf >= CNN_CONF_THRESHOLD:
                return model["name"], cnn_class, cnn_conf
            if best is None or cnn_conf > best[2]:
                best = (model["name"], cnn_class, cnn_conf)
        return best

    def _classify_hybrid(self, frame):
        """YOLO 먼저 시도 -> 확신도 충분하면 그걸로 확정. 부족하면(unknown) CNN
        (다중모델 포함)으로 재시도. 반환값: dict로 모든 중간결과를 담아서 UI/DB에서
        재사용."""
        result = {
            "yolo_class": None, "yolo_conf": 0.0, "box": None,
            "cnn_model": None, "cnn_class": None, "cnn_conf": None,
            "final_class": "unknown", "method": "",
        }

        crop = None
        if self.detector is not None:
            crop, yolo_conf, box, yolo_class = self.detector.detect_and_crop(frame)
            result["yolo_class"] = yolo_class
            result["yolo_conf"] = yolo_conf
            result["box"] = box
            print(f"  [YOLO] class={yolo_class} conf={yolo_conf:.2f}")

            if yolo_class is not None and box is not None and yolo_conf >= DETECTOR_CONF_THRESHOLD:
                result["final_class"] = yolo_class.strip().lower()
                result["method"] = "YOLO"
                return result

        # YOLO가 없거나 확신도 부족 -> CNN으로 재시도
        # (YOLO가 크롭을 줬으면 그 영역을, 아니면 고정 ROI/전체 프레임을 CNN에 입력)
        image = crop if crop is not None else apply_roi(frame)
        cnn_model, cnn_class, cnn_conf = self._classify_with_cnn_fallback(image)
        result["cnn_model"] = cnn_model
        result["cnn_class"] = cnn_class
        result["cnn_conf"] = cnn_conf

        if cnn_conf >= CNN_CONF_THRESHOLD:
            result["final_class"] = cnn_class
            result["method"] = f"CNN({cnn_model})"
        else:
            result["final_class"] = "unknown"
            result["method"] = "YOLO+CNN 둘 다 확신도 부족"

        return result

    # ---------------- 동작 ----------------
    def _apply_phone(self):
        text = self.phone_input.text().strip()
        if is_valid_phone(text):
            self.current_phone = text
            self.current_user_label.setText(f"현재 사용자: {text}")
            self._set_status(f"사용자 전환: {text}")
            self._refresh_my_points()
        else:
            QMessageBox.warning(self, "입력 오류", "전화번호는 숫자만, 4자리 이상 입력해주세요.")

    def _set_guest(self):
        self.current_phone = None
        self.phone_input.clear()
        self.current_user_label.setText("현재 사용자: 게스트")
        self._set_status("게스트로 전환")
        self._refresh_my_points()

    def _set_status(self, text):
        self.status_label.setText(f"상태: {text}")

    def _update_preview(self):
        ret, frame = self.cap.read()
        if not ret:
            return
        self.last_frame = frame

        display = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if FIXED_ROI is not None:
            x1, y1, x2, y2 = FIXED_ROI
            cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2)

        if self.detected_box is not None:
            x1, y1, x2, y2, class_name, conf = self.detected_box
            cv2.rectangle(display, (x1, y1), (x2, y2), (255, 220, 0), 2)
            label = f"{class_name} {conf:.2f}" if class_name else f"{conf:.2f}"
            cv2.putText(
                display, label, (x1, max(0, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 220, 0), 2,
            )

        h, w, ch = display.shape
        qimg = QImage(display.data, w, h, ch * w, QImage.Format_RGB888)
        pix = QPixmap.fromImage(qimg).scaled(
            self.preview_label.width(), self.preview_label.height(), Qt.KeepAspectRatio
        )
        self.preview_label.setPixmap(pix)

    def _run_detection(self):
        """미리보기용 - YOLO 박스만 계속 갱신 (최종 판정은 트리거 시 별도로 다시 함)"""
        if self.detector is None or self.last_frame is None:
            return
        try:
            _, conf, box, class_name = self.detector.detect_and_crop(self.last_frame)
            self.detected_box = (box[0], box[1], box[2], box[3], class_name, conf) if box else None
        except Exception as e:
            print(f"검출 중 오류(무시하고 계속): {e}")

    def _on_trigger(self):
        if self.last_frame is None:
            self._set_status("아직 카메라 프레임이 없습니다.")
            return

        frame = self.last_frame
        cv2.imwrite(LAST_CAPTURE_PATH, frame)  # ROI 조정용 디버그 저장

        r = self._classify_hybrid(frame)
        final_class = r["final_class"]

        self.result_label.setText(f"{final_class}  ({r['method']})")
        yolo_txt = f"{r['yolo_class']}({r['yolo_conf']:.2f})" if r["yolo_class"] else "-"
        cnn_txt = f"{r['cnn_class']}({r['cnn_conf']:.2f})" if r["cnn_class"] else "(시도 안 함)"
        self.detail_label.setText(f"YOLO: {yolo_txt}   |   CNN: {cnn_txt}")
        self._set_status(f"분류 완료: {final_class} [{r['method']}]")

        if self.serial_listener:
            self.serial_listener.send_class(final_class)

        points = 0
        if self.current_phone:
            points = award_points(self.conn, self.current_phone, final_class)
            if points > 0:
                self._set_status(f"{self.current_phone}님 +{points}점 적립")
            elif final_class != "unknown":
                print(f"[알림] '{final_class}'는 포인트 규칙에 없는 클래스명입니다. "
                      f"db/points.py의 POINTS_PER_CLASS와 철자를 맞춰보세요.")

        # cnn_class/cnn_confidence는 NOT NULL일 수 있는 옛 스키마 대응 - CNN을 안
        # 거쳤으면(YOLO로 바로 확정) YOLO 결과값으로 대신 채워넣음
        cnn_class_val = r["cnn_class"] if r["cnn_class"] is not None else (r["yolo_class"] or "unknown")
        cnn_conf_val = r["cnn_conf"] if r["cnn_conf"] is not None else r["yolo_conf"]

        self.conn.execute(
            """
            INSERT INTO sorting_log
                (timestamp, phone, cnn_class, cnn_confidence, yolo_class,
                 yolo_confidence, final_class, points_awarded)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                datetime.now().isoformat(),
                self.current_phone,
                cnn_class_val,
                cnn_conf_val,
                r["yolo_class"],
                r["yolo_conf"],
                final_class,
                points,
            ),
        )
        self.conn.commit()

        self._refresh_db_views()

    def _refresh_db_views(self):
        rows = fetch_recent_log(self.conn, RECENT_LOG_ROWS)
        self.log_table.setRowCount(len(rows))
        for r, (row_id, ts, phone, cnn_class, cnn_conf, yolo_class, yolo_conf,
                final_class, points) in enumerate(rows):
            yolo_disp = f"{yolo_class}({yolo_conf:.2f})" if yolo_class and yolo_conf is not None else "-"
            cnn_disp = f"{cnn_class}({cnn_conf:.2f})" if cnn_class and cnn_conf is not None else "-"
            values = [ts[:19], phone or "-", yolo_disp, cnn_disp, final_class, str(points), ""]
            for c, val in enumerate(values):
                self.log_table.setItem(r, c, QTableWidgetItem(val))

        board = get_leaderboard(self.conn, limit=10)
        self.rank_table.setRowCount(len(board))
        for r, (phone, points) in enumerate(board):
            self.rank_table.setItem(r, 0, QTableWidgetItem(phone))
            self.rank_table.setItem(r, 1, QTableWidgetItem(str(points)))

        self._refresh_my_points()
        self._update_stats_chart()

    def _refresh_my_points(self):
        if not self.current_phone:
            self.my_points_label.setText("게스트 - 전화번호를 입력하면 포인트가 적립/조회됩니다")
            self.my_breakdown_label.setText("")
            return

        total = get_points(self.conn, self.current_phone)
        counts = fetch_class_counts(self.conn, phone=self.current_phone)
        self.my_points_label.setText(f"{self.current_phone} 님 누적 포인트: {total}점")

        parts = []
        for cls in ("metal", "plastic", "paper", "unknown"):
            n = counts.get(cls, 0)
            if n:
                parts.append(f"{cls} {n}건")
        self.my_breakdown_label.setText("  /  ".join(parts) if parts else "아직 배출 기록이 없습니다")

    def _update_stats_chart(self):
        counts = fetch_class_counts(self.conn)
        classes = ["metal", "plastic", "paper", "unknown"]
        values = [counts.get(c, 0) for c in classes]
        colors = ["#9e9e9e", "#42a5f5", "#8bc34a", "#bdbdbd"]

        self.stats_ax.clear()
        self.stats_ax.bar(classes, values, color=colors)
        self.stats_ax.set_title("전체 배출 통계")
        self.stats_ax.set_ylabel("건수")
        for i, v in enumerate(values):
            self.stats_ax.text(i, v, str(v), ha="center", va="bottom")
        self.stats_figure.tight_layout()
        self.stats_canvas.draw()

    def closeEvent(self, event):
        self.preview_timer.stop()
        self.db_timer.stop()
        if self.detector is not None and hasattr(self, "detect_timer"):
            self.detect_timer.stop()
        if self.serial_listener:
            self.serial_listener.stop()
            self.serial_listener.wait(1000)
        if self.cap:
            self.cap.release()
        self.conn.close()
        event.accept()


def main():
    app = QApplication(sys.argv)
    window = Dashboard()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()

# =====================================================================
# 확인/맞춰야 할 것
#   - SERIAL_PORT: 핀헤더가 아니면 /dev/ttyACM0 등으로 변경
#   - CNN 모델(models/모델이름/classifier.tflite + class_names.txt)은 필수
#   - YOLO(models/detector/ 또는 models/아무폴더/의 best.pt나 detector.tflite)는
#     선택 사항 - 없으면 CNN만으로 동작
#   - DETECTOR_CONF_THRESHOLD: YOLO 결과를 바로 확정할지 결정하는 기준값
#   - CNN_CONF_THRESHOLD: CNN 최종 확정 기준값
#   - Jetson 본체 데스크톱(또는 VNC)에서 실행해야 창이 뜸 (SSH만으로는 안 됨)
# =====================================================================
