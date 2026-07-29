# =====================================================================
# 재활용 분류 시스템 - 통합 GUI 대시보드 (PyQt5)
# =====================================================================
# 카메라 미리보기 + 전화번호 입력(사용자용) + 분류 결과 + DB 최근 기록/포인트
# 랭킹을 한 창에서 볼 수 있는 대시보드입니다.
#
# 모델 여러 개 자동 재시도: models/ 안에 모델 폴더가 여러 개 있으면(예: model_freeze,
# model_finetune), 실행할 때 고른 모델로 먼저 분류하고 확신도가 낮아 unknown이
# 나오면 나머지 모델로 순서대로 재시도합니다. 모델을 전부 메모리에 올려두므로
# models/ 안에 모델이 너무 많으면 Jetson Nano 메모리(4GB)를 많이 씁니다 - 2~3개
# 정도가 적당합니다.
#
# 검출 박스(선택 사항): 1_train_yolo_colab.py 로 YOLO를 학습 + tflite로 변환한 뒤
# models/detector/detector.tflite 로 옮겨두면, 화면에 노란색 검출 박스가 자동으로
# 뜹니다. 최종 분류(metal/plastic/paper)는 여전히 CNN이 담당하고, YOLO는 오직
# "여기에 뭔가 있다"는 위치 표시용입니다. 파일이 없으면 박스 없이 CNN 분류만 동작.
#
# 트리거(촬영) 방식은 두 가지를 모두 지원합니다.
#   1) STM32가 SERIAL_PORT로 연결돼 있으면: "TRIGGER" 신호가 오면 자동 촬영
#   2) STM32가 없거나 연결에 실패해도: 화면의 "지금 촬영" 버튼으로 수동 촬영
#      (하드웨어 없이 노트북에서 바로 켜서 테스트할 때도 이 버튼을 씁니다)
#
# 설치 (Jetson Nano에서는 pip보다 apt 패키지가 훨씬 빠르게 설치됩니다):
#   sudo apt install python3-pyqt5 python3-matplotlib fonts-nanum
#   pip3 install pyserial opencv-python --break-system-packages   (이미 있으면 생략)
#   (fonts-nanum은 통계 그래프에 한글이 깨지지 않게 하려고 설치하는 한글 폰트입니다)
#
# 실행 (run/ 폴더 안에서):
#   python3 gui_dashboard.py                 실행 중 모델 번호로 선택
#   python3 gui_dashboard.py model_finetune   모델 폴더 바로 지정
#
# 폴더 구조 (기존과 동일)
#   project/
#   ├── models/model_xxx/   classifier.tflite, class_names.txt
#   ├── run/                이 스크립트 + 4-x 스크립트 + model_select.py
#   └── db/                 points.py, db_setup.py, db_view.py, *.db
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
FIXED_ROI = None                  # 예: (150, 80, 450, 380) - last_capture.jpg 보고 조정
CNN_CONF_THRESHOLD = 0.6
CAMERA_INDEX = 0
DB_PATH = os.path.join(DB_DIR, "sorting_log.db")
LAST_CAPTURE_PATH = os.path.join(BASE_DIR, "last_capture.jpg")
RECENT_LOG_ROWS = 15

# YOLO 검출 박스 (선택 사항 - 1_train_yolo_colab.py 로 학습 + tflite 변환한 파일을
# models/detector/detector.tflite 로 옮겨두면 자동으로 로딩해서 화면에 박스를 그려줍니다.
# 파일이 없으면 조용히 건너뛰고 기존 CNN 분류만 동작합니다.)
DETECTOR_PATH = os.path.join(MODELS_DIR, "detector", "detector.tflite")
DETECTOR_CONF_THRESHOLD = 0.4
DETECTOR_IOU_THRESHOLD = 0.45
DETECTOR_INTERVAL_MS = 250        # 매 프레임(30ms)마다 돌리면 Nano에 부담되므로 주기적으로만 실행


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


def detect_boxes(interpreter, frame, conf_threshold=DETECTOR_CONF_THRESHOLD, iou_threshold=DETECTOR_IOU_THRESHOLD):
    """YOLO(.tflite, ultralytics export) 결과를 디코딩해서 박스 목록을 반환.
    반환값: [(x1, y1, x2, y2, confidence), ...]  (프레임 원본 픽셀 좌표 기준)

    ultralytics export 버전에 따라 좌표가 0~1로 정규화돼 있는 경우와
    모델 입력 크기(imgsz) 기준 픽셀 값인 경우가 둘 다 있어서, 값 범위를 보고
    자동으로 판단합니다. 박스 위치가 이상하게 나오면 last_capture.jpg 와
    비교하면서 이 부분을 확인해보세요.
    """
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
        return []
    confidences = np.max(scores, axis=1)

    mask = confidences >= conf_threshold
    boxes = boxes[mask]
    confidences = confidences[mask]
    if len(boxes) == 0:
        return []

    frame_h, frame_w = frame.shape[:2]
    if boxes.max() <= 1.5:
        # 0~1로 정규화된 경우
        cx = boxes[:, 0] * frame_w
        cy = boxes[:, 1] * frame_h
        w = boxes[:, 2] * frame_w
        h = boxes[:, 3] * frame_h
    else:
        # 모델 입력 크기(in_w, in_h) 기준 픽셀 값인 경우
        cx = boxes[:, 0] * (frame_w / in_w)
        cy = boxes[:, 1] * (frame_h / in_h)
        w = boxes[:, 2] * (frame_w / in_w)
        h = boxes[:, 3] * (frame_h / in_h)

    x1 = cx - w / 2
    y1 = cy - h / 2

    nms_boxes = np.stack([x1, y1, w, h], axis=1).tolist()
    indices = cv2.dnn.NMSBoxes(nms_boxes, confidences.tolist(), conf_threshold, iou_threshold)

    results = []
    for idx in np.array(indices).flatten():
        bx, by, bw, bh = nms_boxes[int(idx)]
        results.append((int(bx), int(by), int(bx + bw), int(by + bh), float(confidences[int(idx)])))
    return results


def apply_roi(frame):
    if FIXED_ROI is None:
        return frame
    x1, y1, x2, y2 = FIXED_ROI
    return frame[y1:y2, x1:x2]


def fetch_recent_log(conn, limit=RECENT_LOG_ROWS):
    cursor = conn.execute(
        "SELECT id, timestamp, phone, cnn_class, cnn_confidence, final_class, points_awarded "
        "FROM sorting_log ORDER BY id DESC LIMIT ?",
        (limit,),
    )
    return cursor.fetchall()


def fetch_class_counts(conn, phone=None):
    """final_class별 건수. phone을 주면 그 사람 것만, 안 주면 전체."""
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
# STM32 시리얼 리스너 (연결되면 자동 트리거, 안 되면 조용히 비활성화)
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
        self.setWindowTitle("재활용 분류 시스템 - 대시보드")
        self.resize(1400, 800)

        self.conn = init_db()
        model_path, class_names_path = select_model(MODELS_DIR)

        # 처음 고른 모델을 1순위로 쓰고, 결과가 unknown(확신도 미달)이면
        # models/ 안의 다른 모델들을 차례로 시도합니다. (모델이 하나뿐이면
        # 기존과 동일하게 동작 - 재시도할 대상이 없음)
        print("CNN 모델 로딩...")
        self.models = [self._load_cnn_model("기본", model_path, class_names_path)]
        for name in list_available_models(MODELS_DIR):
            folder = os.path.join(MODELS_DIR, name)
            other_model_path = os.path.join(folder, "classifier.tflite")
            other_class_names_path = os.path.join(folder, "class_names.txt")
            if os.path.normpath(other_model_path) == os.path.normpath(model_path):
                continue  # 이미 위에서 로딩한 기본 모델
            print(f"대체 모델 로딩: {name}")
            self.models.append(self._load_cnn_model(name, other_model_path, other_class_names_path))

        # YOLO 검출기는 선택 사항 - 파일이 없으면 조용히 비활성화
        self.detector = None
        if os.path.exists(DETECTOR_PATH):
            print("YOLO 검출기 로딩...")
            try:
                self.detector = Interpreter(model_path=DETECTOR_PATH)
                self.detector.allocate_tensors()
            except Exception as e:
                print(f"YOLO 검출기 로딩 실패 (박스 없이 진행): {e}")
                self.detector = None
        else:
            print(f"YOLO 검출기 없음 ({DETECTOR_PATH}) - 검출 박스 없이 진행")

        self.current_phone = None
        self.last_frame = None
        self.detected_boxes = []
        self.serial_listener = None

        self.cap = cv2.VideoCapture(CAMERA_INDEX)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not self.cap.isOpened():
            QMessageBox.critical(self, "오류", f"카메라(index={CAMERA_INDEX})를 열 수 없습니다.")
        for _ in range(10):
            self.cap.read()

        self._build_ui()

        # 카메라 프리뷰 갱신
        self.preview_timer = QTimer(self)
        self.preview_timer.timeout.connect(self._update_preview)
        self.preview_timer.start(30)

        # YOLO 검출 박스 갱신 (매 프레임 돌리면 Nano에 부담되므로 별도 주기로)
        if self.detector is not None:
            self.detect_timer = QTimer(self)
            self.detect_timer.timeout.connect(self._run_detection)
            self.detect_timer.start(DETECTOR_INTERVAL_MS)

        # DB 결과(최근 기록/랭킹) 주기적 갱신
        self.db_timer = QTimer(self)
        self.db_timer.timeout.connect(self._refresh_db_views)
        self.db_timer.start(3000)
        self._refresh_db_views()

        # STM32 시리얼 리스너 (실패해도 앱은 계속 동작, 수동 촬영 버튼으로 대체)
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

        title = QLabel("재활용 분류 시스템 대시보드")
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

        # 사용자 조작 (전화번호 입력 + 촬영 버튼을 한 박스로 묶음)
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

        # 최근 분류 결과
        result_box = QGroupBox("최근 분류 결과")
        result_layout = QVBoxLayout()
        self.result_label = QLabel("-")
        self.result_label.setStyleSheet("font-size:20px; font-weight:bold;")
        self.result_label.setAlignment(Qt.AlignCenter)
        result_layout.addWidget(self.result_label)
        result_box.setLayout(result_layout)
        mid.addWidget(result_box)

        # 나의 포인트 적립 현황 (전화번호 입력한 사용자 기준)
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
        mid.addWidget(my_box, 1)  # 남는 세로 공간을 여기서 흡수 (하단에 몰리지 않게)

        splitter.addWidget(mid_widget)

        # ---------------- 오른쪽: DB 결과 대시보드 ----------------
        right_widget = QWidget()
        right = QVBoxLayout(right_widget)
        right.setSpacing(12)

        log_box = QGroupBox("최근 분류 기록 (DB)")
        log_layout = QVBoxLayout()
        self.log_table = QTableWidget(0, 6)
        self.log_table.setHorizontalHeaderLabels(
            ["시각", "전화번호", "CNN예측", "확신도", "최종", "포인트"]
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

        # 통계 그래프 (전체 배출 클래스별 건수)
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

    # ---------------- 모델 로딩 / 다중모델 재시도 ----------------
    def _load_cnn_model(self, name, model_path, class_names_path):
        interpreter = Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        return {
            "name": name,
            "interpreter": interpreter,
            "class_names": load_class_names(class_names_path),
        }

    def _classify_with_fallback(self, image):
        """1순위 모델부터 시도. 확신도가 기준 미만(unknown)이면 다음 모델로 재시도.
        전부 기준 미달이면 그중 확신도가 가장 높았던 결과를 반환합니다."""
        best = None  # (model_name, cnn_class, cnn_conf)
        for model in self.models:
            cnn_class, cnn_conf = classify(model["interpreter"], image, model["class_names"])
            print(f"  [{model['name']}] class={cnn_class} conf={cnn_conf:.2f}")
            if cnn_conf >= CNN_CONF_THRESHOLD:
                return model["name"], cnn_class, cnn_conf
            if best is None or cnn_conf > best[2]:
                best = (model["name"], cnn_class, cnn_conf)
        return best

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

        # YOLO 검출 박스 (있으면) - 노란색으로 표시해서 초록색 ROI와 구분
        for (bx1, by1, bx2, by2, conf) in self.detected_boxes:
            cv2.rectangle(display, (bx1, by1), (bx2, by2), (255, 220, 0), 2)
            cv2.putText(
                display, f"{conf:.2f}", (bx1, max(0, by1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 220, 0), 1,
            )

        h, w, ch = display.shape
        qimg = QImage(display.data, w, h, ch * w, QImage.Format_RGB888)
        pix = QPixmap.fromImage(qimg).scaled(
            self.preview_label.width(), self.preview_label.height(), Qt.KeepAspectRatio
        )
        self.preview_label.setPixmap(pix)

    def _run_detection(self):
        if self.detector is None or self.last_frame is None:
            return
        try:
            self.detected_boxes = detect_boxes(self.detector, self.last_frame)
        except Exception as e:
            print(f"검출 중 오류(무시하고 계속): {e}")

    def _on_trigger(self):
        if self.last_frame is None:
            self._set_status("아직 카메라 프레임이 없습니다.")
            return

        frame = self.last_frame
        cv2.imwrite(LAST_CAPTURE_PATH, frame)  # ROI 조정용 디버그 저장

        image = apply_roi(frame)
        used_model, cnn_class, cnn_conf = self._classify_with_fallback(image)
        final_class = cnn_class if cnn_conf >= CNN_CONF_THRESHOLD else "unknown"

        model_note = f" [{used_model}]" if len(self.models) > 1 else ""
        self.result_label.setText(f"{final_class}  (확신도 {cnn_conf:.2f}){model_note}")
        self._set_status(f"분류 완료: {final_class}{model_note}")

        if self.serial_listener:
            self.serial_listener.send_class(final_class)

        points = 0
        if self.current_phone:
            points = award_points(self.conn, self.current_phone, final_class)
            if points > 0:
                self._set_status(f"{self.current_phone}님 +{points}점 적립")

        self.conn.execute(
            """
            INSERT INTO sorting_log
                (timestamp, phone, cnn_class, cnn_confidence, final_class, points_awarded)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (datetime.now().isoformat(), self.current_phone, cnn_class, cnn_conf, final_class, points),
        )
        self.conn.commit()

        self._refresh_db_views()

    def _refresh_db_views(self):
        rows = fetch_recent_log(self.conn, RECENT_LOG_ROWS)
        self.log_table.setRowCount(len(rows))
        for r, (row_id, ts, phone, cnn_class, conf, final_class, points) in enumerate(rows):
            values = [ts[:19], phone or "-", cnn_class, f"{conf:.2f}", final_class, str(points)]
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
#   - SERIAL_PORT: 핀헤더가 아니면 /dev/ttyACM0 등으로 변경 (STM32 없이 테스트만
#     할 거면 그대로 둬도 됨 - 연결 실패 메시지만 뜨고 "지금 촬영" 버튼은 정상 동작)
#   - FIXED_ROI: last_capture.jpg 보면서 좌표 조정
#   - Jetson 본체 데스크톱(또는 VNC)에서 실행해야 창이 뜸 (SSH만으로는 안 됨)
#   - PyQt5는 Jetson에서 pip로 설치하면 매우 오래 걸리므로
#     `sudo apt install python3-pyqt5` 로 설치하는 걸 추천
# =====================================================================
