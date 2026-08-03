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
import sqlite3               # DB 파일(sorting_log.db) 읽고 쓰기용 (파이썬 기본 내장 모듈)
from datetime import datetime  # 분류 기록에 남길 시각(timestamp) 생성용

import numpy as np
import cv2                   # 카메라 캡처, 이미지 리사이즈/크롭/그리기(OpenCV)

# PyQt5: 이 대시보드의 창/버튼/표 등 화면 전체를 그리는 GUI 프레임워크
from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QLineEdit, QPushButton,
    QVBoxLayout, QHBoxLayout, QTableWidget, QTableWidgetItem, QGroupBox,
    QHeaderView, QMessageBox, QSplitter, QSizePolicy,
)

# matplotlib: 오른쪽의 "통계 그래프"(막대그래프)를 그리는 데 사용.
# Qt5Agg 백엔드를 지정해야 PyQt5 창 안에 그래프를 직접 임베드할 수 있음
# (지정 안 하면 matplotlib이 자기만의 별도 창을 띄우려고 해서 PyQt5와 충돌함).
import matplotlib
matplotlib.use("Qt5Agg")
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure

# 한글 폰트 설정 (Jetson에 나눔고딕 등이 없으면 그래프 글자가 네모(□)로 보일 수 있음.
# 그럴 땐 `sudo apt install fonts-nanum` 설치 후 아래 폰트명이 맞는지 확인하세요.
# 폰트를 새로 설치했는데도 안 바뀌면 `rm -rf ~/.cache/matplotlib` 로 캐시를 지워야
# matplotlib이 새 폰트를 인식합니다.)
matplotlib.rcParams["font.family"] = "NanumGothic"
matplotlib.rcParams["axes.unicode_minus"] = False

# pyserial은 STM32와 UART로 통신할 때만 필요합니다. 설치가 안 돼 있어도 프로그램이
# 죽지 않고, 아래 SerialListener가 "자동 트리거 비활성화" 상태로 대신 동작해서
# 수동 촬영 버튼만으로 계속 쓸 수 있게 해뒀습니다.
try:
    import serial
except ImportError:
    serial = None

# __file__ 기준으로 경로를 계산해두면, 이 스크립트를 어느 위치에서 실행하든
# (터미널 cwd가 어디든) models/, db/ 폴더를 항상 올바르게 찾을 수 있습니다.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))          # run/ 폴더
MODELS_DIR = os.path.normpath(os.path.join(BASE_DIR, "..", "models"))
DB_DIR = os.path.normpath(os.path.join(BASE_DIR, "..", "db"))

# db/points.py, run/model_select.py 는 별도 폴더에 있는 모듈이라, 파이썬이 import할
# 경로 목록(sys.path)에 db/ 폴더를 직접 추가해줘야 아래 import가 성공합니다.
sys.path.append(DB_DIR)
from points import init_users_table, award_points, is_valid_phone, get_leaderboard, get_points
from model_select import select_model, list_available_models

# tflite_runtime이 설치돼 있으면 그 가벼운 런타임을 우선 사용하고(Jetson 권장),
# 없으면 무거운 tensorflow 안에 들어있는 동일한 Interpreter로 대체합니다.
# (Jetson에 ultralytics를 설치하면서 numpy 버전이 올라가 예전 TensorFlow가
# 깨졌던 적이 있어서, 가능하면 tflite_runtime 경로를 쓰는 걸 권장합니다.)
try:
    from tflite_runtime.interpreter import Interpreter
except ImportError:
    from tensorflow.lite.python.interpreter import Interpreter


# ---------------------------------------------------------------------
# 설정값
# ---------------------------------------------------------------------
SERIAL_PORT = "/dev/ttyTHS1"     # STM32 핀헤더 UART. USB 연결이면 /dev/ttyACM0 등으로 변경
SERIAL_BAUDRATE = 115200        # STM32 쪽 코드와 반드시 같은 값이어야 통신됨
FIXED_ROI = None                  # YOLO가 없거나 못 찾았을 때 CNN에 쓸 좌표. 예: (150, 80, 450, 380)
                                   # (x1, y1, x2, y2) 픽셀 좌표. None이면 전체 프레임을 그대로 씀
CNN_CONF_THRESHOLD = 0.6         # CNN 결과를 최종으로 인정할 확신도 기준 (이 미만이면 다음 모델로 재시도)
CAMERA_INDEX = 0                 # 카메라가 여러 대면 0, 1, 2... 로 바꿔서 맞는 걸 찾아야 함
DB_PATH = os.path.join(DB_DIR, "sorting_log.db")        # 실제 운영 기록이 쌓이는 DB 파일
LAST_CAPTURE_PATH = os.path.join(BASE_DIR, "last_capture.jpg")  # 매번 촬영 시 덮어써서 저장
                                                                  # (FIXED_ROI 좌표를 눈으로 맞춰볼 때 유용)
RECENT_LOG_ROWS = 15             # 화면 "최근 분류 기록" 표에 보여줄 최대 행 수

DETECTOR_DIR = os.path.join(MODELS_DIR, "detector")
DETECTOR_CONF_THRESHOLD = 0.5    # 이 이상이면 YOLO 결과를 바로 최종으로 사용 (CNN 단계 생략)
DETECTOR_INTERVAL_MS = 250       # 미리보기 화면에 박스를 다시 그리는 주기(ms). 카메라 프레임(30ms)
                                  # 마다 매번 YOLO를 돌리면 Jetson에 부담이 커서 더 느슨한 주기로 실행


def _ensure_column(conn, table, column, coltype):
    """SQLite는 여러 4-x 스크립트가 서로 다른 시점에 만들어진 컬럼 조합으로
    같은 DB 파일(sorting_log.db)을 공유합니다. CREATE TABLE IF NOT EXISTS는
    테이블이 이미 있으면 아무것도 안 바꾸므로, 없는 컬럼이 있으면 이 함수로
    ALTER TABLE ADD COLUMN 해서 스키마를 최신 상태로 맞춰줍니다(마이그레이션)."""
    cols = [row[1] for row in conn.execute(f"PRAGMA table_info({table})")]
    if column not in cols:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {coltype}")


def init_db():
    """DB 연결 + 테이블 준비. sorting_log(분류 기록)와 users(포인트, points.py가 관리)
    테이블이 없으면 새로 만들고, 있으면 그대로 두되 컬럼만 최신화(_ensure_column)합니다."""
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    # check_same_thread=False : SerialListener가 별도 스레드에서 이 연결을 쓰기 때문에
    # (triggered 신호가 메인 스레드로 넘어와 처리되긴 하지만, sqlite3 기본 설정은
    # "연결을 만든 스레드에서만 쓰라"고 강제하므로) 이 제한을 풀어줘야 에러가 안 남
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
    # 예전 스키마(db_setup.py 최초 버전)에는 yolo_class/yolo_confidence 컬럼이
    # 아예 없었을 수 있어서, 없으면 추가해줍니다. 참고로 cnn_class/cnn_confidence는
    # 원래 스키마에서 NOT NULL로 정의돼 있어, YOLO만으로 확정된 경우에도 아래
    # _on_trigger()에서 YOLO 값을 대신 채워 넣어 이 제약을 만족시킵니다.
    _ensure_column(conn, "sorting_log", "yolo_class", "TEXT")
    _ensure_column(conn, "sorting_log", "yolo_confidence", "REAL")
    _ensure_column(conn, "sorting_log", "points_awarded", "INTEGER NOT NULL DEFAULT 0")
    # method: "YOLO"/"CNN(모델명)"/"YOLO+CNN 둘 다 확신도 부족" 중 어느 경로로
    # 최종 판정됐는지. infer_ms: 그 판정 1회에 걸린 실제 시간(ms).
    # 둘 다 하이브리드 설계 효과(평시 YOLO 1회로 얼마나 빨리 끝나는지)를
    # 결과 슬라이드용 통계로 뽑기 위해 추가함 - db/db_view.py 참고.
    _ensure_column(conn, "sorting_log", "method", "TEXT")
    _ensure_column(conn, "sorting_log", "infer_ms", "REAL")
    init_users_table(conn)  # points.py 쪽 users(전화번호, 누적포인트) 테이블 준비
    conn.commit()
    return conn


def load_class_names(class_names_path):
    """3_train_cnn_classifier.py가 저장해둔 class_names.txt를 읽어서
    ['metal', 'paper', 'plastic'] 같은 리스트로 반환. 이 순서가 CNN 모델의
    출력 인덱스(0,1,2...)와 그대로 대응되므로 순서를 바꾸면 안 됨."""
    with open(class_names_path, encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def classify(interpreter, image, class_names):
    """CNN(.tflite) 한 번 실행해서 (클래스이름, 확신도) 하나를 돌려주는 함수.
    image는 이미 크롭(또는 ROI 적용)되어 있는 상태여야 함 - 3_train_cnn_
    classifier.py가 크롭된 사진으로만 학습했기 때문."""
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    height, width = input_details[0]["shape"][1:3]  # 모델이 기대하는 입력 크기(예: 224x224)

    img = cv2.resize(image, (width, height))                       # 모델 입력 크기에 맞춤
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)  # OpenCV는 BGR 순서라 RGB로 변환
    img = np.expand_dims(img, axis=0)  # 모델은 (배치, 높이, 너비, 채널) 4차원을 기대하므로
                                        # 앞에 배치 차원 1개를 추가 (사진 한 장 = 배치 크기 1)

    # 모델(.tflite) 안에 Rescaling 레이어가 이미 포함되어 있으므로
    # 여기서 픽셀값을 추가로 정규화하면 안 됨 (이중정규화 버그 주의)
    interpreter.set_tensor(input_details[0]["index"], img)  # 입력 이미지를 모델에 넣고
    interpreter.invoke()                                     # 실제 추론 실행
    output = interpreter.get_tensor(output_details[0]["index"])[0]
    # output 예시: [0.94, 0.03, 0.03] 처럼 클래스별 확률이 담긴 배열 (합계 1에 가까움)

    idx = int(np.argmax(output))          # 확률이 가장 높은 인덱스를 찾고
    return class_names[idx], float(output[idx])  # 그 인덱스를 이름으로 바꿔서 (이름, 확률) 반환


def apply_roi(frame):
    """YOLO가 아예 없거나(모델 미탑재) 크롭에 실패했을 때 대신 쓸 고정 관심영역.
    FIXED_ROI가 None이면 원본 프레임을 그대로 반환(=크롭 안 함)."""
    if FIXED_ROI is None:
        return frame
    x1, y1, x2, y2 = FIXED_ROI
    return frame[y1:y2, x1:x2]


def fetch_recent_log(conn, limit=RECENT_LOG_ROWS):
    cursor = conn.execute(
        "SELECT id, timestamp, phone, cnn_class, cnn_confidence, yolo_class, "
        "yolo_confidence, final_class, points_awarded, method "
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
    """YOLO를 detector.tflite(가벼운 tflite_runtime)로 돌리는 경우 사용.
    best.pt를 export(format="tflite")로 변환한 뒤 나온 파일이 이 클래스의 입력."""

    def __init__(self, path, labels_path):
        self.interpreter = Interpreter(model_path=path)
        self.interpreter.allocate_tensors()
        self.labels = load_labels(labels_path)
        # tflite 파일 자체에는 클래스 "이름"이 저장되지 않으므로(숫자 인덱스만
        # 출력), 1_train_yolo_colab.py가 같이 만들어준 detector_labels.txt가
        # 없으면 이름 대신 "class_0", "class_1"처럼 번호로만 표시됨
        if self.labels is None:
            print(f"[경고] {labels_path} 없음 - 클래스 이름 대신 class_0, class_1...로 표시됩니다")

    def _class_name(self, class_id):
        if self.labels and 0 <= class_id < len(self.labels):
            return self.labels[class_id]
        return f"class_{class_id}"

    def detect_and_crop(self, frame, conf_threshold=DETECTOR_CONF_THRESHOLD):
        """카메라 프레임 한 장을 받아 (크롭이미지, 확신도, 박스좌표, 클래스이름)을
        반환. 확신도가 낮거나 아예 물체를 못 찾으면 크롭/박스는 None으로 옴."""
        input_details = self.interpreter.get_input_details()
        output_details = self.interpreter.get_output_details()
        in_h, in_w = input_details[0]["shape"][1:3]  # 모델이 기대하는 입력 크기(예: 640x640)

        # YOLO 입력 전처리: CNN과 달리 0~1 범위로 직접 나눠줌(/255.0). YOLO tflite
        # export본은 정규화 레이어가 따로 안 들어있는 경우가 많아 여기서 처리
        img = cv2.resize(frame, (in_w, in_h))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        img = np.expand_dims(img, axis=0)

        self.interpreter.set_tensor(input_details[0]["index"], img)
        self.interpreter.invoke()
        output = self.interpreter.get_tensor(output_details[0]["index"])[0]

        # YOLO tflite의 원본 출력 형태는 export 방식에 따라 (4+클래스수, 박스후보수)
        # 또는 (박스후보수, 4+클래스수) 둘 중 하나로 나옵니다. 박스 후보 개수(보통
        # 수천 개)가 (4+클래스수)보다 훨씬 크다는 점을 이용해, 더 작은 쪽을 앞으로
        # 오도록 자동으로 뒤집어(transpose) 형태를 통일시킵니다.
        if output.shape[0] < output.shape[1]:
            output = output.T

        # 통일된 뒤에는 각 행이 박스 후보 하나: 앞 4개는 좌표(cx,cy,w,h), 나머지가
        # 클래스별 확률(score)
        boxes = output[:, :4]
        scores = output[:, 4:]
        if scores.size == 0:
            return None, 0.0, None, None

        class_ids = np.argmax(scores, axis=1)   # 후보마다 가장 확률 높은 클래스
        confidences = np.max(scores, axis=1)    # 그 클래스의 확률값

        # 수천 개의 박스 후보 중 확신도가 가장 높은 딱 하나만 채택
        # (여러 물체를 동시에 처리하는 기능은 없고, "가장 확실한 물체 하나"만 봄 -
        # 컨베이어 벨트에 한 번에 하나씩 올라온다는 전제와 맞아떨어짐)
        best_idx = int(np.argmax(confidences))
        best_conf = float(confidences[best_idx])
        best_class_id = int(class_ids[best_idx])
        if best_conf < conf_threshold:
            # 확신도 미달 - 박스는 안 주지만, 클래스 이름/확신도는 참고용으로 반환
            # (호출부에서 "YOLO가 뭐라고 말은 했었는지" 화면에 표시할 때 씀)
            return None, best_conf, None, self._class_name(best_class_id)

        cx, cy, w, h = boxes[best_idx]
        frame_h, frame_w = frame.shape[:2]

        # 좌표가 0~1로 정규화되어 있는지, 아니면 모델 입력 픽셀 크기(예: 0~640) 기준인지
        # export 버전마다 다르게 나올 수 있어서, 값의 최대치로 어느 쪽인지 추정합니다.
        # (정규화됐다면 cx,cy,w,h가 전부 1.0을 크게 못 넘음 -> 1.5를 안전 기준으로 사용)
        if max(cx, cy, w, h) <= 1.5:
            # 0~1 정규화 좌표 -> 원본 프레임 픽셀 좌표로 변환
            cx, cy, w, h = cx * frame_w, cy * frame_h, w * frame_w, h * frame_h
        else:
            # 모델 입력 크기(in_w x in_h) 기준 픽셀 좌표 -> 원본 프레임 크기로 비율 변환
            cx = cx * (frame_w / in_w)
            cy = cy * (frame_h / in_h)
            w = w * (frame_w / in_w)
            h = h * (frame_h / in_h)

        # 중심좌표+폭/높이 형식을 좌상단/우하단(x1,y1,x2,y2) 형식으로 변환하고,
        # max/min으로 이미지 경계 밖으로 안 나가게 고정
        x1 = max(0, int(cx - w / 2))
        y1 = max(0, int(cy - h / 2))
        x2 = min(frame_w, int(cx + w / 2))
        y2 = min(frame_h, int(cy + h / 2))

        crop = frame[y1:y2, x1:x2]  # 이 박스 영역만 실제로 잘라냄 - 이게 CNN에 넘겨질 이미지
        class_name = self._class_name(best_class_id)
        if crop.size == 0:
            return None, best_conf, None, class_name
        return crop, best_conf, (x1, y1, x2, y2), class_name


class UltralyticsDetector:
    """YOLO를 best.pt(ultralytics 라이브러리)로 직접 돌리는 경우 사용.
    tflite 변환 없이 학습 직후 파일을 바로 쓸 수 있지만, ultralytics/torch가
    설치돼 있어야 하고 TFLiteDetector보다 무겁습니다."""

    def __init__(self, path):
        from ultralytics import YOLO  # 여기서 import: ultralytics가 없는 환경에서도
                                       # 이 클래스를 아예 안 쓰면 프로그램이 안 죽게 하기 위함
        self.model = YOLO(path)

    def detect_and_crop(self, frame, conf_threshold=DETECTOR_CONF_THRESHOLD):
        # predict()가 전처리(리사이즈/정규화)부터 후처리(여러 박스 중 최적 선별의
        # 앞단계인 NMS 등)까지 라이브러리 내부에서 다 알아서 처리해줌 - TFLiteDetector
        # 보다 코드가 훨씬 짧은 이유
        results = self.model.predict(frame, conf=conf_threshold, verbose=False)
        boxes = results[0].boxes  # 이 프레임에서 conf_threshold를 넘긴 박스들
        if boxes is None or len(boxes) == 0:
            return None, 0.0, None, None

        # 여러 박스가 검출됐을 수 있으므로, 그중 확신도가 가장 높은 것 하나만 채택
        best_idx = int(boxes.conf.argmax().item())
        conf = float(boxes.conf[best_idx])
        class_id = int(boxes.cls[best_idx].item())
        class_name = self.model.names.get(class_id, f"class_{class_id}")
        x1, y1, x2, y2 = boxes.xyxy[best_idx].tolist()  # 이미 원본 프레임 픽셀 좌표로 변환되어 있음

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
    """YOLO 모델을 찾아볼 폴더 후보 목록. models/detector/ 를 우선 확인하고,
    그 다음 models/ 바로 아래의 모든 하위 폴더(데이터셋별 폴더: 3_class,
    trash_line_3class 등)도 전부 뒤져봄."""
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
    """STM32와의 UART 통신을 전담하는 별도 스레드.

    왜 스레드로 분리했나: ser.readline()은 데이터가 올 때까지 멈춰서 기다리는
    "블로킹" 호출입니다. 이걸 메인 스레드에서 그대로 부르면 카메라 미리보기나
    버튼 클릭 같은 화면 전체가 같이 멈춰버립니다. 그래서 시리얼 읽기만 별도
    스레드(QThread)에서 무한 루프로 돌리고, "TRIGGER" 신호가 오면 PyQt의
    신호(pyqtSignal)를 emit해서 메인 스레드 쪽 함수(_on_trigger)를 안전하게
    호출하도록 넘겨줍니다."""

    triggered = pyqtSignal()   # STM32가 "TRIGGER\n"을 보내오면 발생
    status = pyqtSignal(str)   # 연결 성공/실패 등 상태 메시지를 화면 하단에 표시하기 위함

    def __init__(self, port, baudrate):
        super().__init__()
        self.port = port
        self.baudrate = baudrate
        self._running = True
        self.ser = None

    def run(self):
        """QThread.start()를 부르면 이 함수가 별도 스레드에서 자동 실행됨."""
        if serial is None:
            self.status.emit("pyserial 미설치 - 자동 트리거 비활성화 (지금 촬영 버튼 사용)")
            return
        try:
            self.ser = serial.Serial(self.port, self.baudrate, timeout=1)
            self.ser.reset_input_buffer()  # 이전에 밀려있던 낡은 데이터 비우기
            time.sleep(2)  # 일부 보드는 시리얼 포트를 열자마자 재부팅되므로 안정화 대기
            self.status.emit(f"STM32 연결됨 ({self.port})")
        except serial.SerialException as e:
            self.status.emit(f"STM32 연결 실패({e}) - 지금 촬영 버튼으로 진행하세요")
            return

        # 연결이 끊기거나 stop()이 불릴 때까지 계속 한 줄씩 읽어옴
        while self._running:
            try:
                line = self.ser.readline().decode("utf-8", errors="ignore").strip()
            except serial.SerialException:
                self.status.emit("STM32 연결이 끊겼습니다.")
                return
            if line == "TRIGGER":
                self.triggered.emit()  # 메인 스레드의 _on_trigger()가 실행되도록 신호만 보냄

    def send_class(self, final_class):
        """판정 결과를 STM32에 "CLASS:metal\\n" 형식의 텍스트로 전송.
        STM32는 이걸 받아서 해당 서보를 작동시키고 벨트를 재가동함."""
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
        # select_model()이 "기본" 모델 하나를 먼저 고르게 함(sys.argv[1]로 바로 지정하거나,
        # 후보가 하나면 자동, 여러 개면 번호 입력 프롬프트).
        model_path, class_names_path = select_model(MODELS_DIR)
        print("CNN 모델 로딩...")
        self.cnn_models = [self._load_cnn_model("기본", model_path, class_names_path)]
        # 그리고 models/ 아래 나머지 폴더들(방금 고른 것 제외)을 전부 "대체 모델"로
        # 추가 로딩. 이후 _classify_with_cnn_fallback()이 이 리스트를 순서대로 재시도함.
        for name in list_available_models(MODELS_DIR):
            folder = os.path.join(MODELS_DIR, name)
            other_model_path = os.path.join(folder, "classifier.tflite")
            other_class_names_path = os.path.join(folder, "class_names.txt")
            if os.path.normpath(other_model_path) == os.path.normpath(model_path):
                continue  # 이미 "기본"으로 로딩한 모델과 같은 파일이면 중복 로딩 방지
            print(f"대체 CNN 모델 로딩: {name}")
            self.cnn_models.append(self._load_cnn_model(name, other_model_path, other_class_names_path))

        # --- YOLO 검출기 (선택 사항) ---
        # CNN과 달리 YOLO는 폴백(재시도) 없이 딱 하나만 골라서 self.detector에 저장.
        # 후보가 없으면 load_detector()가 None을 반환하고, 이후 코드는 self.detector
        # is None 체크로 YOLO 없이도 CNN만으로 정상 동작하도록 방어되어 있음.
        print("YOLO 모델 로딩 시도...")
        self.detector = load_detector()

        self.current_phone = None   # None이면 게스트, 문자열이면 그 전화번호로 포인트 적립
        self.last_frame = None      # 카메라의 가장 최근 프레임 (트리거 시 이걸 캡처해서 씀)
        self.detected_box = None  # (x1, y1, x2, y2, class_name, conf) - 미리보기용
        self.serial_listener = None

        # 카메라 오픈. BUFFERSIZE=1로 낮춰서 오래된 프레임이 버퍼에 쌓이는 걸 방지
        # (안 낮추면 cap.read()가 몇 프레임 전의 낡은 화면을 줄 수 있음)
        self.cap = cv2.VideoCapture(CAMERA_INDEX)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not self.cap.isOpened():
            QMessageBox.critical(self, "오류", f"카메라(index={CAMERA_INDEX})를 열 수 없습니다.")
        for _ in range(10):
            self.cap.read()  # 노출/화이트밸런스가 안정되도록 워밍업으로 몇 프레임 미리 읽음

        self._build_ui()

        # 타이머 3개가 서로 다른 주기로 각자 할 일을 함 (하나의 무한루프 대신 이벤트
        # 기반으로 동작 - PyQt GUI는 반드시 이런 방식이어야 화면이 멈추지 않음)
        self.preview_timer = QTimer(self)
        self.preview_timer.timeout.connect(self._update_preview)
        self.preview_timer.start(30)  # 약 33fps로 카메라 미리보기 갱신

        if self.detector is not None:
            self.detect_timer = QTimer(self)
            self.detect_timer.timeout.connect(self._run_detection)
            self.detect_timer.start(DETECTOR_INTERVAL_MS)  # 250ms마다 미리보기용 박스만 갱신
                                                              # (최종 판정은 트리거 시 별도로 다시 계산)

        self.db_timer = QTimer(self)
        self.db_timer.timeout.connect(self._refresh_db_views)
        self.db_timer.start(3000)  # 3초마다 표/랭킹/그래프를 새로고침 (다른 사람이 방금
                                     # 적립한 포인트도 화면에 곧 반영되게 하기 위함)
        self._refresh_db_views()

        # STM32 시리얼 통신은 GUI가 다 뜬 뒤 마지막에 시작 (연결 실패해도 화면은 이미
        # 정상적으로 뜬 상태라 "지금 촬영" 버튼으로 계속 쓸 수 있음)
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
        """CNN(.tflite) 하나를 메모리에 올리고, 이름표(name)/인터프리터/클래스이름
        목록을 딕셔너리로 묶어서 self.cnn_models 리스트에 넣기 좋게 반환."""
        interpreter = Interpreter(model_path=model_path)
        interpreter.allocate_tensors()  # 모델 실행 전에 반드시 한 번 호출해야 하는 초기화 단계
        return {
            "name": name,
            "interpreter": interpreter,
            "class_names": load_class_names(class_names_path),
        }

    def _classify_with_cnn_fallback(self, image):
        """CNN 모델을 1순위("기본")부터 순서대로 시도. 확신도(CNN_CONF_THRESHOLD)
        기준을 넘기는 모델이 나오는 즉시 그 결과로 확정하고 멈춤. 끝까지 아무도
        기준을 못 넘기면, 그중에서 그나마 확신도가 가장 높았던 결과를 반환
        (완전히 포기하지 않고 "그나마 나은 추측"을 최종 후보로 남겨둠 - 다만
        호출부인 _classify_hybrid에서 이 값도 CNN_CONF_THRESHOLD 미달이면
        결국 unknown 처리됨).
        반환: (model_name, cnn_class, cnn_conf)"""
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
        """이 대시보드의 핵심 판정 로직. YOLO 먼저 시도 -> 확신도 충분하면 그걸로
        바로 확정(CNN은 아예 안 돌림). 확신도가 부족하면 YOLO가 크롭해준 이미지를
        CNN(다중모델 포함)에 넘겨 재시도. 반환값은 dict 하나로, 중간 과정(YOLO/CNN
        각각 뭐라고 판단했는지)까지 전부 담아서 화면 표시와 DB 기록에 그대로
        재사용합니다."""
        result = {
            "yolo_class": None, "yolo_conf": 0.0, "box": None,
            "cnn_model": None, "cnn_class": None, "cnn_conf": None,
            "final_class": "unknown", "method": "",
        }

        crop = None
        if self.detector is not None:
            # YOLO 한 번 실행: 위치(box) + 자체 분류(yolo_class) + 확신도(yolo_conf)를
            # 동시에 얻음 (1_train_yolo_colab.py의 "2-1) YOLO는 사실 분류도 한다" 참고)
            crop, yolo_conf, box, yolo_class = self.detector.detect_and_crop(frame)
            result["yolo_class"] = yolo_class
            result["yolo_conf"] = yolo_conf
            result["box"] = box
            print(f"  [YOLO] class={yolo_class} conf={yolo_conf:.2f}")

            if yolo_class is not None and box is not None and yolo_conf >= DETECTOR_CONF_THRESHOLD:
                # 확신도 충분 -> YOLO의 판단을 그대로 최종 결과로 채택하고 즉시 반환.
                # 이 경로에서는 CNN을 아예 실행하지 않으므로 그만큼 더 빠름.
                result["final_class"] = yolo_class.strip().lower()
                result["method"] = "YOLO"
                return result

        # 여기 도달했다는 건 YOLO가 없거나(self.detector is None) 확신도가 부족했다는 뜻
        # -> CNN으로 재시도. 이때 CNN에 넣을 이미지는 반드시 "크롭된" 것이어야 함
        # (CNN은 3_train_cnn_classifier.py에서 크롭된 사진만 보고 학습했으므로).
        # YOLO가 위치라도 알려줬으면(crop이 있으면) 그 크롭을 쓰고, YOLO 자체가
        # 없거나 크롭조차 실패했으면 차선책으로 고정 ROI(또는 원본 전체)를 사용.
        image = crop if crop is not None else apply_roi(frame)
        cnn_model, cnn_class, cnn_conf = self._classify_with_cnn_fallback(image)
        result["cnn_model"] = cnn_model
        result["cnn_class"] = cnn_class
        result["cnn_conf"] = cnn_conf

        if cnn_conf >= CNN_CONF_THRESHOLD:
            result["final_class"] = cnn_class
            result["method"] = f"CNN({cnn_model})"  # 어떤 CNN 모델이 확정했는지 이름을 남김
                                                      # (화면/로그에 "3_class"처럼 뜨는 이유)
        else:
            # YOLO도, CNN의 모든 후보 모델도 전부 확신도 미달 -> 최종적으로 포기
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
        """미리보기용 - YOLO 박스만 계속 갱신 (최종 판정은 트리거 시 별도로 다시 함).
        DETECTOR_INTERVAL_MS(250ms)마다 호출되며, 여기서 나온 결과는 화면에 노란
        박스를 그리는 용도로만 쓰이고 DB/포인트에는 영향을 주지 않음. 실제 판정은
        트리거가 눌렸을 때 _classify_hybrid()가 그 시점 프레임으로 다시 계산함."""
        if self.detector is None or self.last_frame is None:
            return
        try:
            _, conf, box, class_name = self.detector.detect_and_crop(self.last_frame)
            self.detected_box = (box[0], box[1], box[2], box[3], class_name, conf) if box else None
        except Exception as e:
            # 미리보기용 갱신이 한 번 실패한다고 프로그램 전체가 죽으면 안 되므로
            # 예외를 잡아서 로그만 남기고 다음 주기에 다시 시도
            print(f"검출 중 오류(무시하고 계속): {e}")

    def _on_trigger(self):
        """자동(STM32 "TRIGGER") 또는 수동("지금 촬영" 버튼) 트리거가 오면 실행되는
        메인 파이프라인: 촬영 -> 하이브리드 판정 -> 화면 표시 -> STM32 응답 ->
        포인트 적립 -> DB 기록, 순서대로 전부 여기서 처리합니다."""
        if self.last_frame is None:
            self._set_status("아직 카메라 프레임이 없습니다.")
            return

        # ① 촬영: 미리보기 루프가 계속 채워주던 최신 프레임을 그대로 사용
        # (매 프레임 read()를 이미 하고 있으므로 별도로 "지금 다시 찍기"를 안 해도 최신 상태)
        frame = self.last_frame
        cv2.imwrite(LAST_CAPTURE_PATH, frame)  # ROI 조정용 디버그 저장

        # ② 판정: YOLO -> (필요시) CNN 순서로 최종 클래스를 결정
        # 추론 1회에 걸리는 실제 시간(ms)을 측정 — 발표자료 "결과" 슬라이드의
        # "추론 시간/장" 항목은 아직 실측 전 플레이스홀더였는데, Jetson Nano에서
        # 이 코드를 실제로 돌리면 여기서 찍히는 값이 그 실측치가 됩니다.
        t0 = time.perf_counter()
        r = self._classify_hybrid(frame)
        infer_ms = (time.perf_counter() - t0) * 1000
        final_class = r["final_class"]

        # 콘솔에 매 회 기록 -> 데모/테스트 중 여러 번 트리거해서 평균을 내면
        # "추론 시간/장" 슬라이드 수치로 바로 쓸 수 있습니다.
        print(f"[추론시간] {infer_ms:.1f} ms (method={r['method']}, final={final_class})")

        # ③ 화면 표시: 최종 결과 + YOLO/CNN 각각의 판단을 나란히 보여줌
        self.result_label.setText(f"{final_class}  ({r['method']})")
        yolo_txt = f"{r['yolo_class']}({r['yolo_conf']:.2f})" if r["yolo_class"] else "-"
        cnn_txt = f"{r['cnn_class']}({r['cnn_conf']:.2f})" if r["cnn_class"] else "(시도 안 함)"
        # 화면에도 추론시간을 같이 표시 -> 발표 데모 중 심사위원이 직접 눈으로 확인 가능
        self.detail_label.setText(f"YOLO: {yolo_txt}   |   CNN: {cnn_txt}   |   {infer_ms:.0f} ms")
        self._set_status(f"분류 완료: {final_class} [{r['method']}]")

        # ④ STM32 응답: "CLASS:metal\n" 같은 텍스트를 보내서 서보 작동 + 벨트 재가동을 트리거
        if self.serial_listener:
            self.serial_listener.send_class(final_class)

        # ⑤ 포인트 적립: 게스트(전화번호 미입력)면 적립 없이 0점 처리
        points = 0
        if self.current_phone:
            points = award_points(self.conn, self.current_phone, final_class)
            if points > 0:
                self._set_status(f"{self.current_phone}님 +{points}점 적립")
            elif final_class != "unknown":
                print(f"[알림] '{final_class}'는 포인트 규칙에 없는 클래스명입니다. "
                      f"db/points.py의 POINTS_PER_CLASS와 철자를 맞춰보세요.")

        # ⑥ DB 기록 준비: cnn_class/cnn_confidence는 원래 스키마상 NOT NULL이라,
        # YOLO로 바로 확정되어 CNN을 아예 안 거친 경우엔 그 자리에 YOLO 결과값을
        # 대신 채워 넣어서 제약을 만족시킴 (컬럼 자체를 NULL 허용으로 바꾸려면
        # SQLite 특성상 테이블을 통째로 재생성해야 해서, 값을 채우는 쪽으로 우회)
        cnn_class_val = r["cnn_class"] if r["cnn_class"] is not None else (r["yolo_class"] or "unknown")
        cnn_conf_val = r["cnn_conf"] if r["cnn_conf"] is not None else r["yolo_conf"]

        self.conn.execute(
            """
            INSERT INTO sorting_log
                (timestamp, phone, cnn_class, cnn_confidence, yolo_class,
                 yolo_confidence, final_class, points_awarded, method, infer_ms)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                r["method"],
                infer_ms,
            ),
        )
        self.conn.commit()

        # ⑦ 오른쪽 DB 패널(최근 기록/랭킹/통계 그래프)을 즉시 갱신해서 방금 적립된
        # 내역이 3초 타이머를 기다리지 않고 바로 보이게 함
        self._refresh_db_views()

    def _refresh_db_views(self):
        """DB에서 최근 기록/랭킹/내 포인트/통계 그래프를 전부 다시 읽어와 화면 갱신.
        db_timer(3초)와 _on_trigger() 양쪽에서 호출됨."""
        rows = fetch_recent_log(self.conn, RECENT_LOG_ROWS)
        self.log_table.setRowCount(len(rows))
        for r, (row_id, ts, phone, cnn_class, cnn_conf, yolo_class, yolo_conf,
                final_class, points, method) in enumerate(rows):
            yolo_disp = f"{yolo_class}({yolo_conf:.2f})" if yolo_class and yolo_conf is not None else "-"
            cnn_disp = f"{cnn_class}({cnn_conf:.2f})" if cnn_class and cnn_conf is not None else "-"
            # "판정" 컬럼: _classify_hybrid()가 반환하는 result["method"]
            # (예: "YOLO", "CNN(3_class)")를 그대로 표시. sorting_log.method 컬럼에
            # 저장해두었기 때문에 여기서 다시 불러올 수 있음.
            values = [ts[:19], phone or "-", yolo_disp, cnn_disp, final_class, str(points), method or "-"]
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
