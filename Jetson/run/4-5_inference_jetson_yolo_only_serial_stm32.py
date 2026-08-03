# =====================================================================
# 4-5단계: Jetson Nano 실시간 추론 (STM32 UART + YOLO 단독 - 위치검출+분류 통합)
# =====================================================================
# 4-4와 다른 점: CNN 단계를 아예 없앴습니다. YOLO 모델이 위치(바운딩박스)와
# 클래스(metal/plastic/paper)를 한 번의 추론으로 같이 내놓기 때문에, 그 결과를
# 그대로 최종 분류로 사용합니다. 추론이 한 번만 도니까 더 빠르고 구조도
# 단순하지만, YOLO의 분류 정확도가 CNN(3_train_cnn_classifier.py로 따로
# 학습한 모델)보다 낮을 수 있습니다 - 실제로 써보고 정확도 비교해보세요.
# (다시 CNN 병행 방식으로 돌아가고 싶으면 4-4_inference_jetson_yolo_cnn_serial_stm32.py
# 를 쓰면 됩니다. 두 파일 다 남겨뒀습니다.)
#
# *** 실제 하드웨어 구조 ***
#   Jetson --- UART(1:1) --- STM32 --- (IR센서, 서보 3개, 아두이노에 컨베이어
#   제어 중계). Jetson은 GPIO를 직접 쓰지 않고 STM32하고만 UART로 통신합니다.
#     STM32 -> Jetson : "TRIGGER"        (IR센서가 물체 감지 + 벨트 정지 완료)
#     Jetson -> STM32 : "CLASS:metal" 등  (분류 결과 - STM32가 서보 작동시킴)
#
# YOLO 모델은 두 가지 방식을 지원합니다 (아래 우선순위로 자동 선택).
#   1) best.pt          - ultralytics로 바로 로딩. 클래스 이름이 모델 파일
#      안에 학습 당시 정보로 이미 저장돼 있어서 별도 파일이 필요 없습니다.
#      (torch/ultralytics 설치 필요)
#   2) detector.tflite  - tflite_runtime으로 로딩. tflite 파일 자체에는 클래스
#      "이름"이 저장 안 되므로, 같은 폴더에 detector_labels.txt를 같이 둬야
#      합니다 (한 줄에 클래스 하나씩, 학습 때 순서와 동일하게).
#
#   찾는 위치: models/detector/ 폴더를 먼저 보고, 없으면 models/ 바로 아래
#   모든 하위 폴더를 하나씩 뒤집니다 (예: models/3class/best.pt 도 인식됨).
#
#   중요: 클래스 이름은 정확히 "metal", "plastic", "paper" (소문자)여야
#   포인트가 정상 적립됩니다 (db/points.py의 POINTS_PER_CLASS 참고). 학습
#   데이터셋의 클래스 이름이 다르면(예: Metal, METAL) 소문자로 자동 변환은
#   하지만 철자 자체가 다르면 포인트가 0으로 처리되니 확인하세요.
#
# 하드웨어 설정 (4-2/4-4에서 이어받음)
#   - SERIAL_PORT: /dev/ttyTHS1 (J41 핀헤더 UART)
#   - 카메라 버퍼 크기 1 + 워밍업 10프레임
#   - 시리얼 수신은 별도 스레드 (trigger_event) + 메인 스레드는 미리보기 루프
#
# 사전 준비 (최초 1회):
#   sudo systemctl stop nvgetty && sudo systemctl disable nvgetty
#   sudo usermod -aG dialout $USER
#   sudo reboot
#
# 폴더 구조
#   project/
#   ├── models/
#   │   └── detector/ (또는 아무 하위 폴더)   best.pt 또는
#   │                                          detector.tflite + detector_labels.txt
#   ├── run/     이 스크립트 + 4-x 계열 스크립트들
#   └── db/      points.py, db_setup.py, db_view.py, *.db
#
# 실행: python3 4-5_inference_jetson_yolo_only_serial_stm32.py
#
# 설치
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

try:
    from tflite_runtime.interpreter import Interpreter
except ImportError:
    from tensorflow.lite.python.interpreter import Interpreter


# ---------------------------------------------------------------------
# 설정값
# ---------------------------------------------------------------------
SERIAL_PORT = "/dev/ttyTHS1"   # J41 핀헤더 UART
SERIAL_BAUDRATE = 115200

CAMERA_INDEX = 0
DB_PATH = os.path.join(DB_DIR, "sorting_log.db")
LAST_CAPTURE_PATH = os.path.join(BASE_DIR, "last_capture.jpg")

DETECTOR_DIR = os.path.join(MODELS_DIR, "detector")
DETECTOR_CONF_THRESHOLD = 0.5   # 이 미만이면 "unknown" 처리 (CNN이 없으니 이 값이 최종 기준)

current_user = {"phone": None}
trigger_event = threading.Event()


# ---------------------------------------------------------------------
# 초기화
# ---------------------------------------------------------------------
def _ensure_column(conn, table, column, coltype):
    """예전 스크립트(4-1~4-4)로 이미 만들어진 sorting_log.db를 이어서 쓸 때
    새로 추가된 컬럼이 없으면 채워넣습니다."""
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
    _ensure_column(conn, "sorting_log", "infer_ms", "REAL")  # YOLO 추론 1회 소요시간(ms)
    init_users_table(conn)
    conn.commit()
    return conn


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


# ---------------------------------------------------------------------
# YOLO 검출기 (위치 + 분류를 한 번에 반환)
# ---------------------------------------------------------------------
def load_labels(path):
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


class TFLiteDetector:
    """detector.tflite(ultralytics tflite export) 기반. 클래스 이름은
    같은 폴더의 detector_labels.txt에서 읽어옵니다 (없으면 class_0, class_1...)."""

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
        """반환: (crop 또는 None, confidence, box 또는 None, class_name 또는 None)"""
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
        class_name = self._class_name(best_class_id)
        if crop.size == 0:
            return None, best_conf, None, class_name
        return crop, best_conf, (x1, y1, x2, y2), class_name


class UltralyticsDetector:
    """best.pt를 ultralytics로 바로 로딩. 클래스 이름은 모델 파일 안에
    학습 당시 정보(self.model.names)로 이미 저장돼 있어서 별도 파일 불필요."""

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
    """models/detector/ 를 먼저 보고, 없으면 models/ 바로 아래 모든 하위
    폴더를 뒤집니다 (예: models/3class/best.pt 같은 배치도 인식되게)."""
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
    """찾은 YOLO 모델이 여러 개면 CNN 모델 선택(model_select.py)과 똑같이
    번호로 고르게 합니다. 커맨드라인 인자로 폴더명을 바로 줄 수도 있음
    (예: python3 4-5_....py 3class)."""
    candidates = find_detector_candidates()
    if not candidates:
        print(f"YOLO 모델을 찾지 못했습니다 (찾아본 위치: {', '.join(_find_candidate_dirs())})")
        print("models/detector/ 안에 best.pt 또는 detector.tflite를 넣어주세요.")
        sys.exit(1)

    chosen = None

    if len(sys.argv) > 1:
        names = [os.path.basename(d) for d, _ in candidates]
        if sys.argv[1] in names:
            chosen = candidates[names.index(sys.argv[1])]

    if chosen is None:
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
        print(f"로딩 실패: {e}")
        if kind == "pt":
            print('-> 확인: python3 -c "import torch, ultralytics"')
        sys.exit(1)


# ---------------------------------------------------------------------
# 메인 루프
# ---------------------------------------------------------------------
def main():
    conn = init_db()

    print("YOLO 모델 로딩...")
    detector = load_detector()

    print(f"STM32 시리얼 연결 시도... ({SERIAL_PORT}, {SERIAL_BAUDRATE}bps)")
    ser = serial.Serial(SERIAL_PORT, SERIAL_BAUDRATE, timeout=1)
    ser.reset_input_buffer()
    time.sleep(2)

    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if not cap.isOpened():
        print(f"카메라(index={CAMERA_INDEX})를 열 수 없습니다. CAMERA_INDEX 값을 확인하세요.")
        return
    for _ in range(10):
        cap.read()

    threading.Thread(target=input_thread_func, daemon=True).start()
    threading.Thread(target=serial_listener_func, args=(ser,), daemon=True).start()

    window_name = "Camera Preview (STM32 TRIGGER, YOLO 단독) - Q: Quit"
    last_result_text = ""
    last_result_until = 0.0
    last_box = None
    last_box_label = ""
    last_box_until = 0.0

    print("대기 중... (STM32로부터 TRIGGER 신호 대기, 미리보기 창에서 Q로 종료)")
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("카메라 촬영 실패")
                break

            display = frame.copy()
            if last_box is not None and time.time() < last_box_until:
                bx1, by1, bx2, by2 = last_box
                cv2.rectangle(display, (bx1, by1), (bx2, by2), (255, 220, 0), 2)
                if last_box_label:
                    cv2.putText(
                        display, last_box_label, (bx1, max(0, by1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 220, 0), 2,
                    )

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
            if key in (ord("q"), ord("Q"), 27):
                break

            if trigger_event.is_set():
                trigger_event.clear()
                cv2.imwrite(LAST_CAPTURE_PATH, frame)

                # YOLO 한 번으로 위치 + 클래스 다 얻음 (CNN 없음)
                t0 = time.perf_counter()
                _, yolo_conf, box, yolo_class = detector.detect_and_crop(frame)
                infer_ms = (time.perf_counter() - t0) * 1000
                last_box = box
                last_box_label = f"{yolo_class} {yolo_conf:.2f}" if yolo_class else ""
                last_box_until = time.time() + 3

                if yolo_class is not None and box is not None:
                    final_class = yolo_class.strip().lower()
                else:
                    final_class = "unknown"

                print(f"YOLO class={yolo_class} conf={yolo_conf:.2f} -> final={final_class} ({infer_ms:.1f} ms)")

                last_result_text = f"{final_class} ({yolo_conf:.2f})"
                last_result_until = time.time() + 3

                ser.write(f"CLASS:{final_class}\n".encode("utf-8"))

                phone = current_user["phone"]
                points = 0
                if phone:
                    points = award_points(conn, phone, final_class)
                    if points > 0:
                        print(f"  [포인트] {phone}님 +{points}점 적립!")
                    elif final_class != "unknown":
                        print(f"  [알림] '{final_class}'는 포인트 규칙에 없는 클래스명입니다. "
                              f"db/points.py의 POINTS_PER_CLASS와 철자를 맞춰보세요.")

                # cnn_class / cnn_confidence 컬럼은 db_setup.py로 테이블을 처음 만들
                # 때 둘 다 NOT NULL로 정의돼 있어서 (예전 CNN 버전 스크립트들 기준),
                # 4-5는 CNN이 없어도 값을 채워야 저장이 됨 - yolo 결과로 대신 채워넣음
                conn.execute(
                    """
                    INSERT INTO sorting_log
                        (timestamp, phone, cnn_class, cnn_confidence, yolo_class,
                         yolo_confidence, final_class, points_awarded, infer_ms)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        datetime.now().isoformat(),
                        phone,
                        yolo_class or "unknown",
                        yolo_conf,
                        yolo_class,
                        yolo_conf,
                        final_class,
                        points,
                        infer_ms,
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
#   - SERIAL_PORT / SERIAL_BAUDRATE: STM32 펌웨어와 일치해야 함
#   - models/detector/ (또는 models/아무폴더/) 안에 best.pt 또는
#     detector.tflite + detector_labels.txt 필요
#   - 클래스 이름이 "metal"/"plastic"/"paper"와 정확히 일치해야 포인트 적립됨
#   - DETECTOR_CONF_THRESHOLD: 실제 테스트하면서 오분류율 보고 조정
#     (CNN이 없으니 이 임계값이 유일한 확신도 기준입니다)
#   - 모니터가 연결된 Jetson 본체 데스크톱(또는 VNC)에서 실행해야 미리보기 창이 뜸
# =====================================================================
