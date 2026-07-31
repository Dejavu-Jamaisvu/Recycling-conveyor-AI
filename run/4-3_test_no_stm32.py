# =====================================================================
# 4-2 테스트용: STM32 없이 카메라 + CNN + 포인트 파이프라인을 먼저 검증하는 스크립트
# =====================================================================
# STM32 펌웨어(TRIGGER/CLASS 시리얼 프로토콜)가 아직 준비 안 됐을 때,
# 카메라/조명/ROI/분류 정확도 + 포인트 적립 로직까지 먼저 확인하기 위한 용도입니다.
#
# 실시간 카메라 화면을 창에 띄워두고, 실제 STM32의 "TRIGGER" 신호 대신
# Space 키를 누르면 그 순간 촬영 -> 분류 -> (전화번호 입력되어 있으면) 포인트 적립을 수행합니다.
# 시리얼(pyserial)도, GPIO도 전혀 사용하지 않습니다.
#
# classify() / apply_roi() / 포인트 적립 로직은 4-2_inference_jetson_serial_stm32.py 와
# 완전히 동일합니다. STM32 펌웨어가 준비되면 이 파일은 지우고 그쪽 스크립트로
# 넘어가면 됩니다. (여기서 확인한 FIXED_ROI / CNN_CONF_THRESHOLD 값을 그대로 옮기면 됨)
#
# 폴더 구조 (3개 폴더로 분리, models/는 모델별 하위 폴더로 구성)
#   project/
#   ├── models/
#   │   ├── model_freeze/     classifier.tflite, class_names.txt
#   │   └── model_finetune/   classifier.tflite, class_names.txt
#   ├── run/     이 스크립트 + 4-1/4-2 스크립트들 + model_select.py (여기서 test_snapshots/ 생성됨)
#   └── db/      points.py, db_setup.py, db_view.py, *.db
#
# 실행할 때 어떤 모델을 쓸지 고를 수 있습니다.
#   python3 4-2_test_no_stm32.py                 실행 중 번호로 선택
#   python3 4-2_test_no_stm32.py model_finetune   바로 지정
#
# Jetson Nano에 설치 필요 (최초 1회)
#   pip install opencv-python
#   pip install tflite-runtime
#
# 주의: 카메라 미리보기 창(cv2.imshow)을 띄우므로 모니터가 연결된 상태에서
# (SSH 원격 접속이 아니라 Jetson 본체 데스크톱에서) 실행해야 합니다.
# =====================================================================

import os
import sys
import time
import sqlite3
import threading
from datetime import datetime

import numpy as np
import cv2

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

# 고정 ROI (선택): 물체가 항상 비슷한 자리에 온다면 좌표 지정, 모르면 None
FIXED_ROI = None   # 예: (150, 80, 450, 380)

CNN_CONF_THRESHOLD = 0.6   # 이 이하면 "unknown" 처리
CAMERA_INDEX = 0
# 실제 운영용 sorting_log.db와 섞이지 않게 별도 파일 사용 (db 폴더 안에 생성)
DB_PATH = os.path.join(DB_DIR, "test_sorting_log.db")

SAVE_SNAPSHOTS = True                                    # 촬영 이미지를 저장해서 눈으로 확인하고 싶으면 True
SNAPSHOT_DIR = os.path.join(BASE_DIR, "test_snapshots")  # ROI가 제대로 잡혔는지, 오분류 원인이 뭔지 확인용

# 현재 반납 세션의 사용자 전화번호. 키보드 입력 스레드가 갱신하고,
# 메인 루프가 분류할 때마다 이 값을 읽어서 포인트를 적립합니다.
current_user = {"phone": None}


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
    init_users_table(conn)  # points.py: 사용자별 누적 포인트 테이블 (테스트용 DB에도 별도로 생김)
    conn.commit()
    return conn


def load_class_names(class_names_path):
    with open(class_names_path, encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def apply_roi(frame):
    if FIXED_ROI is None:
        return frame
    x1, y1, x2, y2 = FIXED_ROI
    return frame[y1:y2, x1:x2]


# ---------------------------------------------------------------------
# 키보드로 전화번호 입력받는 스레드
# (미리보기 창은 별도 스레드/루프이므로, 터미널에 그냥 입력하면 됩니다.)
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
            print(f"-> 사용자 전환: {text} (이후 촬영 건은 이 번호로 적립됩니다)")
        else:
            print("전화번호는 숫자만, 4자리 이상 입력해주세요.")


# ---------------------------------------------------------------------
# CNN 분류 (4-2_inference_jetson_serial_stm32.py 와 동일)
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

    # models/ 하위 폴더 중 어떤 모델을 쓸지 선택 (인자로 지정 안 하면 번호로 선택)
    model_path, class_names_path = select_model(MODELS_DIR)
    class_names = load_class_names(class_names_path)

    if SAVE_SNAPSHOTS:
        os.makedirs(SNAPSHOT_DIR, exist_ok=True)

    print("CNN 모델 로딩...")
    cnn_interpreter = Interpreter(model_path=model_path)
    cnn_interpreter.allocate_tensors()

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print(f"카메라(index={CAMERA_INDEX})를 열 수 없습니다. CAMERA_INDEX 값을 확인하세요.")
        return

    # 전화번호 입력용 스레드 시작 (데몬으로 실행 -> 메인 종료되면 같이 종료)
    threading.Thread(target=input_thread_func, daemon=True).start()

    window_name = "Camera Preview - Space: Capture, Q: Quit"
    last_result_text = ""
    last_result_until = 0.0  # 이 시각까지는 화면에 마지막 결과를 계속 표시

    print("\n준비 완료. 미리보기 창에서 Space = 촬영+분류 (STM32의 TRIGGER 대신), Q = 종료\n")
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("카메라 촬영 실패")
                break

            display = frame.copy()
            if FIXED_ROI is not None:
                x1, y1, x2, y2 = FIXED_ROI
                cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # 현재 어떤 사용자로 적립 중인지 화면에도 표시
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
            if key != ord(" "):
                continue

            image = apply_roi(frame)
            t0 = time.perf_counter()
            cnn_class, cnn_conf = classify(cnn_interpreter, image, class_names)
            infer_ms = (time.perf_counter() - t0) * 1000
            final_class = cnn_class if cnn_conf >= CNN_CONF_THRESHOLD else "unknown"

            print(f"  -> class={cnn_class}  conf={cnn_conf:.2f}  최종={final_class}  ({infer_ms:.1f} ms)")
            last_result_text = f"{final_class} ({cnn_conf:.2f})"
            last_result_until = time.time() + 3  # 3초간 화면에 결과 표시

            if SAVE_SNAPSHOTS:
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                fname = os.path.join(SNAPSHOT_DIR, f"{ts}_{final_class}.jpg")
                cv2.imwrite(fname, image)
                print(f"  (스냅샷 저장: {fname})")

            # 포인트 적립 (전화번호가 입력되어 있을 때만)
            phone = current_user["phone"]
            points = 0
            if phone:
                points = award_points(conn, phone, final_class)
                if points > 0:
                    print(f"  [포인트] {phone}님 +{points}점 적립!")

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
        conn.close()


if __name__ == "__main__":
    main()

# =====================================================================
# 사용 방법
#   1) python3 4-2_test_no_stm32.py 실행 (모니터 연결된 Jetson 데스크톱에서)
#   2) 터미널에 전화번호 입력 (또는 Enter로 게스트)
#   3) 뜬 미리보기 창에서 물체를 카메라 앞에 놓고 Space -> 분류 결과가
#      화면에 잠깐 표시되고 터미널에도 출력됨 + 포인트 적립. Q 또는 ESC로 종료
#   4) test_snapshots/ 폴더에서 실제 찍힌 이미지(ROI 적용된) 눈으로 확인
#   5) db 폴더로 이동해서 python3 db_view.py points test_sorting_log.db 로 포인트 랭킹 확인
#      (db_view.py는 db 폴더 안에 있고, 두 번째 인자로 테스트용 DB 파일명을 지정하면 됨)
#   6) FIXED_ROI, CNN_CONF_THRESHOLD 를 실측하며 조정
#   7) STM32 펌웨어 완성되면 -> 4-2_inference_jetson_serial_stm32.py 로 전환
#      (이 테스트 스크립트에서 맞춘 FIXED_ROI / CNN_CONF_THRESHOLD 값을
#       그대로 옮겨 적으면 됨)
# =====================================================================
