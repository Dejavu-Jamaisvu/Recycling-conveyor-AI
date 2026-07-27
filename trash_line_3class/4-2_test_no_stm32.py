# =====================================================================
# 4-2 테스트용: STM32 없이 카메라 + CNN 파이프라인만 먼저 검증하는 스크립트
# =====================================================================
# STM32 펌웨어(TRIGGER/CLASS 시리얼 프로토콜)가 아직 준비 안 됐을 때,
# 카메라/조명/ROI/분류 정확도부터 먼저 확인하기 위한 용도입니다.
#
# 실시간 카메라 화면을 창에 띄워두고, 실제 STM32의 "TRIGGER" 신호 대신
# Space 키를 누르면 그 순간 촬영 -> 분류를 수행합니다.
# 시리얼(pyserial)도, GPIO도 전혀 사용하지 않습니다.
#
# classify() / apply_roi() 로직은 4-2_inference_jetson_serial_stm32.py 와
# 완전히 동일합니다. STM32 펌웨어가 준비되면 이 파일은 지우고 그쪽 스크립트로
# 넘어가면 됩니다.
#
# 사전 준비물 (같은 폴더에 넣어두기)
#   - classifier.tflite, class_names.txt
#
# Jetson Nano에 설치 필요 (최초 1회)
#   pip install opencv-python
#   pip install tflite-runtime
#
# 주의: 카메라 미리보기 창(cv2.imshow)을 띄우므로 모니터가 연결된 상태에서
# (SSH 원격 접속이 아니라 Jetson 본체 데스크톱에서) 실행해야 합니다.
# =====================================================================

import os
import time
import sqlite3
from datetime import datetime

import numpy as np
import cv2

try:
    from tflite_runtime.interpreter import Interpreter
except ImportError:
    from tensorflow.lite.python.interpreter import Interpreter


# ---------------------------------------------------------------------
# 설정값 (확인/수정 필요)
# ---------------------------------------------------------------------
CNN_MODEL_PATH = "classifier.tflite"
CLASS_NAMES_PATH = "class_names.txt"

# 고정 ROI (선택): 물체가 항상 비슷한 자리에 온다면 좌표 지정, 모르면 None
FIXED_ROI = None   # 예: (150, 80, 450, 380)

CNN_CONF_THRESHOLD = 0.6   # 이 이하면 "unknown" 처리
CAMERA_INDEX = 0
DB_PATH = "test_sorting_log.db"   # 실제 운영용 sorting_log.db와 섞이지 않게 별도 파일 사용

SAVE_SNAPSHOTS = True              # 매 촬영 이미지를 저장해서 눈으로 확인하고 싶으면 True
SNAPSHOT_DIR = "test_snapshots"    # ROI가 제대로 잡혔는지, 오분류 원인이 뭔지 확인용


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
    class_names = load_class_names()

    if SAVE_SNAPSHOTS:
        os.makedirs(SNAPSHOT_DIR, exist_ok=True)

    print("CNN 모델 로딩...")
    cnn_interpreter = Interpreter(model_path=CNN_MODEL_PATH)
    cnn_interpreter.allocate_tensors()

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print(f"카메라(index={CAMERA_INDEX})를 열 수 없습니다. CAMERA_INDEX 값을 확인하세요.")
        return

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
            cnn_class, cnn_conf = classify(cnn_interpreter, image, class_names)
            final_class = cnn_class if cnn_conf >= CNN_CONF_THRESHOLD else "unknown"

            print(f"  -> class={cnn_class}  conf={cnn_conf:.2f}  최종={final_class}")
            last_result_text = f"{final_class} ({cnn_conf:.2f})"
            last_result_until = time.time() + 3  # 3초간 화면에 결과 표시

            if SAVE_SNAPSHOTS:
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                fname = os.path.join(SNAPSHOT_DIR, f"{ts}_{final_class}.jpg")
                cv2.imwrite(fname, image)
                print(f"  (스냅샷 저장: {fname})")

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
        cv2.destroyAllWindows()
        conn.close()


if __name__ == "__main__":
    main()

# =====================================================================
# 사용 방법
#   1) python3 4-2_test_no_stm32.py 실행 (모니터 연결된 Jetson 데스크톱에서)
#   2) 뜬 미리보기 창에서 물체를 카메라 앞에 놓고 Space -> 분류 결과가
#      화면에 잠깐 표시되고 터미널에도 출력됨. Q 또는 ESC로 종료
#   3) test_snapshots/ 폴더에서 실제 찍힌 이미지(ROI 적용된) 눈으로 확인
#   4) FIXED_ROI, CNN_CONF_THRESHOLD 를 실측하며 조정
#   5) STM32 펌웨어 완성되면 -> 4-2_inference_jetson_serial_stm32.py 로 전환
#      (이 테스트 스크립트에서 맞춘 FIXED_ROI / CNN_CONF_THRESHOLD 값을
#       그대로 옮겨 적으면 됨)
# =====================================================================
