# =====================================================================
# 1단계: YOLO 위치 검출 모델 학습 (Google Colab에서 실행 권장)
# =====================================================================
# 사용 순서
# 1) https://colab.research.google.com 접속 → 새 노트북
# 2) 상단 메뉴 [런타임] > [런타임 유형 변경] > 하드웨어 가속기 = GPU 선택
# 3) 이 파일 내용을 셀에 붙여넣고 위에서부터 순서대로 실행
# 4) ROBOFLOW_API_KEY, WORKSPACE, PROJECT, VERSION 은 본인 Roboflow
#    프로젝트 페이지 우측 상단 "Download Dataset" 버튼 눌렀을 때 나오는
#    코드에서 그대로 복사하면 됩니다.
# =====================================================================

# --- 설치 ---
# !pip install ultralytics roboflow -q

# --- 1) Roboflow에서 바운딩박스 라벨 포함 데이터셋 다운로드 ---
from roboflow import Roboflow

ROBOFLOW_API_KEY = "08pEqA03ywLShGQ8Vk09"
WORKSPACE = "s-workspace-ntur3"
PROJECT = "1trashset"
VERSION = 1  # Roboflow 프로젝트 버전 번호

rf = Roboflow(api_key=ROBOFLOW_API_KEY)
project = rf.workspace(WORKSPACE).project(PROJECT)
dataset = project.version(VERSION).download("yolov8")
# 다운로드된 폴더 안에 data.yaml (클래스 이름 정의) + train/valid/test 가 생김

print("데이터셋 위치:", dataset.location)

# --- 2) YOLOv8n(nano)으로 전이학습 ---
# nano 버전을 쓰는 이유: Jetson Nano처럼 연산이 약한 보드에 올리기엔
# 가장 가벼운 버전이 안전합니다. (s/m/l/x 로 갈수록 무겁고 정확하지만 느려짐)
from ultralytics import YOLO

model = YOLO("yolov8n.pt")

results = model.train(
    data=f"{dataset.location}/data.yaml",
    epochs=100,
    imgsz=640,
    batch=16,
    name="waste_yolo",
    patience=20,       # 20 epoch 동안 성능 개선 없으면 조기 종료
)

# --- 3) 학습 결과 확인 ---
# 학습이 끝나면 다음 경로에 결과가 저장됩니다.
#   runs/detect/waste_yolo/weights/best.pt   <- 최종 모델 (이걸 사용)
#   runs/detect/waste_yolo/confusion_matrix.png  <- 클래스별 오분류 확인
#   runs/detect/waste_yolo/results.png           <- 학습 곡선(mAP, loss 등)
#
# best.pt 를 다운로드해서 로컬에 저장해두세요.
# (Colab 왼쪽 파일 탐색기에서 우클릭 > 다운로드)

# --- 4) 학습된 모델로 실제 이미지 테스트 (선택) ---
# best_model = YOLO("runs/detect/waste_yolo/weights/best.pt")
# results = best_model.predict("테스트할_이미지_경로.jpg", save=True, conf=0.5)

# =====================================================================
# 다음 단계: best.pt 로 원본 학습 이미지들의 바운딩박스를 잘라내서
# CNN 분류기용 데이터셋을 만듭니다. -> 2_crop_bboxes_for_cnn.py 참고
# =====================================================================