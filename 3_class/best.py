import argparse
import pathlib

from ultralytics import YOLO

RUNS_DIR = pathlib.Path(__file__).parent / "runs/detect"


def find_latest_weights() -> pathlib.Path:
    candidates = list(RUNS_DIR.glob("*/weights/best.pt"))
    if not candidates:
        raise FileNotFoundError(f"best.pt를 찾을 수 없습니다: {RUNS_DIR}/*/weights/best.pt")
    return max(candidates, key=lambda p: p.stat().st_mtime)


parser = argparse.ArgumentParser(description="YOLO best.pt -> tflite 변환")
parser.add_argument("--weights", type=pathlib.Path, default=None, help="best.pt 경로 (생략 시 가장 최근 학습 결과 자동 사용)")
parser.add_argument("--imgsz", type=int, default=640)
args = parser.parse_args()

weights = args.weights or find_latest_weights()
print(f"사용할 가중치: {weights}")

model = YOLO(str(weights))
model.export(format="tflite", imgsz=args.imgsz)