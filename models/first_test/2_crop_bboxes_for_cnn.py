# =====================================================================
# 2단계: Roboflow 바운딩박스 라벨로 이미지를 잘라 CNN 학습용 데이터셋 생성
# =====================================================================
# Roboflow에서 받은 YOLO 포맷 데이터셋 구조 (1번 스크립트로 받은 그 폴더):
#   dataset/
#     train/images/*.jpg   train/labels/*.txt
#     valid/images/*.jpg   valid/labels/*.txt
#     test/images/*.jpg    test/labels/*.txt
#     data.yaml            (클래스 이름 순서 정의)
#
# labels/*.txt 한 줄 형식 (YOLO 표준):
#   class_id x_center y_center width height   (모두 0~1로 정규화된 값)
#
# 이 스크립트는 사람이 직접 그린 정답 바운딩박스(라벨)를 기준으로 자르기
# 때문에, 나중에 YOLO 예측 결과로 크롭할 때보다 훨씬 깨끗한 학습 데이터가
# 만들어집니다. (모델 예측 오차가 섞이지 않음)
#
# 실행 전 아래 DATASET_DIR 경로만 본인 환경에 맞게 수정하세요.
# =====================================================================

import os
import yaml
from PIL import Image

DATASET_DIR = "dataset"      # 1_train_yolo_colab.py 의 dataset.location 경로
OUTPUT_DIR = "cnn_dataset"   # 크롭된 이미지가 저장될 폴더 (자동 생성됨)
SPLITS = ["train", "valid", "test"]
PADDING_RATIO = 0.08         # 바운딩박스 주변 여백을 8% 추가 (물체가 딱 붙어 잘리는 것 방지)


def load_class_names(dataset_dir):
    with open(os.path.join(dataset_dir, "data.yaml"), encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg["names"]


def crop_split(dataset_dir, output_dir, split, class_names):
    img_dir = os.path.join(dataset_dir, split, "images")
    lbl_dir = os.path.join(dataset_dir, split, "labels")
    if not os.path.isdir(img_dir):
        print(f"[스킵] {split} 폴더 없음")
        return 0

    count = 0
    for fname in os.listdir(img_dir):
        if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        stem = os.path.splitext(fname)[0]
        label_path = os.path.join(lbl_dir, stem + ".txt")
        if not os.path.exists(label_path):
            continue

        img_path = os.path.join(img_dir, fname)
        img = Image.open(img_path).convert("RGB")
        W, H = img.size

        with open(label_path) as lf:
            lines = [l.strip() for l in lf if l.strip()]

        for i, line in enumerate(lines):
            parts = line.split()
            cls_id = int(parts[0])
            xc, yc, w, h = map(float, parts[1:5])
            cls_name = class_names[cls_id]

            # 여백 추가
            w *= (1 + PADDING_RATIO)
            h *= (1 + PADDING_RATIO)

            # YOLO 정규화 좌표 -> 픽셀 좌표
            x1 = int(max(0, (xc - w / 2) * W))
            y1 = int(max(0, (yc - h / 2) * H))
            x2 = int(min(W, (xc + w / 2) * W))
            y2 = int(min(H, (yc + h / 2) * H))

            if x2 <= x1 or y2 <= y1:
                continue  # 잘못된 박스 방어

            crop = img.crop((x1, y1, x2, y2))

            out_dir = os.path.join(output_dir, split, cls_name)
            os.makedirs(out_dir, exist_ok=True)
            crop.save(os.path.join(out_dir, f"{stem}_{i}.jpg"), quality=95)
            count += 1

    return count


def main():
    class_names = load_class_names(DATASET_DIR)
    print("클래스:", class_names)

    total = 0
    for split in SPLITS:
        n = crop_split(DATASET_DIR, OUTPUT_DIR, split, class_names)
        print(f"{split}: {n}개 크롭 완료")
        total += n

    print(f"\n전체 {total}개 이미지 크롭 완료 -> {OUTPUT_DIR}/")
    print("각 클래스별 폴더 안의 이미지 개수가 너무 차이나면(클래스 불균형)")
    print("학습이 한쪽 클래스로 쏠릴 수 있으니 개수를 한번 확인해보세요.")


if __name__ == "__main__":
    main()

# =====================================================================
# 다음 단계: cnn_dataset/ 폴더로 CNN 분류기 학습 -> 3_train_cnn_classifier.py 참고
# =====================================================================
