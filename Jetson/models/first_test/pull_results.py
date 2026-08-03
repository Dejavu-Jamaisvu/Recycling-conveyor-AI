#!/usr/bin/env python3
# Colab 노트북(1.ipynb)의 3-1 셀을 실행하면 출력되는 폴더 ID를 아래에 붙여넣고 실행하세요.
# 처음 한 번만 채워두면, 이후 학습을 다시 돌릴 때마다 이 스크립트만 실행하면 됩니다.
#
#   pip install gdown   (최초 1회)
#   python3 first_test/pull_results.py

import subprocess
import pathlib

FOLDER_ID = "여기에_폴더_ID_붙여넣기"

DEST = pathlib.Path(__file__).parent / "runs" / "detect" / "waste_yolo" / "weights"
DEST.mkdir(parents=True, exist_ok=True)

subprocess.run(["gdown", "--folder", FOLDER_ID, "-O", str(DEST)], check=True)
print(f"다운로드 완료: {DEST}")
