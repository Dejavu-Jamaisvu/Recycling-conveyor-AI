#!/usr/bin/env python3
# 3.ipynb의 저장 셀을 실행하면 출력되는 폴더 ID를 아래에 붙여넣고 실행하세요.
#
#   pip install gdown   (최초 1회)
#   python3 3_class/pull_results.py

import subprocess
import pathlib

FOLDER_ID = "1NVM42UWV7zxyu_MisV7o4fllAnvL7d6a"

DEST = pathlib.Path(__file__).parent
DEST.mkdir(parents=True, exist_ok=True)

subprocess.run(["gdown", "--folder", FOLDER_ID, "-O", str(DEST)], check=True)
print(f"다운로드 완료: {DEST}")
