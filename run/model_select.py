# =====================================================================
# 모델 선택 공용 모듈
# =====================================================================
# models/ 폴더 아래에 모델별로 하위 폴더를 두는 구조를 전제로 합니다.
#
#   models/
#   ├── model_freeze/
#   │   ├── classifier.tflite
#   │   └── class_names.txt
#   ├── model_finetune/
#   │   ├── classifier.tflite
#   │   └── class_names.txt
#   └── model_v2/
#       ├── classifier.tflite
#       └── class_names.txt
#
# 4-1/4-2 계열 스크립트 실행할 때 이 모듈의 select_model()을 호출하면:
#   - 커맨드라인 인자로 폴더명을 주면 그걸 바로 사용
#       예) python3 4-2_test_no_stm32.py model_finetune
#   - 인자 없이 실행하고 모델이 여러 개면 터미널에서 번호로 선택
#   - 모델 폴더가 하나뿐이면 물어보지 않고 그걸 바로 사용
# =====================================================================

import os
import sys


def list_available_models(models_dir):
    """models_dir 하위 폴더 중 classifier.tflite + class_names.txt 가 둘 다 있는 폴더만 나열."""
    found = []
    if not os.path.isdir(models_dir):
        return found
    for name in sorted(os.listdir(models_dir)):
        folder = os.path.join(models_dir, name)
        if (
            os.path.isdir(folder)
            and os.path.exists(os.path.join(folder, "classifier.tflite"))
            and os.path.exists(os.path.join(folder, "class_names.txt"))
        ):
            found.append(name)
    return found


def select_model(models_dir):
    """모델 폴더 하나를 선택해서 (classifier.tflite 경로, class_names.txt 경로)를 반환."""
    models = list_available_models(models_dir)

    if not models:
        print(f"[오류] {models_dir} 안에 사용 가능한 모델 폴더가 없습니다.")
        print("models/모델이름/ 폴더 안에 classifier.tflite, class_names.txt를 넣어주세요.")
        sys.exit(1)

    chosen = None

    # 커맨드라인 인자로 폴더명을 바로 준 경우 (예: python3 script.py model_finetune)
    if len(sys.argv) > 1 and sys.argv[1] in models:
        chosen = sys.argv[1]
    elif len(models) == 1:
        chosen = models[0]
    else:
        print("\n사용 가능한 모델:")
        for i, name in enumerate(models, 1):
            print(f"  {i}. {name}")
        while True:
            sel = input("사용할 모델 번호 입력 > ").strip()
            if sel.isdigit() and 1 <= int(sel) <= len(models):
                chosen = models[int(sel) - 1]
                break
            print("올바른 번호를 입력하세요.")

    print(f"[모델 선택] {chosen}")
    model_dir = os.path.join(models_dir, chosen)
    return os.path.join(model_dir, "classifier.tflite"), os.path.join(model_dir, "class_names.txt")
