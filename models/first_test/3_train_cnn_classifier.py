# =====================================================================
# 3단계: CNN 분류기 학습 (MobileNetV2 전이학습, freeze / 파인튜닝 두 버전)
# =====================================================================
# 2_crop_bboxes_for_cnn.py 로 만든 cnn_dataset/ 폴더를 사용합니다.
# cnn_dataset/train/plastic, cnn_dataset/train/glass, cnn_dataset/train/can ...
#
# freeze 버전과 파인튜닝 버전을 각각 학습해서 검증 정확도를 비교한 뒤,
# 더 나은 쪽을 최종 모델로 선택하면 됩니다. (Google Colab GPU 권장)
# =====================================================================

import tensorflow as tf
from tensorflow.keras import layers, models

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 15
DATA_DIR = "cnn_dataset"   # 2번 스크립트의 OUTPUT_DIR과 동일 경로로 수정

# --- 데이터 불러오기 ---
train_ds = tf.keras.utils.image_dataset_from_directory(
    f"{DATA_DIR}/train", image_size=IMG_SIZE, batch_size=BATCH_SIZE
)
val_ds = tf.keras.utils.image_dataset_from_directory(
    f"{DATA_DIR}/valid", image_size=IMG_SIZE, batch_size=BATCH_SIZE
)

class_names = train_ds.class_names
print("클래스:", class_names)  # 예: ['can', 'glass', 'plastic']

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# 데이터 증강: 위치가 살짝씩 달라지는 실제 환경 대비
augment = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
    layers.RandomBrightness(0.15),  # 조명 변화 대비 (유리·캔 반사 이슈 보완)
])
normalization = layers.Rescaling(1.0 / 127.5, offset=-1)  # MobileNetV2 표준 전처리


def build_model(base_trainable=False, fine_tune_at=100):
    base = tf.keras.applications.MobileNetV2(
        input_shape=IMG_SIZE + (3,), include_top=False, weights="imagenet"
    )
    base.trainable = base_trainable
    if base_trainable:
        # fine_tune_at 이전 레이어는 계속 고정, 이후 레이어만 재학습
        for layer in base.layers[:fine_tune_at]:
            layer.trainable = False

    inputs = tf.keras.Input(shape=IMG_SIZE + (3,))
    x = augment(inputs)
    x = normalization(x)
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(len(class_names), activation="softmax")(x)

    model = models.Model(inputs, outputs)
    lr = 1e-5 if base_trainable else 1e-3  # 파인튜닝은 학습률을 훨씬 낮게
    model.compile(
        optimizer=tf.keras.optimizers.Adam(lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


# --- 1) freeze 버전: 특징 추출부 고정, 마지막 분류층만 학습 ---
print("\n=== freeze 버전 학습 ===")
freeze_model = build_model(base_trainable=False)
freeze_history = freeze_model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS)
freeze_model.save("model_freeze.keras")

# --- 2) 파인튜닝 버전: 상위 레이어까지 낮은 학습률로 재학습 ---
print("\n=== 파인튜닝 버전 학습 ===")
finetune_model = build_model(base_trainable=True, fine_tune_at=100)
finetune_history = finetune_model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS)
finetune_model.save("model_finetune.keras")

# --- 3) 두 모델 검증 정확도 비교 ---
freeze_loss, freeze_acc = freeze_model.evaluate(val_ds)
finetune_loss, finetune_acc = finetune_model.evaluate(val_ds)
print(f"\nfreeze 검증 정확도:   {freeze_acc:.4f}")
print(f"finetune 검증 정확도: {finetune_acc:.4f}")

best_model = finetune_model if finetune_acc >= freeze_acc else freeze_model
best_name = "finetune" if finetune_acc >= freeze_acc else "freeze"
print(f"\n-> 최종 선택 모델: {best_name} 버전")

# --- 4) 최종 모델을 .tflite로 변환 (Jetson Nano 등 경량 기기 배포용) ---
converter = tf.lite.TFLiteConverter.from_keras_model(best_model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]  # 크기/속도 최적화
tflite_model = converter.convert()

with open("classifier.tflite", "wb") as f:
    f.write(tflite_model)

# 클래스 순서도 같이 저장해둬야 나중에 추론 결과 인덱스를 해석할 수 있음
with open("class_names.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(class_names))

print("\n완료: classifier.tflite, class_names.txt 저장됨")

# =====================================================================
# 다음 단계
# 1) best.pt (YOLO), classifier.tflite (CNN), class_names.txt 를
#    Jetson Nano로 옮김
# 2) 실제 카메라로 YOLO 검출 -> 바운딩박스 크롭 -> CNN 추론 파이프라인 테스트
# 3) 정지-촬영 방식이면: IR센서 감지 -> 벨트 정지 -> 촬영 -> 이 파이프라인
#    -> 서보 매핑 -> 벨트 재가동 순서로 통합
# =====================================================================
