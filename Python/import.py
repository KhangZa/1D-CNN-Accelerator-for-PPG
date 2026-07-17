import os
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Conv1D, ReLU, AveragePooling1D, Dense,
    BatchNormalization, Dropout, GlobalAveragePooling1D,
    MultiHeadAttention
)
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import StandardScaler

# ==========================================
# 1. CẤU HÌNH ĐƯỜNG DẪN VÀ THAM SỐ
# ==========================================
SPLIT_DIR = "/content/drive/MyDrive/Train/TEMPONet/LOSOSplits"
MODEL_DIR = "/content/drive/MyDrive/Train/TEMPONet/FINAL_EXACT_LOSO_ARCH_MHA"
os.makedirs(MODEL_DIR, exist_ok=True)

BATCH_SIZE = 128
EPOCHS_PRETRAIN = 80
EPOCHS_FINETUNE = 20
LEARNING_RATE_PRETRAIN = 0.001
LEARNING_RATE_FINETUNE = 1e-5
PATIENCE_SCHEDULER = 5
PATIENCE_EARLY_STOP = 10
MIN_LR = 1e-6

# ==========================================
# 2. HÀM XÂY DỰNG MODEL (GIỮ NGUYÊN 100% ARCH MHA)
# ==========================================
def build_new_model(input_shape=(256, 4)):
    inputs = Input(shape=input_shape)

    # === Block 1 (32 channels) ===
    x = Conv1D(filters=32, kernel_size=9, dilation_rate=1, padding='same')(inputs)
    x = ReLU()(x)
    x = Conv1D(filters=32, kernel_size=9, dilation_rate=1, padding='same')(x)
    x = ReLU()(x)
    x = Conv1D(filters=32, kernel_size=9, dilation_rate=1, padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = AveragePooling1D(pool_size=4)(x)
    x = Dropout(0.2)(x)

    # === Block 2 (48 channels) ===
    x = Conv1D(filters=48, kernel_size=9, dilation_rate=1, padding='same')(x)
    x = ReLU()(x)
    x = Conv1D(filters=48, kernel_size=9, dilation_rate=1, padding='same')(x)
    x = ReLU()(x)
    x = Conv1D(filters=48, kernel_size=9, dilation_rate=1, padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = AveragePooling1D(pool_size=2)(x)
    x = Dropout(0.2)(x)

    # === Block 3 (64 channels) ===
    x = Conv1D(filters=64, kernel_size=9, dilation_rate=1, padding='same')(x)
    x = ReLU()(x)
    x = Conv1D(filters=64, kernel_size=9, dilation_rate=1, padding='same')(x)
    x = ReLU()(x)
    x = Conv1D(filters=64, kernel_size=9, dilation_rate=1, padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = AveragePooling1D(pool_size=2)(x)
    x = Dropout(0.2)(x)

    # === Attention Block ===
    attn_output = MultiHeadAttention(num_heads=4, key_dim=16)(
        query=x, value=x, key=x
    )
    x = BatchNormalization()(attn_output)

    # === Các lớp Dense Output ===
    x = GlobalAveragePooling1D()(x)
    x = Dense(32, activation='relu')(x)
    outputs = Dense(1)(x)

    model = Model(inputs=inputs, outputs=outputs)
    return model

# ==========================================
# 3. HÀM TRAIN/TEST CHO 1 CHỦ THỂ (LOSO FOLD)
# ==========================================
def train_one_loso(test_subj, split_file, is_first_fold=False):
    with open(split_file, "rb") as f:
        split = pickle.load(f)

    # Tải dữ liệu
    X_train_raw = split['X_train']
    y_train = split['y_train']
    X_test_subj_raw = split['X_test']
    y_test_subj = split['y_test']

    # Tách 25% dữ liệu Test của chủ thể để Fine-tuning
    split_idx = int(len(X_test_subj_raw) * 0.25)
    X_finetune_raw = X_test_subj_raw[:split_idx]
    y_finetune = y_test_subj[:split_idx]
    X_test_final_raw = X_test_subj_raw[split_idx:]
    y_test_final = y_test_subj[split_idx:]

    # Chuẩn hóa dữ liệu
    scaler = StandardScaler()
    X_train_2d = X_train_raw.reshape(-1, X_train_raw.shape[-1])
    scaler.fit(X_train_2d)

    # Lưu Scaler riêng cho Fold này
    scaler_path = os.path.join(MODEL_DIR, f"Scaler_LOSO_{test_subj}.pkl")
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)

    X_train = scaler.transform(X_train_2d).reshape(X_train_raw.shape)
    X_finetune_scaled = scaler.transform(X_finetune_raw.reshape(-1, 4)).reshape(X_finetune_raw.shape)
    X_test_final_scaled = scaler.transform(X_test_final_raw.reshape(-1, 4)).reshape(X_test_final_raw.shape)

    # Khởi tạo mô hình
    model = build_new_model(input_shape=X_train.shape[1:])

    # In Parameter Summary
    if is_first_fold:
        model.summary()

    optimizer_pretrain = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE_PRETRAIN)

    model.compile(
        optimizer=optimizer_pretrain,
        loss='mse',
        metrics=['mae']
    )

    # Callbacks
    generic_model_path = os.path.join(MODEL_DIR, f"TEMPONet_MHA_LOSO_{test_subj}_best.h5")
    callback_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=PATIENCE_SCHEDULER, min_lr=MIN_LR, verbose=1)
    callback_es = EarlyStopping(monitor='val_loss', patience=PATIENCE_EARLY_STOP, verbose=1)
    callback_mc = ModelCheckpoint(filepath=generic_model_path, monitor='val_loss', save_best_only=True, verbose=0)

    print(f"\n--- BƯỚC 1: Huấn luyện chung (Test Fold: {test_subj}) ---")
    model.fit(
        X_train, y_train,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS_PRETRAIN,
        validation_data=(X_finetune_scaled, y_finetune),
        callbacks=[callback_lr, callback_es, callback_mc],
        verbose=1
    )

    # BƯỚC 2: Tinh chỉnh (Fine-tuning)
    print(f"\n--- BƯỚC 2: Tinh chỉnh cho {test_subj} ---")
    best_model = tf.keras.models.load_model(generic_model_path, compile=False)
    optimizer_finetune = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE_FINETUNE)

    best_model.compile(optimizer=optimizer_finetune, loss='mse', metrics=['mae'])
    best_model.fit(
        X_finetune_scaled, y_finetune,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS_FINETUNE,
        verbose=2
    )

    best_model.save(generic_model_path)

    # BƯỚC 3: Đánh giá
    print(f"\n--- BƯỚC 3: Đánh giá kết quả cuối cùng trên 75% dữ liệu Test ---")
    results = best_model.evaluate(X_test_final_scaled, y_test_final, verbose=0)
    best_mae = results[1]

    print(f"[{test_subj}] Hoàn tất! MAE Cuối cùng = {best_mae:.4f} BPM")
    return results[0], best_mae

# ==========================================
# 4. THỰC THI LOSO TRÊN TẤT CẢ CHỦ THỂ
# ==========================================
test_losses = []
test_maes = []

print("Bắt đầu thực hiện LOSO với kiến trúc TEMPONet MHA...")

files = sorted(os.listdir(SPLIT_DIR))
for i, subj_file in enumerate(files):
    if subj_file.endswith("_split.pkl"):
        test_subj = subj_file.split("_")[0]
        split_file = os.path.join(SPLIT_DIR, subj_file)

        is_first = (i == 0) # Flag báo hiệu fold đầu tiên
        loss, mae = train_one_loso(test_subj, split_file, is_first_fold=is_first)

        test_losses.append(loss)
        test_maes.append(mae)

# ==========================================
# 5. TỔNG KẾT
# ==========================================
print("\n" + "="*50)
print(" TỔNG KẾT ĐÁNH GIÁ LOSO ")
print("="*50)
print(f"MAE trung bình (Overall): {np.mean(test_maes):.4f} BPM")
print(f"Độ lệch chuẩn (Std):      {np.std(test_maes):.4f} BPM")
print("="*50)