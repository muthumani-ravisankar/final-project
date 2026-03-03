import os
import librosa
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")

# ---------------- SEEDS ----------------
tf.random.set_seed(42)
np.random.seed(42)

# ---------------- CONFIG ----------------
class Config:
    SR = 16000
    N_MELS = 64
    MAX_PAD_LEN = 300
    BATCH_SIZE = 32
    EPOCHS = 25         # <= REDUCED FOR GPU
    LR = 0.0003
    TRAIN_SAMPLES = 20000

config = Config()

# Enable GPU memory growth (recommended)
gpus = tf.config.list_physical_devices("GPU")
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"✅ Enabled memory growth on {len(gpus)} GPU(s).")
    except Exception as e:
        print("⚠ Could not enable GPU memory growth:", e)

# ---------------- PATHS ----------------
def get_paths():
    base = r"C:/Users/padma/Downloads/LA (1)/LA"
    return {
        "train_audio": os.path.join(base, "ASVspoof2019_LA_train", "flac"),
        "protocols": os.path.join(base, "ASVspoof2019_LA_cm_protocols"),
    }

paths = get_paths()

# ---------------- FIND PROTOCOL FILE ----------------
def find_train_protocol_file():
    proto_dir = paths["protocols"]
    all_files = os.listdir(proto_dir)

    print("\n📂 Protocols directory contents:")
    for f in all_files:
        print("   -", f)

    candidates = [f for f in all_files if "cm" in f.lower() and "train" in f.lower()]
    if not candidates:
        print("❌ No train protocol file found.")
        return None

    chosen = candidates[0]
    print(f"\n✅ Using protocol file: {chosen}")
    return os.path.join(proto_dir, chosen)

# ---------------- PROTOCOL LOADER ----------------
def load_protocol():
    proto = find_train_protocol_file()
    audio_dir = paths["train_audio"]

    print("\n🔍 Train audio path:", audio_dir)

    all_audio = [f for f in os.listdir(audio_dir) if f.endswith(".flac")]
    file_map = {os.path.splitext(f)[0]: f for f in all_audio}

    print(f"📂 Found {len(file_map)} train files")

    rows, matched, skipped = [], 0, 0

    with open(proto, "r") as f:
        for line in f:
            parts = line.strip().split()
            file_id = parts[1]      # correct ID column
            label_str = parts[-1]   # bonafide/spoof

            if file_id not in file_map:
                skipped += 1
                continue

            label = 1 if label_str == "bonafide" else 0
            matched += 1

            rows.append({"filename": file_map[file_id], "label": label})

    df = pd.DataFrame(rows)

    print(f"📄 Matched: {matched}, Skipped: {skipped}")

    return df

# ---------------- FEATURE EXTRACTION ----------------
def extract_features(path):
    try:
        y, sr = librosa.load(path, sr=config.SR, duration=4)

        S = librosa.feature.melspectrogram(
            y=y,
            sr=sr,
            n_mels=config.N_MELS
        )

        logS = librosa.power_to_db(S, ref=np.max)
        d1 = librosa.feature.delta(logS)
        d2 = librosa.feature.delta(logS, order=2)

        feat = np.vstack([logS, d1, d2])

        if feat.shape[1] < config.MAX_PAD_LEN:
            feat = np.pad(feat, ((0, 0), (0, config.MAX_PAD_LEN - feat.shape[1])))
        else:
            feat = feat[:, :config.MAX_PAD_LEN]

        feat = (feat - np.mean(feat)) / (np.std(feat) + 1e-8)
        return feat.astype(np.float32)

    except Exception as e:
        print(f"Feature error for {path}: {e}")
        return None

# ---------------- DATA GENERATOR ----------------
class DataGen(tf.keras.utils.Sequence):
    def __init__(self, df, shuffle=True):
        self.df = df.reset_index(drop=True)
        self.shuffle = shuffle
        self.indexes = np.arange(len(self.df))
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.df) / config.BATCH_SIZE))

    def __getitem__(self, idx):
        inds = self.indexes[idx * config.BATCH_SIZE:(idx + 1) * config.BATCH_SIZE]
        X, y = [], []

        for i in inds:
            row = self.df.iloc[i]
            feat = extract_features(os.path.join(paths["train_audio"], row["filename"]))
            if feat is not None:
                X.append(feat)
                y.append(row["label"])

        if not X:
            return (
                np.zeros((0, config.N_MELS * 3, config.MAX_PAD_LEN, 1)),
                np.zeros((0,))
            )

        return np.array(X)[..., np.newaxis], np.array(y)

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)

# ---------------- MODEL ----------------
def build_model():
    inp = tf.keras.Input(shape=(config.N_MELS * 3, config.MAX_PAD_LEN, 1))

    x = tf.keras.layers.Conv2D(32, 3, padding="same", activation="relu")(inp)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling2D(2)(x)

    x = tf.keras.layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling2D(2)(x)

    x = tf.keras.layers.Conv2D(128, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)

    x = tf.keras.layers.Dense(128, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.4)(x)

    out = tf.keras.layers.Dense(1, activation="sigmoid")(x)

    model = tf.keras.Model(inp, out)
    model.compile(optimizer=tf.keras.optimizers.Adam(config.LR),
                  loss="binary_crossentropy",
                  metrics=["accuracy"])
    return model

# ---------------- TRAINING ----------------
def train_model():
    df = load_protocol()
    if df.empty:
        print("❌ No data loaded.")
        return None

    real = df[df["label"] == 1]
    spoof = df[df["label"] == 0]

    per_class = config.TRAIN_SAMPLES // 2
    df_balanced = pd.concat([
        real.sample(min(len(real), per_class), random_state=42),
        spoof.sample(min(len(spoof), per_class), random_state=42)
    ])

    df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)

    train_df, val_df = train_test_split(
        df_balanced, test_size=0.2, stratify=df_balanced["label"], random_state=42
    )

    train_gen = DataGen(train_df)
    val_gen = DataGen(val_df, shuffle=False)

    model = build_model()
    model.summary()

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            "best_audio_model.h5",       # <= SAVE BEST MODEL
            monitor="val_accuracy",
            save_best_only=True,
            mode="max",
            verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=7,
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=4,
            verbose=1
        )
    ]

    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=config.EPOCHS,
        callbacks=callbacks,
        verbose=1
    )

    print("\n🎉 Training complete!")
    print("💾 Best model saved as: best_audio_model.h5")

    return model, history


# ---------------- MAIN ----------------
if __name__ == "__main__":
    model, history = train_model()
