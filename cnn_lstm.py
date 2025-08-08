# train_libras_bilstm_v3.py
# -*- coding: utf-8 -*-
"""
LIBRAS landmarks -> BiLSTM com anti-overfitting + alinhamento de features
- feature_mode: 'absolute' ou 'wrist_centered'  (use 'wrist_centered' p/ bater com o infer_live.py)
- tf.data com augmentations: jitter, rotate, scale, time-crop, time-mask, temporal-dropout
- AdamW + label smoothing + ReduceLROnPlateau + EarlyStopping
- Salva: best_model.keras, libras_actions.npy, results_v3/* (plots, conf mats, norm_stats.json)
"""

import os, json, random, sys
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from glob import glob
from collections import defaultdict
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from sklearn.preprocessing import label_binarize
from sklearn.metrics import precision_recall_curve, average_precision_score

# --------------------- CONFIG ---------------------
DATA_DIR         = "libras_data"
RESULTS_DIR      = "results_v3"
MODEL_OUT        = "best_model.keras"
ACTIONS_OUT      = "libras_actions.npy"
SEED             = 42
TEST_SIZE        = 0.2
BATCH_SIZE       = 32
EPOCHS           = 120
INIT_LR          = 3e-4
WEIGHT_DECAY     = 1e-4
LABEL_SMOOTH     = 0.05
FEATURE_MODE     = "wrist_centered"   # 'absolute' ou 'wrist_centered'
USE_GROUP_SPLIT  = True               # tenta separar por 'grupo' (antes do 1º "_")
ROT_DEG          = 8.0                # rotação máxima (graus)
SCALE_MIN, SCALE_MAX = 0.9, 1.1       # escala espacial
JITTER_STD       = 0.01               # ruído gaussiano
TIME_MASK_RATIO  = 0.1                # fração de timesteps mascarados
TEMP_DROPOUT_P   = 0.05               # prob. de zerar um timestep
SAVE_HISTORY_PNG = True

os.makedirs(RESULTS_DIR, exist_ok=True)
np.random.seed(SEED); tf.random.set_seed(SEED); random.seed(SEED)

# ------------------ LOAD DATA --------------------
def list_classes(data_dir):
    acts = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    acts.sort()
    return np.array(acts)

def load_sequences(data_dir):
    actions = list_classes(data_dir)
    X, y, meta = [], [], []
    for ci, act in enumerate(actions):
        for fp in glob(os.path.join(data_dir, act, "*.npy")):
            try:
                arr = np.load(fp)  # (T, F)
                X.append(arr)
                y.append(ci)
                # grupo: prefixo do nome do arquivo antes do primeiro "_"
                base = os.path.splitext(os.path.basename(fp))[0]
                group = base.split("_")[0] if "_" in base else base
                meta.append({"path": fp, "class": act, "group": group})
            except Exception as e:
                print(f"[WARN] {fp}: {e}")
    X = np.array(X, dtype=object)  # mantemos variável T
    y = np.array(y, dtype=int)
    return X, y, actions, meta

def pad_or_crop_to_T(x, T):
    # pad/crop temporal no centro
    t = x.shape[0]
    if t == T:
        return x
    if t > T:
        start = (t - T) // 2
        return x[start:start+T]
    # pad
    pad_len = T - t
    pad = np.zeros((pad_len, x.shape[1]), dtype=x.dtype)
    return np.concatenate([x, pad], axis=0)

def infer_TF_from_first(X):
    # infere T e F do primeiro exemplo
    x0 = X[0]
    return x0.shape[0], x0.shape[1]

# ---------- FEATURE TRANSFORMS (CPU/Numpy) ----------
def to_wrist_centered(x):
    """
    x: (T, F). F pode ser 63 (21*3) ou 126 (2*21*3). Se inesperado, retorna x.
    Wrist-centered por mão: subtrai o landmark 0 (pulso) de cada mão.
    """
    F = x.shape[1]
    if F not in (63, 126):
        return x  # desconhecido
    if F == 63:
        pts = x.reshape(x.shape[0], 21, 3)
        wrist = pts[:, 0:1, :]             # (T,1,3)
        pts = pts - wrist
        return pts.reshape(x.shape[0], -1)
    else:
        pts = x.reshape(x.shape[0], 42, 3)
        # assume mão direita 0..20 e esquerda 21..41
        wrist_r = pts[:, 0:1, :]
        wrist_l = pts[:, 21:22, :]
        pts[:, 0:21, :]  = pts[:, 0:21, :]  - wrist_r
        pts[:, 21:42, :] = pts[:, 21:42, :] - wrist_l
        return pts.reshape(x.shape[0], -1)

def apply_feature_mode(x, mode):
    if mode == "wrist_centered":
        return to_wrist_centered(x)
    return x  # 'absolute'

# ----------------- SPLIT ------------------------
def group_stratified_split(X, y, meta, test_size=0.2, seed=42):
    """
    Mantém proporção por classe e tenta separar grupos (pessoa/sessão).
    Estratégia simples: sorteia grupos até atingir ~test_size.
    """
    # agrupa por (classe, grupo)
    by_group = defaultdict(list)
    for i, m in enumerate(meta):
        key = (y[i], m["group"])
        by_group[key].append(i)

    # coleta grupos por classe
    groups_by_class = defaultdict(list)
    for (cls, grp), idxs in by_group.items():
        groups_by_class[cls].append((grp, idxs))

    rng = np.random.RandomState(seed)
    test_idx = []
    train_idx = []

    for cls, grp_list in groups_by_class.items():
        rng.shuffle(grp_list)
        cls_all = sum((idxs for (_, idxs) in grp_list), [])
        target = int(np.ceil(len(cls_all) * test_size))

        picked = []
        count = 0
        for grp, idxs in grp_list:
            if count >= target:
                break
            picked.extend(idxs); count += len(idxs)

        cls_test = set(picked)
        for _, idxs in grp_list:
            for i in idxs:
                (test_idx if i in cls_test else train_idx).append(i)

    return np.array(train_idx, int), np.array(test_idx, int)

# ------------- tf.data + AUGMENT -----------------
def make_dataset(X, y, T_fixed, mu, sd, batch, training, feature_mode):
    """Cria tf.data a partir de listas de np.arrays (durações variáveis)."""

    # prepara arrays já pad/crop + feature_mode
    X2 = []
    for xi in X:
        xi = apply_feature_mode(xi, feature_mode)
        xi = pad_or_crop_to_T(xi, T_fixed)
        X2.append(xi.astype(np.float32))
    X2 = np.stack(X2, axis=0)  # (N, T, F)
    y2 = tf.keras.utils.to_categorical(y).astype(np.float32)

    # normalização (z-score) com stats do treino
    X2 = (X2 - mu) / (sd + 1e-8)

    ds = tf.data.Dataset.from_tensor_slices((X2, y2))
    if training:
        ds = ds.shuffle(len(X2), seed=SEED, reshuffle_each_iteration=True)

        def aug_fn(x, y):
            # x: (T, F), assumimos F múltiplo de 3
            T = tf.shape(x)[0]; F = tf.shape(x)[1]

            # reshape para (T, P, 3)
            P = F // 3
            x3 = tf.reshape(x, (T, P, 3))

            # jitter gaussian
            x3 = x3 + tf.random.normal(tf.shape(x3), stddev=JITTER_STD, dtype=tf.float32)

            # rotate (x,y)
            theta = tf.random.uniform([], minval=-np.deg2rad(ROT_DEG), maxval=np.deg2rad(ROT_DEG))
            c, s = tf.cos(theta), tf.sin(theta)
            rot = tf.stack([[c, -s],[s, c]])  # 2x2
            xy = x3[..., :2]                 # (T,P,2)
            xy = tf.reshape(xy, (-1, 2)) @ rot
            xy = tf.reshape(xy, (T, P, 2))
            x3 = tf.concat([xy, x3[..., 2:3]], axis=-1)

            # scale
            scale = tf.random.uniform([], SCALE_MIN, SCALE_MAX)
            x3 = x3 * scale

            # temporal dropout
            mask_keep = tf.cast(tf.random.uniform((T,)) > TEMP_DROPOUT_P, tf.float32)
            mask_keep = tf.reshape(mask_keep, (T,1,1))
            x3 = x3 * mask_keep

            # time mask (zera bloco contíguo)
            L = tf.cast(tf.round(TIME_MASK_RATIO * tf.cast(T, tf.float32)), tf.int32)
            L = tf.maximum(L, 0)
            def time_mask_block(x3):
                start = tf.random.uniform([], minval=0, maxval=tf.maximum(T - L, 1), dtype=tf.int32)
                mask = tf.concat([tf.zeros((start,1,1)), tf.ones((L,1,1)), tf.zeros((T-start-L,1,1))], axis=0)
                return x3 * (1.0 - mask)
            x3 = tf.cond(L > 0, lambda: time_mask_block(x3), lambda: x3)

            x_aug = tf.reshape(x3, (T, F))
            return x_aug, y

        ds = ds.map(aug_fn, num_parallel_calls=tf.data.AUTOTUNE)

    ds = ds.batch(batch).prefetch(tf.data.AUTOTUNE)
    return ds

# ----------------- MODEL ------------------------
def build_model(input_shape, n_classes):
    from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
    try:
        opt = tf.keras.optimizers.AdamW(learning_rate=INIT_LR, weight_decay=WEIGHT_DECAY)
    except Exception:
        from tensorflow.keras.optimizers import Adam
        opt = Adam(learning_rate=INIT_LR)
        print("[WARN] AdamW indisponível; usando Adam.")

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=input_shape),
        Bidirectional(LSTM(128, return_sequences=True, dropout=0.3)),
        Bidirectional(LSTM(64,  return_sequences=False, dropout=0.3)),
        Dense(128, activation="relu"),
        Dropout(0.4),
        Dense(n_classes, activation="softmax")
    ])

    loss = tf.keras.losses.CategoricalCrossentropy(label_smoothing=LABEL_SMOOTH)
    model.compile(optimizer=opt, loss=loss, metrics=["categorical_accuracy"])
    model.summary()
    return model

# ----------------- PLOTS ------------------------
def plot_training(history, outdir):
    # loss
    plt.figure(figsize=(12,6))
    plt.plot(history.history["loss"], label="train")
    plt.plot(history.history["val_loss"], label="val")
    plt.title("Loss")
    plt.legend(); plt.grid(True); plt.tight_layout()
    plt.savefig(os.path.join(outdir, "loss.png")); plt.close()
    # acc
    plt.figure(figsize=(12,6))
    plt.plot(history.history["categorical_accuracy"], label="train")
    plt.plot(history.history["val_categorical_accuracy"], label="val")
    plt.title("Accuracy")
    plt.legend(); plt.grid(True); plt.tight_layout()
    plt.savefig(os.path.join(outdir, "accuracy.png")); plt.close()

def eval_and_plots(model, X_test, y_test, actions, outdir):
    # probs
    y_prob = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_prob, axis=1)
    y_true = np.argmax(y_test, axis=1)

    print("\n=== Classification Report ===")
    print(classification_report(y_true, y_pred, target_names=actions, digits=3))

    # conf mat abs
    cm = confusion_matrix(y_true, y_pred, labels=range(len(actions)))
    fig, ax = plt.subplots(figsize=(8,6))
    ConfusionMatrixDisplay(cm, display_labels=actions).plot(ax=ax, xticks_rotation=45, colorbar=False)
    ax.set_title("Matriz de confusão (abs)")
    plt.tight_layout(); plt.savefig(os.path.join(outdir, "confusion_abs.png")); plt.close()

    # conf mat norm
    cmn = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(8,6))
    ConfusionMatrixDisplay(cmn, display_labels=actions).plot(ax=ax, xticks_rotation=45, colorbar=False, values_format=".2f")
    ax.set_title("Matriz de confusão (norm)")
    plt.tight_layout(); plt.savefig(os.path.join(outdir, "confusion_norm.png")); plt.close()

    # PR macro
    Yb = label_binarize(y_true, classes=range(len(actions)))
    ap_macro = average_precision_score(Yb, y_prob, average="macro")
    p, r, _ = precision_recall_curve(Yb.ravel(), y_prob.ravel())
    plt.figure(figsize=(7,5))
    plt.plot(r, p); plt.title(f"Precision-Recall (AP macro={ap_macro:.3f})")
    plt.xlabel("Recall"); plt.ylabel("Precisão"); plt.grid(True); plt.tight_layout()
    plt.savefig(os.path.join(outdir, "pr_macro.png")); plt.close()

# ----------------- MAIN ------------------------
def main():
    X_raw, y, actions, meta = load_sequences(DATA_DIR)
    if len(X_raw) == 0:
        raise RuntimeError("Sem dados.")

    # T,F do primeiro exemplo
    T0, F = infer_TF_from_first(X_raw)

    # split (com ou sem grupos)
    if USE_GROUP_SPLIT:
        tr_idx, te_idx = group_stratified_split(X_raw, y, meta, test_size=TEST_SIZE, seed=SEED)
    else:
        tr_idx, te_idx = train_test_split(np.arange(len(X_raw)), test_size=TEST_SIZE, random_state=SEED, stratify=y)

    X_train_list = [X_raw[i] for i in tr_idx]
    X_test_list  = [X_raw[i] for i in te_idx]
    y_train = y[tr_idx]; y_test = y[te_idx]

    # fixa T = mediana do conjunto (ou T0)
    T_fixed = int(np.median([xi.shape[0] for xi in X_raw]))
    if T_fixed < 16: T_fixed = max(T0, 16)  # garante janela mínima útil

    # aplica feature_mode, pad/crop e calcula stats (NO TREINO)
    X_tmp = []
    for xi in X_train_list:
        xi = apply_feature_mode(xi, FEATURE_MODE)
        xi = pad_or_crop_to_T(xi, T_fixed)
        X_tmp.append(xi)
    X_tmp = np.stack(X_tmp, axis=0).astype(np.float32)  # (N,T,F)

    mu = X_tmp.mean(axis=(0,1), keepdims=True)
    sd = X_tmp.std(axis=(0,1), keepdims=True) + 1e-8

    # salva norm_stats.json para inferência ao vivo
    norm_stats = {"mu": mu.squeeze().tolist(), "sd": sd.squeeze().tolist(), "feature_mode": FEATURE_MODE, "T": T_fixed, "F": int(X_tmp.shape[2])}
    with open(os.path.join(RESULTS_DIR, "norm_stats.json"), "w", encoding="utf-8") as f:
        json.dump(norm_stats, f, indent=2)
    print("[OK] norm_stats.json salvo.")

    # tf.data
    ds_train = make_dataset(X_train_list, y_train, T_fixed, mu, sd, BATCH_SIZE, training=True,  feature_mode=FEATURE_MODE)
    ds_test  = make_dataset(X_test_list,  y_test,  T_fixed, mu, sd, BATCH_SIZE, training=False, feature_mode=FEATURE_MODE)

    # modelo
    model = build_model((T_fixed, X_tmp.shape[2]), len(actions))

    cbs = [
        tf.keras.callbacks.ModelCheckpoint(MODEL_OUT, monitor="val_categorical_accuracy", mode="max", save_best_only=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6, verbose=1),
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True),
    ]

    hist = model.fit(ds_train, validation_data=ds_test, epochs=EPOCHS, callbacks=cbs)

    if SAVE_HISTORY_PNG:
        plot_training(hist, RESULTS_DIR)

    # avaliação final (materializa X_test para métricas detalhadas)
    X_test_np, y_test_np = next(iter(ds_test.unbatch().batch(10_000)))
    eval_and_plots(model, X_test_np.numpy(), y_test_np.numpy(), actions, RESULTS_DIR)

    # salvar classes
    np.save(ACTIONS_OUT, actions)
    print(f"[OK] Modelo: {MODEL_OUT}")
    print(f"[OK] Classes: {ACTIONS_OUT}")
    print(f"[INFO] Use {os.path.join(RESULTS_DIR, 'norm_stats.json')} no script ao vivo e mantenha FEATURE_MODE='{FEATURE_MODE}'.")
    print(f"[INFO] T usado: {norm_stats['T']}  | F: {norm_stats['F']}  | feature_mode: {FEATURE_MODE}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[FATAL] {e}")
        sys.exit(1)
