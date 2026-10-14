"""模型/CV 脚本共享常量与数据加载函数。

集中管理此前在 06/07/08/09/10/11/12 中逐份复制的：LightGBM 超参数、
6 折 expanding walk-forward 切分、固定轮数，以及特征/标签加载器。
改参数只需动这一处。
"""
import os
import pandas as pd

from config import DATA_DIR, FEATURES_DIR

BASE = str(DATA_DIR)
FEAT = str(FEATURES_DIR)

# LightGBM 超参数（所有训练脚本共用）。
PARAMS = dict(
    objective="regression",
    metric="rmse",
    learning_rate=0.05,
    num_leaves=127,
    min_data_in_leaf=200,
    feature_fraction=0.8,
    bagging_fraction=0.8,
    bagging_freq=1,
    lambda_l2=1.0,
    verbosity=-1,
    seed=42,
)

# 固定轮数口径：155 = fold5（train 0-63 / val 64-70）早停得到的 best_iteration，
# 该折最接近全量窗口，故作为 07 全量重训与 09/10/11/12 固定轮 CV 的经验最优值。
# 全量 71 个月比 64 个月仅多 ~11% 样本，最优轮数应略增或持平，取 155 保守不欠拟合。
NUM_ROUNDS = 155

# 6 折 expanding walk-forward：train = month <= tr_hi，validate = 其后 7 个月。
FOLDS = [
    (28, [29, 30, 31, 32, 33, 34, 35]),
    (35, [36, 37, 38, 39, 40, 41, 42]),
    (42, [43, 44, 45, 46, 47, 48, 49]),
    (49, [50, 51, 52, 53, 54, 55, 56]),
    (56, [57, 58, 59, 60, 61, 62, 63]),
    (63, [64, 65, 66, 67, 68, 69, 70]),
]


def load_features(split):
    """按 sample_id 左连接三张单样本特征表（parquet 中间产物）。"""
    feats = None
    for name in ["market", "order", "transaction"]:
        f = pd.read_parquet(os.path.join(FEAT, f"{split}_{name}.parquet")).set_index("sample_id")
        feats = f if feats is None else feats.join(f, how="left")
    return feats.reset_index()


def load_label():
    return pd.read_feather(BASE + "\\train\\label.feather")


def load_X():
    """训练特征与标签合并；返回 (X, feature_cols)。"""
    label = load_label()
    X = load_features("train").merge(
        label[["sample_id", "month", "target"]], on="sample_id", how="inner"
    )
    feature_cols = [c for c in X.columns if c not in ("sample_id", "month", "target")]
    return X, feature_cols