"""Full training on all 71 months + test inference + submission.csv."""
import pandas as pd
import numpy as np
import lightgbm as lgb
import time
import os

from config import DATA_DIR, FEATURES_DIR, REPO_DIR

BASE = str(DATA_DIR)
FEAT = str(FEATURES_DIR)

label = pd.read_feather(BASE + "\\train\\label.feather")


def load_features(split):
    feats = None
    for name in ["market", "order", "transaction"]:
        f = pd.read_parquet(os.path.join(FEAT, f"{split}_{name}.parquet")).set_index("sample_id")
        feats = f if feats is None else feats.join(f, how="left")
    return feats.reset_index()


train = load_features("train").merge(label[["sample_id", "target"]], on="sample_id", how="inner")
print("train:", train.shape)

feature_cols = [c for c in train.columns if c not in ("sample_id", "target")]

params = dict(
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

dtr = lgb.Dataset(train[feature_cols], train["target"])
t = time.time()
# use the best_iteration found in CV as a guide; retrain with fixed rounds ~ best*1.15 (train/cv shrink)
best_iter = int(os.environ.get("BEST_ITER", "600"))
model = lgb.train(params, dtr, num_boost_round=best_iter)
print(f"full train done in {time.time()-t:.1f}s  rounds={best_iter}")

model.save_model(str(REPO_DIR / "lgb_full.txt"))

test = load_features("test")
print("test:", test.shape)
pred = model.predict(test[feature_cols], num_iteration=model.best_iteration)

sub = pd.DataFrame({"sample_id": test["sample_id"], "prediction": pred})
sub.to_csv(str(REPO_DIR / "submission.csv"), index=False)
print("submission:", sub.shape)
print(sub.head())
print("\nprediction stats:")
print(sub["prediction"].describe())