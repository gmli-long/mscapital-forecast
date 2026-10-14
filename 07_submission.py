"""Full training on all 71 months + test inference + submission.csv."""
import pandas as pd
import numpy as np
import lightgbm as lgb
import time
import os

from config import REPO_DIR
from common import PARAMS, NUM_ROUNDS, load_features, load_label

label = load_label()

train = load_features("train").merge(label[["sample_id", "target"]], on="sample_id", how="inner")
print("train:", train.shape)

feature_cols = [c for c in train.columns if c not in ("sample_id", "target")]

params = dict(PARAMS)

dtr = lgb.Dataset(train[feature_cols], train["target"])
t = time.time()
# 固定 155 轮 = fold5（train 0-63 / val 64-70）早停得到的最优轮数，与全量口径最接近。
# 全量 71 个月仅比 64 个月多 ~11% 样本，最优轮数应略增或持平，取 155 保守不欠拟合。
# 可通过环境变量 BEST_ITER 覆盖。
best_iter = int(os.environ.get("BEST_ITER", str(NUM_ROUNDS)))
model = lgb.train(params, dtr, num_boost_round=best_iter)
print(f"full train done in {time.time()-t:.1f}s  rounds={best_iter}")

model.save_model(str(REPO_DIR / "lgb_full.txt"))

test = load_features("test")
print("test:", test.shape)
pred = model.predict(test[feature_cols], num_iteration=best_iter)

sub = pd.DataFrame({"sample_id": test["sample_id"], "prediction": pred})
sub.to_csv(str(REPO_DIR / "submission.csv"), index=False)
print("submission:", sub.shape)
print(sub.head())
print("\nprediction stats:")
print(sub["prediction"].describe())