"""集中路径配置：数据目录与项目产物目录统一在此管理。

- 原始 feather 数据目录默认取 kagglehub 标准缓存路径，
  可用环境变量 ``KAGGLE_DATA_DIR`` 覆盖。
- 项目内产物目录（parquet / features）及结果文件路径均相对本文件
  所在的仓库根目录推导，克隆到任意位置后无需再改脚本内路径。
"""
import os
from pathlib import Path

REPO_DIR = Path(__file__).resolve().parent

DEFAULT_DATA_DIR = (
    Path.home() / ".cache" / "kagglehub" / "competitions"
    / "ms-capital-real-financial-market-forecasting"
)
DATA_DIR = Path(os.environ.get("KAGGLE_DATA_DIR", str(DEFAULT_DATA_DIR)))

PARQUET_DIR = REPO_DIR / "parquet"     # feather → parquet 中间产物
FEATURES_DIR = REPO_DIR / "features"   # 每样本聚合特征