# MS Capital 真实金融市场预测

> Kaggle 竞赛「[MS Capital – Real Financial Market Forecasting](https://www.kaggle.com/competitions/ms-capital-real-financial-market-forecasting)」的数据本地复现项目 —— 基于订单簿快照与逐笔委托/成交流的高频金融时序回归预测。**本地自评、未提交竞赛排名**。

端到端完成：数据探查 → 列式存储转换 → 流式特征工程（71 维）→ LightGBM 时序建模 → 多折时间交叉验证 → 特征稳定性分析与剪枝 → 统计显著性检验 → 提交文件生成。

---

## 1. 任务与数据

**任务**：给定每个样本（`sample_id`）在预测时刻之前的一段高频市场数据，预测未来的价格走势（回归目标 `target`，即未来收益）。

**数据规模**：约 9.5 GB（Apache Feather 格式），单表行数达数亿级（约 2 亿行），无法全量载入内存。

| 文件 | 说明 | 大小(约) |
|------|------|---------|
| `train/label.feather` | 训练标签：`sample_id, month(0–70), target` | 9.6 MB |
| `train/market.feather` | 订单簿快照（两级一档，含成交汇总） | 4.19 GB |
| `train/order.feather` | 逐笔委托流 | 1.24 GB |
| `train/transaction.feather` | 逐笔成交流 | 0.48 GB |
| `test/{market,order,transaction}.feather` | 测试集（无 label） | 3.57 GB |

**字段结构**：

- `market`：`sample_id, seconds_before_predict, bid_price_1, ask_price_1, bid_volume_1, ask_volume_1, transaction_avgprice, transaction_volume, transaction_count`
- `order`：`sample_id, seconds_before_predict, price, volume, side(0=买/1=卖), order_action(0=新增/1=撤单)`
- `transaction`：`sample_id, seconds_before_predict, price, volume, side`
- `label`：`sample_id, month, target`

> `seconds_before_predict`（记作 `sbp`）：该条记录距「预测时刻」的秒数，是构造时间加权的关键。三类流的时间跨度不同：`market` 快照覆盖 **0–600s**，而 `order` / `transaction` 逐笔流仅覆盖 **0–60s**——因此 `t_*` / `o_*` 的 `exp(-sbp/600)` 权重退化为近乎等权（`e^-0.1≈0.90` 到 `1.0`），「600s 中期」的衰减含义只对 `m_*` 成立。

**数据字典核验（`13_data_checks.py`）**：

- **单一资产/指数**：按 `month` 分组的中间价分布显示，每个月都是同一形状——主体集中在 ~1.0（月份均值 0.99–1.00 几乎不变），外加极少量（~0.5%）落在 0.5–0.9 的低价样本散落于所有月份；既无价格时间趋势，也非多资产的多价格峰，配合样本内 mid 标准差仅 ~0.0015，判为单一资产。`sample_id` 与 `month` 严格单调（spearman≈0.9999，71 个月区间零重叠），为时序口径的误差序列提供排序依据；
- **`side` 编码方向**：`0=买 / 1=卖` 经主动成交方向检验确认——将 `sbp` 取整后对齐成交与快照（对齐率 ~39%，约 40 万条），side=0 有 ~88% 成交价高于中间价、~87% 落在卖一及以上（主动吃卖一），side=1 仅 ~9%（主动吃买一）。

---

## 2. 方法与管道

```
01 探查/校验     00_inspect / 01_sample_relations / 02_peek / 13_data_checks
02 格式转换      04_convert_parquet.py   feather → parquet（行组 1M + zstd）
03 特征工程      05_build_features_duckdb.py  DuckDB 流式 SQL 聚合 → 每样本 71 维
04 建模验证      06_model.py              LightGBM + 按月时序切分
05 全量训练      07_submission.py         全 71 月训练 + 测试推理 → submission.csv
06 泛化评估      08_multifold_cv / 09_fixed_rounds_cv
07 特征精炼      10_feature_stability / 11_prune_scan
```

### 2.1 特征工程（71 维）

对 3 类数据分别按 `sample_id` 聚合，产出 `m_* / o_* / t_*` 三组特征：

- **订单簿（market）**：中间价 `mid=(bid1+ask1)/2`、买卖价差 `spread`、微价格 `microprice`、订单簿不平衡度 `imb=(bid_vol1−ask_vol1)/(bid_vol1+ask_vol1)`、深度、成交均价/量/笔数统计；
- **逐笔委托（order）**：委托笔数/成交量、VWAP、净成交量 `o_net_vol`、买卖方向不平衡 `o_ofi_ratio`、新单/撤单、近 10 秒净量；
- **逐笔成交（transaction）**：成交笔数/量、VWAP、净流向 `t_net_vol`、买卖占比、近 10 秒净量；
- **时间加权与变化**：以 `exp(-sbp/30)`、`exp(-sbp/600)` 做指数衰减加权捕捉「近短期/近中期」强度，并构造中间价斜率；用 `arg_min/arg_max(...)` 取「最近/最早」快照刻画截面状态。

### 2.2 建模

LightGBM 回归，优化 RMSE：

```
objective=regression, metric=rmse, learning_rate=0.05,
num_leaves=127, min_data_in_leaf=200, feature_fraction=0.8,
bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, seed=42
```

- **严格时间切分**：按月份切分，`0–63 月训练 / 64–70 月验证`，不做随机划分，避免时间泄露；
- **6 折 expanding walk-forward**：每折验证「未来 7 个月」，训练此前全部月份，评估泛化稳定性；
- **固定轮数复验**：以固定 155 轮（= 末折早停 best_iteration，无早停）重跑 6 折，量化早停带来的乐观偏差；
- **全量训练与提交**：沿用固定 155 轮（= 末折早停 best_iteration）在全 71 月上重训，测试推理生成 `submission.csv`。

### 2.3 特征精炼

- 跨 6 折统计每个特征的 `gain` 重要性份额（`mean_share`）、变异系数与 Top-20 出现频率；
- 按累计平均贡献 ≤ 95% 选取稳定特征，再扫描 `K = {10,15,…,71}` 定位拐点。

---

## 3. 结果

### 3.1 单折基线（64–70 月验证）

| 指标 | 模型 | 零预测基线 |
|------|------|-----------|
| RMSE | **0.003230** | 0.003258 |
| MAE  | 0.002102 | — |
| Pearson | 0.1330 | — |
| Spearman | 0.1224 | — |
| R² | 0.0168 | — |

### 3.2 6 折 walk-forward（固定 155 轮，无早停）

| fold | 验证月份 | RMSE | Pearson |
|------|---------|------|---------|
| 0 | 29–35 | 0.002450 | 0.1119 |
| 1 | 36–42 | 0.002043 | 0.1150 |
| 2 | 43–49 | 0.002099 | 0.1221 |
| 3 | 50–56 | 0.002853 | 0.1157 |
| 4 | 57–63 | 0.002237 | 0.1250 |
| 5 | 64–70 | 0.003230 | 0.1330 |
| **均值** | — | **0.002485** | **0.1208** |

- **早停偏差**：固定轮 vs 早停，RMSE 差异仅 **+0.011%**，可忽略 → 提交预测可信；
- 绝对 RMSE 随验证月份波动（0.0020~0.0032），主要源于不同月份目标波动率（regime）的差异；模型相对零预测基线的比值 **RMSE/base0 稳定在 0.991~0.994**，比绝对 RMSE 更能反映模型的跨折一致性。

### 3.3 特征剪枝

两个口径并存，含义不同：

- **拐点 40**：按跨折平均贡献排序后扫描 `K={10,…,71}`，K=40 起 RMSE 基本不再下降（0.002486），再往上边际≈0——这是「剪枝维数」的口径；
- **95% 累计贡献 61**：`cum_mean_share ≤ 95%` 规则选出的特征数，写进 `pruned_features.txt`（顺序 = `feature_stability.csv` top-61），比 40 更保守；
- 高价值特征集中于「净持仓流 / 价差-中间价波动」类，Top 特征如 `t_net_vol_10s`、`m_spread_std`、`t_netvol_w30`；长尾特征贡献趋零。

### 3.4 统计显著性检验与 seed 敏感性

合并 6 折 walk-forward 的 743,695 个验证样本（pooled 口径），检验「优于零预测基线」是否为噪声：

| 检验 | 结果 | 结论 |
|------|------|------|
| Diebold-Mariano（HAC，max_lag ∈ {0,5,10,20,40} 扫描） | z = 25.9 ~ 26.3 稳定，p ≈ 0 | 不依赖 lag 选择，显著优于零预测 |
| Moving-block Bootstrap（L=20, B=5000） | 损失差 95% CI = [8.6e-8, 1.0e-7]，不含 0 | 保留时间自相关，稳健排除噪声 |
| Pearson 相关显著性 | r = 0.1207，95% CI = [0.1185, 0.1230]，z = 104.6 | 显著但微小 |

- pooled RMSE = **0.002520**，零预测基线 RMSE = 0.002538，相对提升 **0.725%**。
- **seed 敏感性**：固定 155 轮、**20 个 seed** 单线程重训末折（64–70）以隔离线程调度噪声，`RMSE/base0` 全部落在 **[0.9913, 0.9921]**（均 < 1），「优于零预测」不依赖特定随机种子（最差 seed=1）。
- **口径说明**：大样本下 DM/p 值几乎必然显著——它只回答「提升是否恰好为 0」，不回答「是否够用」。结论是提升**统计显著非零但量级微弱**（0.725%）；是否值得实盘须另按交易成本评估。

---

## 4. 运行指南

### 环境

- Python 3.10+
- 依赖：`pip install -r requirements.txt`

### 数据准备

原始数据由 Kaggle 竞赛提供，通过 `kagglehub` 下载后默认存放于 `~/.cache/kagglehub/competitions/ms-capital-real-financial-market-forecasting`。

所有脚本的路径统一由 `config.py` 管理，无需在脚本内修改硬编码路径：

- 原始数据目录默认取上述 kagglehub 缓存路径，若存放在自定义位置，设置环境变量即可覆盖：

  ```bash
  # Windows (PowerShell)
  $env:KAGGLE_DATA_DIR = "D:\data\ms-capital"
  ```

- `parquet/`（转换中间产物）与 `features/`（聚合特征）等派生产物路径均相对仓库根目录推导，在 Windows 下克隆到任意位置后开箱即用。

### 步骤

```bash
# 1. 探查数据结构与规模
python 00_inspect.py
python 01_sample_relations.py
python 02_peek.py
python 13_data_checks.py   # 数据字典核验（单资产 / side 编码方向）

# 2. feather → parquet（列式 + zstd，供 DuckDB 高效读取）
python 04_convert_parquet.py

# 3. DuckDB 流式聚合，生成 71 维特征（输出 features/*.parquet）
python 05_build_features_duckdb.py

# 4. 基线建模 + 时序验证
python 06_model.py

# 5. 6 折 walk-forward 泛化评估
python 08_multifold_cv.py
python 09_fixed_rounds_cv.py

# 6. 特征稳定性分析与剪枝
python 10_feature_stability.py
python 11_prune_scan.py

# 7. 统计显著性检验 + seed 敏感性
python 12_significance.py

# 8. 全量训练 + 测试推理 + 生成提交
python 07_submission.py
```

---

## 5. 仓库结构

```
.
├── config.py                  # 统一路径配置（数据目录 + 产物目录）
├── common.py                  # 共享参数/切分/数据加载（06-12 共用）
├── 00_inspect.py              # 数据 schema / 行数探查
├── 01_sample_relations.py     # sample_id 关联关系
├── 02_peek.py                 # 快速抽样查看
├── 03_build_features.py       # Polars 流式聚合（早期方案；group_by 聚合触发 DuplicateError，未根因即改用 DuckDB）
├── 04_convert_parquet.py      # feather → parquet 转换
├── 05_build_features_duckdb.py# DuckDB 流式特征聚合（核心）
├── 06_model.py                # 基线模型 + 时序验证
├── 07_submission.py           # 全量训练 + 测试推理
├── 08_multifold_cv.py         # 6 折 expanding walk-forward
├── 09_fixed_rounds_cv.py      # 固定轮数去早停偏差
├── 10_feature_stability.py    # 特征重要性稳定性分析
├── 11_prune_scan.py           # Top-K 剪枝扫描
├── 12_significance.py         # 统计显著性检验 + seed 敏感性
├── 13_data_checks.py          # 数据字典核验（单资产 / side 编码方向）
├── cv_results.csv             # 早停多折结果
├── cv_fixed_results.csv       # 固定轮多折结果
├── feature_stability.csv      # 特征稳定性表
├── prune_scan.csv             # 剪枝扫描结果
├── significance_results.csv   # 显著性/seed 敏感性结果（20 个 seed）
├── significance_summary.csv   # 显著性汇总（DM z/p、block bootstrap CI、Pearson CI 全部落盘）
├── pruned_features.txt        # 精炼后的特征子集（95% 累计贡献口径，61 维）
└── submission.csv             # 最终提交文件
```

> `parquet/`（转换后的中间数据）与 `features/`（聚合特征）体积过大，已通过 `.gitignore` 排除；按上述步骤可在本地重新生成。

---

## 6. 仅作参考的后续方向

- 引入更细的订单簿二阶信息与多档深度序列（LSTM / Transformer 建模）；
- 目标口径进一步对齐（多步收益、分位回归）；
- 更强的时序正则与分布漂移检测（按月份分桶校准）。