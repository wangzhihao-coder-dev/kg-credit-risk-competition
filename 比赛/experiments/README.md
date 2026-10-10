# 运行与复现

本目录保存参赛报告用到的全部实验配置与汇总结果。命令均从项目根目录（`比赛/` 的上一级）运行，Python 3.11：

```bash
uv venv .venv-credit --python 3.11
uv pip install --python .venv-credit/bin/python -r 比赛/experiments/requirements.lock.txt
.venv-credit/bin/python -m unittest discover -s 比赛/tests -v
```

生成报告 PDF 还需要 `reportlab`（未列入锁定文件，可用任一装有 reportlab 的 Python 运行最后一步）。

## 1. 取数（需要 WRDS 授权）

原始数据不进入 Git，按下表放到 `比赛/data/raw/wrds/` 下。每个目录都带 `manifest.json`（文件 SHA256），读取时逐一核对。

| 目录 | 内容 | 获取方式 |
| --- | --- | --- |
| `related_party_2014_2023/` | BoardEx 任职与链接、13F ≥5% 持股、Audit Analytics 诉讼与破产、Compustat 证券/公司代码 | WRDS Jupyter 上运行 `src/wrds_jupyter/02_export.py` |
| `distress_events_2012_2024/` | 延迟披露、财务重述、CRSP 月度股价、退市、CCM 链接 | WRDS Jupyter 上运行 `src/wrds_jupyter/04_export_events.py` |
| `snapshot_population_2015_2023/` | Compustat 时点年度财务（全库，季度观察日） | WRDS 网页查询，参数见目录内 `query_parameters.json` |
| `supply_chain_history_2015_2023/` | Compustat 客户—供应商披露 | WRDS 网页查询，参数见目录内 `query_parameters.json` |
| `ratings_history_2015_2023/` | S&P 主体评级历史（关联方市场信号） | WRDS 查询，SQL 见目录内 `history.sql` |
| `ciq_gvkey_history/` | Capital IQ—GVKEY 映射 | WRDS 查询，SQL 见目录内 `links.sql` |

两个 Jupyter 脚本从环境变量读取账号：先 `export WRDS_USERNAME=<你的 WRDS 账号>`。脚本把每条 SQL 写到输出目录的 `sql/`，结果哈希写到 `manifest.json`；下载并核对后解包到上表对应目录。

## 2. 建图谱面板与训练

```bash
.venv-credit/bin/python 比赛/src/build_distress_panel.py
.venv-credit/bin/python 比赛/src/run_distress_models.py --config distress_model_distress_config.json
.venv-credit/bin/python 比赛/src/run_distress_models.py --config distress_model_distress_unlisted_config.json
.venv-credit/bin/python 比赛/src/run_distress_models.py --config distress_model_bankruptcy_unlisted_config.json
.venv-credit/bin/python 比赛/src/run_distress_gnn_study.py --config distress_model_distress_unlisted_config.json
.venv-credit/bin/python 比赛/src/run_distress_gnn_study.py --config distress_model_bankruptcy_unlisted_config.json
```

- `build_distress_panel.py`：按季度建企业关联图谱（供应、人员、股东三类边）与节点特征、困境/破产标签，写入 `data/processed/distress_panel/` 与 `distress_panel_audit.json`。人员与股东边先用集合去重，数组中边的排列顺序不固定，边集合本身可复现。
- `run_distress_models.py`：报表评分卡、同特征神经网络、两阶段关联特征模型与关系型图神经网络的主比较。`distress_config` 为完整信息视角（报告图4），两个 `unlisted` 配置为非上市客户视角（报告主结果）。超参数在查看测试期之前固定，只按验证期 AP 选型。
- `run_distress_gnn_study.py`：关系消融（单一关系、关联方也只用报表信息）、评分卡等级与关系归因。

`src/` 中其余模块（`audit_rating_history`、`build_related_graph`、`global_graph`、`global_panel`、`run_global_contagion`、`rating_state`、`graph_models`、`experiment_metrics`、`build_final_report_rated`）提供上述脚本导入的数据校验、关系构造、评级状态、指标与排版函数。

## 3. 业务分析与报告

```bash
.venv-credit/bin/python 比赛/src/analyze_business_value.py
<装有 reportlab 的 python> 比赛/src/build_competition_report.py
```

`analyze_business_value.py` 读取非上市视角困境模型的测试期预测（`predictions_distress_own_unlisted.npz`），计算按季度滚动的增益曲线、名单命中率、预警提前期、“只有图谱模型能识别”的企业数、匿名传导案例，以及按实测债务计的业务价值。

业务价值口径（`business_value.json` 的 `measured_debt`）：测试期每季度按各模型风险分取前 5% 客户；其中事后陷入困境的企业按企业去重，取首次被列入名单那个季度的债务余额 DLC+DLTT，只用观察时点可见的时点年度财务。缺失值不填 0，单独计数；加元不换算，按币种分列（约占全部困境企业债务 0.4%）。随机抽查给出解析期望值。债务集中于少数大企业，另外输出按企业重抽样的 95% 区间和剔除债务最大 10 家后的差额（`concentration`）。金额是企业对全部债权人的总债务，不是单家银行的敞口，数值为 2021–2023 测试期合计。换算可规避损失时（`scenario_loss`）使用 `assumptions` 里带 `type` 的参数：LGD 45%（`regulatory`）、提前处置减损比例 20%/30%/50%（`scenario_assumption`）、复核比例 5%（`design_choice`）。

`build_competition_report.py` 只读取本目录的汇总 JSON，生成 `比赛/reports/参赛报告_经营类客户关联图谱风控模型.pdf` 与同名 Markdown，并检查正文不超过 8 页。

## 结果文件对照

| 文件 | 生成步骤 | 报告中的用途 |
| --- | --- | --- |
| `distress_panel_audit.json` | 建面板 | 样本规模、季度快照数、每季度各类关联边数 |
| `distress_distress_unlisted_results.json` | 训练（非上市，困境） | 主结果表、模型比较 |
| `distress_bankruptcy_unlisted_results.json` | 训练（非上市，破产） | 主结果表“破产 AP”列 |
| `distress_distress_results.json` | 训练（完整信息，困境） | 图4 不同信息条件对比 |
| `distress_distress_unlisted_gnn_study.json` | 消融（困境） | 图7、图9、关系归因 |
| `distress_bankruptcy_unlisted_gnn_study.json` | 消融（破产） | 破产事件发生率 |
| `business_value.json` | 业务分析 | 第四节应用验证、第五节投入产出 |
