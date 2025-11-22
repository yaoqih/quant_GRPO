# quant_GRPO

基于 [qlib](https://github.com/microsoft/qlib) 的 Transformer + GRPO T+1 日线轮仓策略实验脚手架，包含数据预处理、模型训练、策略回测三个阶段的统一配置。

## 快速上手

1. 安装依赖
   ```bash
   pip install -e qlib/  # 安装仓库自带的qlib源码
   pip install torch pandas pyyaml
   ```
2. 准备日频数据（可以在数据下载完成后执行）
   ```bash
   python qlib/scripts/get_data.py qlib_data_cn --target_dir data/qlib_1d
   ```
3. 启动训练（默认使用 `pipelines/transformer_grpo/config_cn_t1.yaml`）
   ```bash
   python -m pipelines.transformer_grpo.trainer \
     --config pipelines/transformer_grpo/config_cn_t1.yaml
   ```
   - 训练日志、指标与模型 `state_dict` 将保存到 `runs/transformer_grpo/<timestamp>/`。
4. 使用已保存的 checkpoint 回测指定时间段
   ```bash
   python -m pipelines.transformer_grpo.c \
     --config pipelines/transformer_grpo/config_cn_t1.yaml \
     --checkpoint runs/transformer_grpo/<timestamp>/best.pt \
     --segment test --out_dir runs/eval/test
   ```

> 数据仍在下载时，可以先修改配置、熟悉 workflow；训练脚本会在检测到 segment 为空时直接报错，避免误跑。

## Workflow 结构

- `pipelines/transformer_grpo/data_pipeline.py`
  - 通过 `DailyBatchFactory` 把 qlib `DataHandler` 输出的多层索引表转换成“按交易日切片”的 `DailyBatch`，并完成 T+1 轮动策略所需的特征+收益标签对齐。
  - `DailyBatchDataset` 暴露 `calendar`、`feature_dim` 等元信息，供 PyTorch `DataLoader` 与回测模块共享。

- `pipelines/transformer_grpo/model.py`
  - `TransformerPolicy`：跨截面的 Transformer Encoder，将同一交易日所有股票特征视作 token，产生 action logits 与 value 估计。
  - `act` 方法支持温度采样与贪婪决策，方便在线交易或随机策略对比。

- `pipelines/transformer_grpo/trainer.py`
  - `GRPOTrainer` 将 GRPO（群体相对优势）目标与 value baseline、entropy bonus 组合，形成强化学习式的损失函数。
  - 自动创建工作目录、保存配置、周期性评估，并在验证集上基于 Sharpe（默认可在配置中修改）挑选最佳模型。

- `pipelines/transformer_grpo/backtest.py` & `run_backtest.py`
  - `run_policy_on_dataset` 顺序遍历 `DailyBatch`，模拟“收盘决策->次日调仓”策略，输出逐日交易记录。
  - `run_backtest` 汇总收益率、净值曲线，`compute_performance` 计算累计收益、年化、Sharpe、最大回撤、胜率等指标，并以 CSV/JSON 的形式落盘。

## 配置说明

`pipelines/transformer_grpo/config_cn_t1.yaml` 覆盖了整个流水线所需参数，主要分为：

- `qlib`: 指定数据目录、地区等；默认指向 `./data/qlib_1d`。
- `data`: 
  - `handler` 部分完全复用 qlib 的 `Alpha360` 配置，可以按需替换为自定义 handler（例如加入更多 Alpha 因子、限制股票池等）。
  - `segments` 定义训练/验证/测试区间，均可随意拆分。
  - `label` 表达式默认是 “次日开盘/前日开盘 - 1”，符合 T+1 换仓逻辑；若需要用次日收盘收益或包含手续费，只要修改 Label 表达式即可。pipelines 会读取 handler 配置里的 `label` 列表尝试匹配返回的列名，若无法匹配则退回到首列，也可在 `label_name` 中手动指定。
  - `min_instruments` / `max_instruments` 控制每日候选股票上限，避免在超大股票池上训练导致显存不足。
- `model`: Transformer 结构超参。
- `training`: GRPO 训练参数（学习率、熵系数、温度、监控指标等），以及输出目录。
- `backtest`: 提供默认的回测段落及交易成本假设，`run_backtest.py` 会读取这里的风险自由率等信息。

## 下一步可以做什么

- 在数据下载完成后，更新 `config_cn_t1.yaml` 中的 `segments` 时间范围和股票池，确保覆盖目标训练集。
- 如果需要引入更复杂的行情特征（分钟级别、财务因子等），只需在 qlib handler 中扩展 feature/processor，然后复用 `DailyBatchFactory` 即可。
- 结合交易端需求，可以在 `run_backtest.py` 中增加生成实盘下单所需信号、或接入已有的订单模拟器。
