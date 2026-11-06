# 项目架构与数据

```text
前端 ──HTTP──> backend/main.py ──> service.py
                                   ├─ data.py ──> forecasting/predict.py
                                   ├─ optimizer.py ──> Gurobi
                                   ├─ evaluation.py
                                   └─ repository.py ──> InfluxDB
                                           └─ telemetry.py
Grafana ──InfluxQL──> 同一个 InfluxDB
```

## 文件职责

| 文件/目录 | 职责 |
|---|---|
| `backend/main.py` | HTTP路由、输入错误、存储错误及应用生命周期 |
| `backend/schemas.py` | 请求/返回字段、设备单位、SOC范围与输入校验 |
| `backend/service.py` | 新建方案、滚动执行、暂停、恢复及后台线程 |
| `backend/data.py` | CSV场景、训练模型输入、历史缩放、滚动预测修正 |
| `backend/optimizer.py` | Gurobi MILP，分阶段优化软目标、削峰和费用 |
| `backend/evaluation.py` | 用执行前的预测计算MAE/RMSE，避免用已修正观测值评分 |
| `backend/database.py` | Docker InfluxDB连接与初始化；不清空数据库 |
| `backend/repository.py` | 快照、版本检查、重试和时序数据发布顺序 |
| `backend/telemetry.py` | 把仿真状态映射成15分钟时序点，不参与求解 |
| `frontend/app.py` | 操作表单、API调用、曲线、评估、JSON导出、Grafana链接 |
| `frontend/theme.py` | 页面样式 |
| `forecasting/predict.py` | 加载缓存模型、校验输入、执行推理；无数据库副作用 |
| `forecasting/train_pv.py` | 光伏LSTM-Transformer训练，保存最佳权重与训练图 |
| `forecasting/train_load.py` | 负荷LSTM训练，保存最佳权重与训练图 |
| `deploy/grafana/dashboards/ems.json` | Grafana图表、查询、方案/发布版本变量 |
| `deploy/grafana/provisioning/` | 自动创建数据源与加载看板 |
| `scripts/ems.ps1` | Windows下启动WSL中的Docker，使用同一个Compose文件 |
| `scripts/verify_stack.py` | 创建验证方案，检查API、InfluxDB和Grafana查询结果一致 |
| `tests/` | 无服务单元/回归测试，存储采用受控测试客户端 |

## 执行规则

1. 读取预测起点以前的历史数据，生成全天96点预测与日前计划。
2. 每步只揭晓当前时段的光伏与负荷，从上一步SOC重新优化剩余时域。
3. 仅执行第一步，计算SOC、电量与费用；保存修订计划和执行历史。
4. 手动请求携带 `expected_step`，重复点击或请求重试不会重复推进。
5. 暂停、重启或自动线程异常均不会自动继续执行；恢复后需要用户继续。

功率单位kW，电量kWh，电价元/kWh，SOC为0到1。
日前计划保持不变；最新滚动计划只包含剩余窗口；完整修订历史继续保存在快照中。
模型预测和缩放只读取历史；未执行的实际数据不会写到时序表，也不会通过API泄露。

## 原生时序结构

`ems_dispatch`：

- tags：`simulation_id`、`generation`、`phase`。
- phase：`day_ahead`（日前计划）、`rolling`（最新滚动计划）、`executed`（已执行结果）。
- time：方案创建日UTC零点 + `slot × 15分钟`。
- fields：`slot`、`issued_step`、`pv_kw`、`load_kw`、`charge_kw`、`discharge_kw`、
  `import_kw`、`export_kw`、`curtail_kw`、`soc`、`buy_price`、`sell_price`、
  `cost_yuan`、`target_excess_kw`、`battery_kw`、`grid_kw`、`grid_mode`、`battery_mode`。
- `battery_kw = 放电 - 充电`，`grid_kw = 购电 - 售电`；SOC是当前时段结束值。
- `issued_step` 标记滚动计划的起点；日前和已执行结果使用 -1。

`ems_metrics` 保存该版本的累计指标。`ems_simulations` 在固定创建时间保存一个快照点，
其 `generation` 指向完整的时序版本。快照中的JSON压缩并分块，保证恢复和全部修订历史不丢失。

查询示例（先从快照取得 generation）：

```sql
SELECT "generation" FROM "ems_simulations"
WHERE "simulation_id" = '方案ID';

SELECT "pv_kw", "load_kw", "soc" FROM "ems_dispatch"
WHERE "simulation_id" = '方案ID'
  AND "generation" = '上一步得到的版本'
  AND "phase" = 'executed'
  AND time >= '2026-10-20T00:00:00Z'
  AND time < '2026-10-21T00:00:00Z';
```

## 写入失败与恢复

InfluxDB不提供多数据点事务，因此不把批量写入当成原子事务：

1. 从完整状态计算确定性的generation，先写这一版本的所有时序点。
2. 全部确认成功后，再写快照及发布指针。
3. 部分写入失败时，旧快照与旧发布指针保持有效；Grafana只查询该指针对应的数据。
4. 短暂网络/服务错误最多重试3次，使用相同measurement/tags/time，重复写入覆盖同一点。
5. 若快照已提交但响应丢失，回读发布指针确认成功，避免误报或重复推进。
6. 服务启动时，为缺少时序数据的已有快照回填一次，保持原版本和执行进度。

旧发布版本和失败写入的孤立版本不参与当前查询；它们仍占用磁盘。
当前按本机研究规模保留所有版本，不会自动清理历史；大规模运行前需增加按已发布版本的清理策略。
所有统计查询都必须同时限定方案、generation和phase，避免跨版本重复统计。

版本保护依赖进程内写锁，**只能运行一个API worker和一个API实例**。
当前还没有现场遥测、账号权限、分布式锁或硬件控制，不作为生产现场控制系统。

参考：[InfluxDB重复点与字段规则](https://docs.influxdata.com/influxdb/v1/write_protocols/line_protocol_reference/)、
[Grafana数据源与看板配置](https://grafana.com/docs/grafana/latest/administration/provisioning/)。
