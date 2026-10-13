# EMS

光储微电网能量管理系统：光伏与负荷预测、储能调度优化及 InfluxDB 数据记录。

## 项目结构

```text
EMS/
├── main.py                       # 主程序：预测、调度优化、数据库记录
├── data/                         # 原始表格数据，统一为 UTF-8 BOM CSV
│   ├── gansu_pv.csv               # 甘肃光伏与气象数据
│   ├── quanzhou_load.csv          # 泉州负荷与气象数据
│   └── electricity_prices.csv    # 分时购电、售电价格
├── training/                     # 两个独立训练程序
│   ├── train_pv.py                # LSTM-Transformer 光伏预测
│   └── train_load.py              # LSTM 负荷预测
├── test.py                       # Gurobi 求解器检查
├── ems_dashboard.json             # Grafana 能量管理看板配置
├── gurobi.lic                     # 本地求解器许可
└── README.md
```

## 运行

配置主程序中的 InfluxDB 和本机 Gurobi 后，在项目根目录运行：

```bash
python main.py
```

两个训练程序分别运行，会显示训练指标和图表：

```bash
python training/train_pv.py
python training/train_load.py
```

依赖：numpy、pandas、torch、matplotlib、scikit-learn、scipy、pyomo、gurobipy、influxdb。

## 数据与现有行为

- 三个 CSV 保留原 Excel 的有效数据、列名和行序；原工作簿均只有一个工作表。负荷表末尾的空白格式行不导出，与原 `read_excel` 的读取结果一致。
- 光伏 CSV 前三列是时间信息，后18列与原 NPY 对应。光伏训练和主程序继续使用前48,000行（500天，每天96点），沿用原来的17个输入特征和功率列。
- 负荷时间列在读取后转为日期时间，原有预处理保持不变。
- 主程序保持原有历史平均值加随机波动的预测方式；两个训练程序仍独立运行，未接入主程序。
- 数据路径以程序文件位置为基准，不依赖启动时的工作目录。
