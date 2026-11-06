# Docker 部署

项目只使用一套Docker Compose部署方式；Windows通过WSL运行同一套容器。

## 准备文件

- Docker Engine / Docker Desktop 与 Compose v2。
- `outputs/pv/best_model.pt`、`outputs/load/best_model.pt`。
- 项目根目录的 `gurobi.lic`；许可证需要适用于容器环境，WLS还需访问许可证服务。

模型只读挂载，不打包进镜像；许可证作为只读secret挂载。数据库使用现有 `influx-data` 命名数据卷，
Grafana使用 `grafana-data` 数据卷；`docker compose down` 保留数据。

## 启动和停止

Linux或Docker Desktop，在根目录运行：

```bash
docker compose up -d --build --wait
docker compose ps
docker compose logs --tail=50 api grafana
# 停止并保留数据
docker compose down
```

Windows + WSL Ubuntu 内的Docker：

```powershell
.\scripts\ems.ps1
.\scripts\ems.ps1 -Stop
```

脚本保持WSL运行、构建镜像并启动同一个Compose栈，不启动Windows原生数据库。
不要并行启动另一个API实例，共享状态只支持单个API worker。

| 服务 | 访问地址 |
|---|---|
| 前端 | `http://localhost:8501` |
| Grafana | `http://localhost:3000` |
| API | Docker内部 `api:8000` |
| InfluxDB | Docker内部 `influxdb:8086` |

Grafana自动加载 `deploy/grafana/` 中的数据源和看板。
本项目默认将 Docker Grafana 映射到宿主机3000端口。

## 配置

在项目根目录的 `.env` 中设置需要的值（该文件Git忽略）：

```dotenv
EMS_UI_PORT=8501
EMS_GRAFANA_PORT=3000
EMS_GRAFANA_URL=http://localhost:3000
EMS_GRAFANA_USER=admin
EMS_GRAFANA_PASSWORD=replace-with-your-password
```

Grafana账号在空数据卷首次启动时初始化。已有数据卷的密码不会被环境变量覆盖，
需要在Grafana中修改。未设置时本机默认 `admin/admin`。
`EMS_GRAFANA_URL` 是浏览器访问地址，不是Docker内部服务名。

默认只将前端和Grafana绑定到127.0.0.1。部署到云服务器时，通过带HTTPS和认证的反向代理访问；
数据库保持在Docker内部网络。可以使用SSH端口转发访问管理页面：

```bash
ssh -L 8501:127.0.0.1:8501 -L 3000:127.0.0.1:3000 user@server
```

若需要临时分享前端，可选用已有Cloudflare隧道：

```powershell
.\scripts\ems.ps1 -Public
```

Linux等价命令：`docker compose --profile public up -d --wait`。
隧道只转发前端，不转发Grafana；远端访问Grafana需配置自己的受保护地址，
并设置 `EMS_GRAFANA_URL`。临时域名可能变化，电脑关闭后服务停止。

## 整栈验证

以下命令创建一个验证方案，验证单步、重复请求、自动运行、暂停，以及三种曲线和累计指标。
它还会执行看板全部InfluxQL查询；验证方案最终保持暂停状态，便于人工查看。

```bash
docker compose exec -T api python - < scripts/verify_stack.py
```

Windows：

```powershell
Get-Content scripts/verify_stack.py -Raw | wsl -d Ubuntu -u root --cd $PWD --exec docker compose exec -T api python -
```

自定义Grafana账号时，为验证进程传入 `EMS_GRAFANA_USER` 和 `EMS_GRAFANA_PASSWORD`。

## 备份与恢复

在WSL或Linux终端、项目根目录运行：

```bash
mkdir -p backups
docker compose exec -T influxdb influxd backup -portable -database microgrid_ems /tmp/ems-backup
docker compose cp influxdb:/tmp/ems-backup ./backups/
```

每次备份使用新的容器内目录和主机目录，避免混用旧备份文件。
Grafana看板和数据源以代码为准；如修改了账号或组织设置，还需备份Grafana数据卷。

恢复前停止API和前端，使用已启动的InfluxDB将备份恢复到一个新的数据库名，先核对记录：

```bash
docker compose stop api frontend
docker compose cp ./backups/ems-backup influxdb:/tmp/ems-restore
docker compose exec -T influxdb influxd restore -portable -db microgrid_ems -newdb ems_restored /tmp/ems-restore
```

上例不覆盖现有库。若要切换到恢复库，需要同步修改Compose中API的 `EMS_INFLUX_DATABASE`
和Grafana的 `EMS_INFLUX_DATABASE`，然后重新创建API、前端和Grafana容器。
备份当前为手动执行，尚未配置自动备份计划。
