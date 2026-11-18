# EDM-IR 实验日志中心

本地 Elasticsearch + Logstash + Kibana，版本统一固定 9.5.6，使用官方 ARM64 镜像。可按实验 ID 接入单卡、多卡及其他项目；本系统只记录操作，不创建云资源、不自动启动实验。EDM-IR 原采集包保持封存。

入口：http://127.0.0.1:15601 ，账号与密码在 `.local/LOGIN.txt`（本地文件，不进 Git）。仪表盘为 **EDM-IR 实验操作中心**，含操作时间线、失败与未完成、回收与实例状态、命令与输出。Discover 支持 KQL、时间范围与实验 ID 筛选。默认显示最近 30 天，时区 Asia/Singapore。

## 启停

在本目录执行 `./ctl.sh up`、`./ctl.sh status`、`./ctl.sh stop`。`stop` 保留索引、Logstash 游标和事件文件；不提供删除数据卷的快捷命令。Colima 使用独立配置 `edmir-logs`，6 vCPU、10 GiB RAM、60 GiB 稀疏数据盘；没有改动其他 Docker context。只挂载本目录。内存上限 ES 4 GiB（heap 2 GiB）、Kibana 2 GiB、Logstash 2 GiB（heap 768 MiB）。端口仅绑定本机：Kibana 15601、ES 19200。Elasticsearch 使用 Basic 许可，不启用付费试用。

依赖通过 Homebrew 安装 `colima docker docker-compose`。Docker CLI 与 Docker Engine 均实际使用；Colima 提供 macOS 下运行 Linux 容器的虚拟机。组件数据保存在 Docker named volumes，日志源文件在 `.local/events/`。VM 内的 `vm.max_map_count` 由启动脚本设置。

## 后续操作如何入库

```sh
python3 ops.py run --run 本轮实验ID --stage prepare --timeout 300 -- python3 -u your_preparation.py
python3 ops.py record --run 本轮实验ID --action platform_operation --outcome success \
  --message '已核验平台实例状态为 deleted' --source /绝对路径/平台回执.json
```

`run` 以参数数组执行命令，不隐式拼接 shell。记录开始、stdout/stderr、结束、退出码、耗时；超时终止命令进程组，失败和中断均有终态。只在实际获准执行实验时包装其启动命令，暂停期间不得借日志测试运行 GPU 任务。不要逐 kernel 或逐计时样本同步写日志，以免干扰测量；将采集器外层输出作为操作日志，原始性能样本继续写入原结果目录。

仅通过 `ops.py run` 执行的命令会自动记入日志。它不会监控所有终端、Codex 工具调用或浏览器按钮；平台操作需要 `record` 写入已观察的结果和回执路径。写一条“已删除”日志不会实际删除实例，也不能替代平台状态核验。当前没有接入云账户 API 或自动拉账单；费率、预算、停费状态来自明确标注的本地回执。

远端采集时可将这个无额外依赖的 `ops.py` 放在服务器，用 `--spool` 指定远端事件目录；经 SSH/rsync 增量回传到本地 `.local/events/` 下具有独立名称的 NDJSON 文件，由 Logstash 读取。不同主机不要覆盖同一个事件文件。不开放公网 Logstash 接口。

## 单卡实验日志接入（2026-11-13）

已配置来源 `thesis-a100-smoke`：读取远端 `/workspace/thesis-smoke/results/*/run.log`，每 15 秒通过已有 SSH 密钥增量拉取。不会在单卡服务器安装 ELK，也不会启动、重启或停止实验。连接参数保存在本机 `.local/remotes/thesis-a100-smoke.json`，不进 Git；更换服务器后需要更新地址、端口并核验新主机身份。

```sh
cd /Users/neo/code/llm-serving/ops/log-center
./ctl.sh up                                    # 启动 ELK，并恢复已配置日志采集器
python3 remote.py status thesis-a100-smoke      # 查看 SSH 连接、文件偏移与回传数量
./ctl.sh remote-stop                           # 只停日志回传，ELK 保持可浏览
./ctl.sh remote-start                          # 从检查点继续回传
./ctl.sh stop                                  # 停回传、三个服务和本地 Colima
```

Kibana 的 Discover/操作时间线使用筛选 `experiment.id: thesis-smoke*`，或 `event.action: remote_log`。每个 results 子目录形成独立实验 ID，例如 `thesis-smoke-20261113T054057Z`。可将时间刷新设为 10 秒；实际展示延迟还包括约 15 秒的 SSH 拉取周期和索引刷新。

本次接入现有任务的日志文件，不能补造此前命令退出码，也不会自动读取其他目录、所有终端或 GPU 利用率指标。日志中的时间文字原样保留，索引时间明确标注为接收时间，`evidence.timestamp_basis` 为 `received_at_original_line_time_unparsed`。SSH 中断记为日志传输故障，不代表实验失败。

读取位置和私钥块脱敏状态持久化于 `.local/remotes/<名称>/state.json`；文件 inode 改变或长度小于已读位置时记录轮换，再读新文件。正在写入、尚未换行的末尾内容会等下一轮补齐。远端同一文件如果在两次轮询间原地截断又增长到超过旧偏移，无法仅靠 inode 和长度识别；实验应使用新的 results 子目录，避免覆盖旧日志。Mac 睡眠或日志中心停止期间，远端任务照常运行；原日志保留时，下次启动可补入。

账号为 `edmir`，密码查看 `.local/LOGIN.txt`。停止日志采集只影响本机观察，不停止云端计费。

## 历史导入与可靠性

```sh
python3 ops.py import-w1 --session ../../experiments/2026-11-11-shadeform-w1
```

导入本次八卡采集的阶段记录、数值失败、会话输出、平台操作回执及 129 个文件的校验结果。重复导入同一内容不增加事件；检测到文件内容改变会产生新的失败记录。事件 ID 决定 ES 文档 ID，重复投递覆盖同一事件，不重复计算。Logstash 保留 sincedb、512 MiB 持久队列和拒收文件；不自动删除历史索引或原始日志。

每条事件包含 `evidence.kind`、`timestamp_basis`、源文件路径和 SHA256。进程起点来自原始 stdout 文件名中的时间戳；没有实际时间的历史行使用导入时间并明确标记，不冒充真实执行时刻。历史平台操作是人工回执，与采集器原生记录分开。恢复清单校验时间是本地实际校验时间，不是实验执行时间。

常用筛选：

- `experiment.id: shadeform-w1-20261111`
- `event.outcome: failure`
- `event.action: numerical_check_failed`
- `event.action: command_finished and process.exit_code > 0`
- `evidence.timestamp_basis: import_time_actual_line_time_unknown`

密码、常见 token/API key/Authorization 字段及私钥块在写入事件前脱敏；不采集环境变量、不扫描 SSH 或 Downloads 目录。脱敏不是任意秘密格式的保证，命令应从文件或受控环境读取凭据，避免主动输出凭据。完整原始实验日志仍留在原目录，日志中心只作检索与操作追踪。

## 验证与备份

`python3 -m unittest discover -s tests -v` 验证真实退出码、进程超时、启动失败终态、时间语义、敏感字段处理、重复导入及篡改检测。`python3 verify.py --check-restart` 会重启本地 Logstash，验证停止期间的日志补入、重复投递去重、全部事件 ID 入库、认证与只读权限，以及 Kibana 看板及其引用对象；不会联系云平台。它还将当前看板导出到本地，供备份使用。

`./ctl.sh backup` 保存事件源和最近导出的 Kibana 对象，日志索引可由事件源重建。在界面调整过看板后先运行 `python3 verify.py` 更新导出。认证信息另存于 `.local/stack.env`，应单独保留在本地私有备份中，不能进入共享归档。重复执行 `./ctl.sh up` 会保留已有看板；只有首次启动缺少默认看板时才创建。若确实需要恢复默认布局，显式执行 `python3 setup.py kibana`（会覆盖默认看板及其检索对象）。

系统部署与导入检查记在本地 `.local/acceptance.json`。当前所有工作都在本机，不产生云计算或存储费用。

官方依据：[Docker 部署](https://www.elastic.co/docs/deploy-manage/deploy/self-managed/install-elasticsearch-docker-compose)、[组件版本一致性](https://www.elastic.co/docs/deploy-manage/deploy/self-managed/install-kibana)、[Colima](https://github.com/abiosoft/colima)。
