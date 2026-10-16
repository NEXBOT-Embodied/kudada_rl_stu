# 学生实验手册

## 课程目标与使用方式

课程必做任务是改善**零速度指令下的站立表现**。站立和行走使用同一个独立策略。你会从可训练的检查点开始，提出假设，只改一个因素续训，并用相同条件比较 A（起点）、B（原配置续训）、C（改进配置续训）。行走和启停测试用于检查修改的副作用。

本仓库提供机器人模型、Motrix 训练环境、PPO 课堂训练器、可训练起点、评估工具、MuJoCo 可视化和固定接口 ONNX 导出。课程没有要求从零训练，也不包含跌倒起身任务。

开始前按 [README](../README.md) 完成安装与功能验证。**下文所有命令在学生仓库根目录执行**，统一使用本仓库 `.venv`，无需激活环境。Linux 用户将 `.\run.ps1` 替换为 `bash run.sh`。文件用途和模型接口见 [仓库与模型说明](repository_guide.md)。

## 1. 观察教学起点

先执行 `doctor`，保存解释器、包版本与 CUDA 状态。播放 `models/teaching_start.pt` 的站立与行走，记录前 5 秒和后续稳态的区别：是否有身体晃动、脚部调整、启动漂移，是否最终稳定。不要凭 GUI 猜测平均速度或漂移数值。

```powershell
.\run.ps1 doctor
.\run.ps1 play stand --checkpoint models/teaching_start.pt --seconds 60
.\run.ps1 play walk --checkpoint models/teaching_start.pt --seconds 30
.\run.ps1 evaluate --checkpoint models/teaching_start.pt --output reports/local/A
```

评估 A 必须在改参数前保留。完整评估包含 5 组固定条件，站立 60 秒、行走 30 秒、连续启停 10/20/10 秒；启动前 5 秒与稳态分别统计。详细课堂主指标与预演结果由教师发放。README 中的课前 2 秒短评估只检查环境功能，不能替代本节完整评估。

打开 `reports/local/A/stand.json`：每个 `rows` 元素对应一组初始条件。`planar_speed_startup_mean` 是 0–5 秒内每步水平速度模的平均；`planar_speed_steady_mean` 从第 5 秒后开始。前后方向的有符号速度相互抵消不能说明没有晃动。`planar_travel_startup_m` 是启动期间累计走过的水平路程，`max_drift_startup_m` 是相对起点的最大位移；二者含义不同。

`survived=false` 的样本保留失败原因与失败时间，倒地后的静止帧不参与评分。不要删掉失败条件，也不要把重新出生后的站立当作存活。

### 课堂 E1：用一个关节辨认控制路线

先预测零动作 PD 与零速度 RL 中，哪条路线的关节目标保持固定。按教师安排顺序运行：

```powershell
.\run.ps1 play pd --seconds 10 --signals reports/local/e1-pd.csv
.\run.ps1 play stand --checkpoint models/teaching_start.pt --seconds 10 --signals reports/local/e1-rl.csv
```

两次默认都记录关节 0（左髋偏航）。终端 `joint_signals` 的 `action_span`、`target_span_rad`、`position_span_rad` 分别给出动作、目标角与实际角的变化范围；CSV 每行记录同一控制时刻的动作、目标角、实际角、角速度和力矩，同名 JSON 绑定模型与采样身份。先检验预测，再引用一行读数解释目标与反馈的关系。PD 诊断增益为 KP×4，这两次短运行用于理解控制路径。

## 2. 提出可检验的假设

阅读 `course/train.py` 的 `StandingCourseEnv._reward_terms`。用自己的话描述 `standing_speed_cost` 惩罚什么，何时生效，它改变奖励还是 Actor 输入。阅读 `configs/baseline.json` 与 `configs/improve.json`，列出二者唯一的实验因素。

`configs/improve.json` 是待验证的单因素提案。当前 C5000 预演是反例：B 原配置续训已经让站立速度接近零，C 的微小数值差不足以主张有效改善，且 C 的行走横移更大。请按完整结果决定是否接受候选；配置名“improve”不表示它已优于对照。教师会发放实际预演证据和失败分析。

先写预测再训练，例如：“提高零速度阶段的水平速度惩罚可能减少启动晃动；它也可能让行走启动更慢，因此必须测试启停。”请写清观察依据、主指标、可能副作用和对照方式。教师发放的参考改善目标不等于你的运行必然达到目标。

## 3. 做 B/C 续训

从同一个 `models/teaching_start.pt` 开始、相同预算、相同 seed 和优化配置。B 使用原配置，C 只改指定因素。训练器记录来源检查点哈希、配置、每次迭代日志、最终检查点和源代码哈希。

```powershell
.\run.ps1 train --config configs/baseline.json --run-dir runs/B
.\run.ps1 train --config configs/improve.json --run-dir runs/C
```

`latest.pt` 是该运行的最新检查点；保留对应 `verification.json` 和训练配置。重复实验时使用新的运行目录，并在后续评估中同步修改路径。

课程训练使用独立策略热启动，Actor/Critic 权重来自起点。Adam 重新建立、探索标准差固定。若使用带 HIM 的架构，其感知编码器冻结；本次教学起点 CP550 使用普通 Actor，没有 HIM 编码器。因此它是课堂热启动续训，不是完整恢复原研究训练进程；B/C 的算法与预算一致。

课程单因素实验完成前，不同时修改网络、动作缩放、PD、步长、随机化和奖励。不要以训练奖励提高替代 A/B/C 物理评估。本课固定 CPU；若内存不足，整组 B/C 同时降低 `num_envs` 并重新记录预算，不能只缩短其中一组。

## 4. 对照评估

```powershell
.\run.ps1 evaluate --checkpoint runs/B/latest.pt --output reports/local/B
.\run.ps1 evaluate --checkpoint runs/C/latest.pt --output reports/local/C
.\run.ps1 compare --a reports/local/A --b reports/local/B --c reports/local/C --output reports/local/ABC.json
```

对 B/C 执行与 A 完全一致的 `evaluate` 命令，并执行 `compare`。工具校验种子、初始物理状态、时长、命令序列、控制周期、物理步长、评估代码和模型加载/环境/控制参数/资产源码哈希；条件不同会拒绝比较。缺少来源绑定或不满足 5 条件与完整时长的报告只能作为 smoke。教师旧预演的同目录 `source_binding.json` 是事后保存并绑定原始报告字节的来源快照，比较报告标记为 `retrospective`；新评估直接在报告中记录 `in_report`。

先检查每组是否存活、是否触发关节速度保护，再看教师指定的站立主指标。启动、稳态、漂移、姿态和动作变化都要保留。改善只出现在一个条件中，需报告结果分布。检查 `walk.json` 与 `transition.json`，说明站立改动是否牺牲行走或启动/停止。

如果 B 的速度已经接近零，百分比会夸大微小数值差。比较工具在参考速度低于 `1e-4 m/s` 时不计算百分比，此时关注绝对差、测量分辨率和回归。当前发放的 C5000 配置是用于判断副作用的提案，不是保证通过的标准答案。

原始轨迹保存在 `stand.trace.npz`、`walk.trace.npz` 和 `transition.trace.npz`。数组 `active` 表明该行仍属于首次物理回合；作图时必须据此屏蔽复位后的数据。报告中的指标已经按首次回合处理。

## 5. 导出与部署交接

以下以 C 为导出示例，实际交付应根据完整评估选择候选；若选择其他检查点，同步修改输入路径和输出名称。

```powershell
.\run.ps1 export --checkpoint runs/C/latest.pt --observations runs/C/observations.npz --output reports/local/C/candidate.onnx
.\run.ps1 play stand --onnx reports/local/C/candidate.onnx --seconds 60
.\run.ps1 play transition --onnx reports/local/C/candidate.onnx --seconds 40
```

导出自己的检查点，再加载**导出后的 ONNX**回放，避免只播放 PT。查看 `.verification.json` 的输入/输出形状、历史顺序、模型哈希和数值误差。数值误差小于 `1e-4` 是导出检查，不是物理性能达标。

向教师提交：实验假设与唯一改动、A/B/C 配置、训练日志和检查点、完整评估 JSON/NPZ、比较结果、候选 ONNX 及校验报告。请给出可重复命令和已知限制。

真机验证由教师组织。使用教师已经核对过的控制程序和参数，备份默认模型，保留回退方法。先零速度观察，再按教师流程增加低速命令。不要在仿真文档中自行添加真机没有的观测字段。

## 可选：录制回放视频

```powershell
.\run.ps1 play stand --checkpoint models/teaching_start.pt --seconds 10 --headless --video reports/local/start.mp4
```

GUI 或命令报错时参考 [故障排查](troubleshooting.md)。

## 实验记录

|字段|填写内容|
|---|---|
|起点 SHA256|由发布说明或模型加载输出获得|
|观察到的问题|区分启动与稳态|
|单因素假设|因素、方向、预期效果和副作用|
|B/C 配置|完整 JSON 与训练预算|
|评估条件|seed、5 组初始状态、时长与引擎|
|站立结果|每条件主指标、漂移、姿态、保护与存活|
|回归结果|行走速度、横移、偏航、启停表现|
|导出校验|IO、parity、ONNX 与 PT 哈希|
|结论|接受、拒绝或需要更多证据，以及理由|
