# 学生实验手册

## 课程目标与使用方式

两天里你会走完一遍完整闭环：看到教学起点 A 的站立问题，理解控制链和强化学习，在玩具任务里体会奖励怎样决定行为，从 A 续训出本组模型，导出 ONNX，第二天部署到青龙 Nano 上，再在监护下调部署参数。课程不要求推导 PPO，也不要求设计出有效的奖励。

开始前按 [README](../README.md) 完成安装与功能验证。**下文所有命令在学生仓库根目录执行**，统一使用本仓库 `.venv`，无需激活环境。Linux 用户将 `.\run.ps1` 替换为 `bash run.sh`。文件用途和模型接口见 [仓库与模型说明](repository_guide.md)。

## 1. 观察教学起点 A

播放 `models/teaching_start.pt` 的站立与行走。站立时注意身体是否持续晃动、脚是否在调整；行走时注意是否走偏。先写下看到的现象，不凭 GUI 猜数值。

```powershell
.\run.ps1 play stand --checkpoint models/teaching_start.pt --seconds 30
.\run.ps1 play walk --checkpoint models/teaching_start.pt --seconds 30
```

### 课堂 E1：用一个关节辨认控制路线

先预测零动作 PD 与零速度 RL 中，哪条路线的关节目标保持固定。按教师安排顺序运行：

```powershell
.\run.ps1 play pd --seconds 10 --signals reports/local/e1-pd.csv
.\run.ps1 play stand --checkpoint models/teaching_start.pt --seconds 10 --signals reports/local/e1-rl.csv
```

两次默认都记录关节 0（左髋偏航）。终端 `joint_signals` 的 `action_span`、`target_span_rad`、`position_span_rad` 分别给出动作、目标角与实际角的变化范围；CSV 每行记录同一控制时刻的动作、目标角、实际角、角速度和力矩。先检验预测，再引用一行读数解释目标与反馈的关系。PD 诊断增益为 KP×4，这两次短运行用于理解控制路径。

## 玩具任务：改奖励，看行为

下午 14:30 在课件里完成，不需要运行本仓库命令。上午的滑台从 −0.6 m 出发要停在 0 m；你可以调四个奖励项（到达奖励、距离惩罚、用力惩罚、每步奖励），先写预测，再点训练，浏览器里约 1 秒训完并自动回放。

每次尝试：选奖励 → 写预测 → 训练 → 看完回放 → 用一句话解释“奖励鼓励了什么，所以它学成了这样”。一次只改一项。至少完成一次，有时间再试第二次，结束时点“导出实验记录”。

## 2. 从 A 续训本组模型

先预测：让 A 按原配置再练一会儿，站立晃动和行走走偏会怎样？写下预测再训练。

```powershell
.\run.ps1 train --config configs/baseline.json --run-dir runs/G01
```

把 `G01` 换成本组组号。训练 20 轮，CPU 上约 6–7 分钟；终端每轮打印一行，`reward` 是平均奖励，`failures` 是本轮倒地次数。结束后 `runs/G01/latest.pt` 是本组模型。重复训练时换一个新的运行目录。

课堂训练从 A 热启动，Actor/Critic 权重来自 A。奖励升高说明策略更符合奖励的要求，不能直接说明机器人表现更好，还要看回放。

训练报错或太慢时，改用教师预训的 B：它用同一份课堂配置从 A 续训 20 轮得到，文件已在 `models/` 里（`baseline_B.pt`、`baseline_B.onnx`，哈希见 `models/manifest.json` 的 `fallback`）。下面各节把 `runs/G01/latest.pt` 换成 `models/baseline_B.pt` 即可；导出这一步可以跳过，直接用 `models/baseline_B.onnx` 做 sim2sim，展示时说明用的是 B。

## 3. 回放对比

用同样的命令分别回放 A 和本组模型，说出什么变了、什么没变：

```powershell
.\run.ps1 play stand --checkpoint runs/G01/latest.pt --seconds 30
.\run.ps1 play walk --checkpoint runs/G01/latest.pt --seconds 30
```

需要数字时，用短评估生成简表：

```powershell
.\run.ps1 evaluate --checkpoint models/teaching_start.pt --output reports/local/A --modes stand walk --seconds 20
.\run.ps1 evaluate --checkpoint runs/G01/latest.pt --output reports/local/G01 --modes stand walk --seconds 20
```

打开各目录的 `stand.json` 与 `walk.json`：`rows` 的每个元素是一组初始条件；`survived` 表示是否站住，`planar_speed_steady_mean` 是第 5 秒后的平均水平速度（站立时越接近 0 越稳），`max_lateral_m` 是行走时最大横向偏移。启停模式 `transition` 固定 40 秒，课后可去掉 `--modes` 与 `--seconds` 跑完整评估。

## 4. 导出 ONNX 并做 sim2sim

```powershell
.\run.ps1 export --checkpoint runs/G01/latest.pt --observations runs/G01/observations.npz --output reports/local/G01/G01.onnx
.\run.ps1 play stand --onnx reports/local/G01/G01.onnx --seconds 30
.\run.ps1 play walk --onnx reports/local/G01/G01.onnx --seconds 30
```

训练和评估用 MotrixSim，`play` 用 MuJoCo，所以用导出的 ONNX 回放就是一次 sim2sim。站立不倒、小速度能走，才能进入第二天的上机队列。导出时会生成 `.verification.json`，记录输入/输出形状 `[1,924] → [1,11]` 和数值误差。

向教师提交 ONNX 文件名、SHA256（导出输出中的 `sha256`）和 sim2sim 结果。真机部署由机位负责人按现场流程执行，学生跟读 diff 与加载日志。

## 5. 部署参数怎样影响行为

同一个模型，部署时的 PD 增益、动作缩放和动作低通不同，行为也会不同。下面的参数只用于课堂仿真演示，训练和评估不使用；每次只改一项，先站立、再行走各看一次：

```powershell
.\run.ps1 play stand --checkpoint models/teaching_start.pt --seconds 15 --policy-kp-scale 0.5
.\run.ps1 play walk --checkpoint models/teaching_start.pt --seconds 15 --action-filter-rate 10
```

`--policy-kp-scale` 调整策略模式的 KP，`--kd-scale` 调整 KD；PD 模式仍用 `--kp-scale` 调整 KP。`--action-scale` 替换动作缩放（真机默认 0.25），`--action-filter-rate` 替换动作低通速率（单位 1/s）。结果中的 `deployment_overrides` 记录传入值和实际参数，`teaching_demo_only=true` 表示这是仿真演示，不代表真机配置。

## 可选：录制回放视频

```powershell
.\run.ps1 play stand --checkpoint models/teaching_start.pt --seconds 10 --headless --video reports/local/start.mp4
```

GUI 或命令报错时参考 [故障排查](troubleshooting.md)。

## 实验记录

|字段|填写内容|
|---|---|
|A 的现象|站立与行走各一句|
|玩具任务|改了哪一项、预测、实际行为、一句解释|
|续训前的预测|晃动、走偏会怎样|
|续训结果|运行目录、训练轮数、最后一轮 reward 与 failures|
|回放对比|什么变了、什么没变，附简表数字|
|sim2sim|ONNX 文件名、SHA256、站立与行走是否通过|
|真机观察|第二天填写：与 A 或演示模型相比的现象|
|部署参数调整|改了哪一项、取值、前后现象|
