# Kudada：两天站立策略优化课程

课程必做任务是改善**零速度指令下的站立表现**。站立和行走使用同一个独立策略。你会从可训练的检查点开始，提出假设，只改一个因素续训，并用相同条件比较 A（起点）、B（原配置续训）、C（改进配置续训）。行走和启停测试用于检查修改的副作用。

本仓库提供机器人模型、Motrix 训练环境、PPO 课堂训练器、可训练起点、评估工具、MuJoCo 可视化和固定接口 ONNX 导出。课程没有要求从零训练，也不包含跌倒起身任务。

## 获取仓库

已安装 Git 时，在 PowerShell 中执行：

```powershell
git clone https://github.com/NEXBOT-Embodied/kudada_rl_stu.git kudada_rl_stu_repo
cd kudada_rl_stu_repo
```

也可以下载 GitHub ZIP 或使用教师发放的学生包，完整解压后进入仓库目录。

## 开始前

首次参加课程，请先按 [Day1课前准备清单](docs/preclass_preparation.md) 完成软硬件准备、基础知识预习和功能检查。

推荐 Windows 10/11、Python 3.10.x、至少 8 GB 内存。本课训练器固定使用 CPU；NVIDIA GPU 不是必要条件。GPU 用于另行配置的研发拓展。安装需联网，安装完成后训练和回放可离线。

```powershell
# 已安装 uv：创建本仓库 .venv 并安装依赖
.\install.ps1
.\run.ps1 doctor

# 或用已有 Python 3.10 创建本仓库 .venv
.\install.ps1 -PythonPath C:\path\to\python.exe

# 或直接给已有 Conda 环境安装依赖
.\install-conda.ps1 -PythonPath C:\path\to\conda\envs\course\python.exe
.\run.ps1 -PythonPath C:\path\to\conda\envs\course\python.exe doctor
```

`run.ps1` 默认只使用本仓库 `.venv`，不会自动寻找邻目录，也不会读取失效的 Conda 标记。使用已有环境时，后续每条命令都显式加 `-PythonPath`。Linux 入口为 `bash install.sh /path/to/python3.10` 和 `.venv/bin/python -m course.cli ...`。

## 课堂闭环

以下命令在本仓库目录执行。评估 A 必须在改参数前保留。完整评估包含 5 组固定条件，站立 60 秒、行走 30 秒、连续启停 10/20/10 秒；启动前 5 秒与稳态分别统计。详细课堂主指标与预演结果由教师发放。

`configs/improve.json` 是待验证的单因素提案。当前 C5000 预演是反例：B 原配置续训已经让站立速度接近零，C 的微小数值差不足以主张有效改善，且 C 的行走横移更大。请按完整结果决定是否接受候选；配置名“improve”不表示它已优于对照。教师会发放实际预演证据和失败分析。

```powershell
.\run.ps1 play pd --seconds 10
.\run.ps1 play stand --checkpoint models/teaching_start.pt --seconds 60
.\run.ps1 play walk --checkpoint models/teaching_start.pt --seconds 30
.\run.ps1 evaluate --checkpoint models/teaching_start.pt --output reports/local/A

.\run.ps1 train --config configs/baseline.json --run-dir runs/B
.\run.ps1 train --config configs/improve.json --run-dir runs/C

# latest.pt 是该运行的最新检查点；保留对应 verification.json 和训练配置。
.\run.ps1 evaluate --checkpoint runs/B/latest.pt --output reports/local/B
.\run.ps1 evaluate --checkpoint runs/C/latest.pt --output reports/local/C
.\run.ps1 compare --a reports/local/A --b reports/local/B --c reports/local/C --output reports/local/ABC.json

.\run.ps1 export --checkpoint runs/C/latest.pt --observations runs/C/observations.npz --output reports/local/C/candidate.onnx
.\run.ps1 play stand --onnx reports/local/C/candidate.onnx --seconds 60
.\run.ps1 play transition --onnx reports/local/C/candidate.onnx --seconds 40
```

`play pd` 是零动作 PD 诊断，默认位置增益乘 4；`play stand` 加载真实 RL 模型、输入零速度命令，使用机器人控制参数。二者不能混为同一个测试。`play` 在 MuJoCo 中回放，`train` 和 `evaluate` 在 MotrixLab/MotrixSim 中执行，报告记录各自引擎。

录制视频：` .\run.ps1 play stand --checkpoint models/teaching_start.pt --seconds 10 --headless --video reports/local/start.mp4`。快速检查可用 `evaluate --modes stand --seconds 2 --conditions 5`，这种缩短时长的结果明确标记为 smoke，不能替代完整课堂评估。

## 仓库内容

|路径|用途|
|---|---|
|`models/teaching_start.pt` / `.onnx`|可训练检查点与对应部署文件，身份和限制见教师发布说明|
|`configs/`|B/C 同等预算、单变量实验配置|
|`course/train.py`|课堂 PPO：热启动 Actor/Critic，重新建立 Adam；HIM 架构时冻结其编码器|
|`course/evaluate.py`|5 条件配对评估及每步原始轨迹|
|`course/runtime.py`|真实模型加载、固定 `[1,924] → [1,11]` ONNX 导出|
|`scratch_policy/`|独立 Actor、兼容 HIM 演示模型与训练环境继承关系|
|`kudada_motrix/`, `tinker_policy/env.py`|机器人状态、观测和控制解码|
|`vendor/`|固定版本的 Motrix 环境源码，不含整个研发工作区|
|`assets/kudada/`, `references/tinker/`|MJCF、mesh 与已有控制契约 YAML|
|`docs/preclass_preparation.md`|Day1前的软硬件、知识准备和功能检查清单|
|`docs/student_lab.md`|课堂操作、记录和交付要求|
|`docs/troubleshooting.md`|失败处理|

模型的 ONNX 接口是 float32 `[1,924] → [1,11]`，由 21 帧、每帧 44 项组成，最旧帧在前，启动历史补零。前 10 项用于关节动作，第 11 项在当前控制程序中被忽略。真实位置、真实速度和朝向误差不作为 Actor 输入；仿真可用它们算奖励、Critic 和评估指标。

CP550 的普通 Actor 实际取历史末尾 5 帧（220 项）作为网络特征；输入接口仍接收完整 21 帧（924 项）。教师 CP855 演示模型还使用 21 帧的 HIM 感知编码器。查看检查点 `actor_config` 可以确认模型结构，不能只根据 ONNX 输入长度猜算法。

仿真评估、导出校验和真机验证是三个不同的证据。候选通过课堂检查后交教师组织实机验证；学生提交候选不会自动替换机器人默认模型。完整来源和许可见 [NOTICE.md](NOTICE.md)。

教学起点来自真实早期独立训练 CP550，有持续前后摆动和行走横移，供仿真分析与优化。它不是本次可直接上机的默认演示模型。教师另持 CP855 的部署演示模型；两份模型身份不可互换。
