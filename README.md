# Kudada：两天站立策略优化课程

这是课程的学生仓库。请在 Day1 开始前完成本页的知识准备、环境安装和功能验证：**每组至少一台电脑能够运行课程环境，每位学员会基本 Python、命令行和配置文件操作。**课程实验流程见 [学生实验手册](docs/student_lab.md)。

## 软硬件准备

| 项目 | 准备要求 |
|---|---|
| 电脑 | 推荐 64 位 Windows 10/11；课程环境已在 Windows 验证 |
| 内存 | 建议至少 8 GB，推荐 16 GB |
| 硬盘 | 建议预留 10 GB 空闲空间，用于环境、模型和实验输出 |
| GPU | 本课训练器使用 CPU，无需准备 NVIDIA GPU |
| 网络 | 课前安装依赖需要联网；安装完成后可离线训练和仿真 |
| 软件 | PowerShell、代码编辑器（推荐 VS Code）、浏览器；用 Git 克隆时需安装 Git |
| 随身物品 | 电源适配器、鼠标；课堂训练时连接电源 |
| 机器人 | 由课程组织方统一安排，学员无需自行购买或携带 |

内存和磁盘数值是准备建议。每组建议 2–3 人，至少准备一台已经通过下文验证的电脑。

## 前置知识

| 内容 | Day1 前建议达到的程度 |
|---|---|
| 命令行 | 会进入目录、执行命令、识别文件路径，能保留完整报错 |
| Python | 理解变量、列表、循环、函数和导入；能读懂简单脚本 |
| JSON 配置 | 能找到一个参数、修改数值并保存，知道修改前要保留原文件 |
| 基础数学 | 理解平均值、二维向量、速度与位移；知道角度和弧度不同 |
| 机器人概念 | 初步了解关节角、关节速度、IMU、目标角和实际角 |
| 实验记录 | 会保存配置、模型文件、运行命令和结果，避免覆盖旧结果 |

PPO、GAE、奖励设计、观测历史和 ONNX 导出由课程讲授，不要求课前熟练掌握。建议安排 30–60 分钟预习，重点完成本页操作，再浏览 [实验手册](docs/student_lab.md) 的课程目标。

## 获取仓库

已安装 Git 时，在 PowerShell 中执行：

```powershell
git clone https://github.com/NEXBOT-Embodied/kudada_rl_stu.git kudada_rl_stu_repo
cd kudada_rl_stu_repo
```

也可以下载 GitHub ZIP 或使用教师发放的学生包。完整解压后，在包含 `README.md`、`install.ps1` 和 `run.ps1` 的目录打开 PowerShell。建议使用简单路径，例如 `C:\course\kudada_rl_stu_repo`。

**下文所有命令都在学生仓库目录执行。**

## 环境安装

需要 Python **3.10.x**。PyTorch、Motrix、MuJoCo、ONNX Runtime 等由安装脚本统一安装，无需自行挑选版本。以下三种方式选一种即可。

### 方式一：使用 uv

先按 [uv 官方安装说明](https://docs.astral.sh/uv/getting-started/installation/) 安装 uv，重新打开 PowerShell，确认 `uv --version` 能运行，然后执行：

```powershell
.\install.ps1
```

脚本会创建本仓库的 `.venv`，使用 Python 3.10.21 并安装课程依赖。

### 方式二：使用已有 Python 3.10 创建本仓库环境

已有 Python 3.10 或 Conda Python 3.10 时，可用它创建本仓库的 `.venv`。将示例路径替换为自己的 `python.exe` 绝对路径：

```powershell
.\install.ps1 -PythonPath 'C:\path\to\python.exe'
```

方式一、二安装后，都可以直接运行 `.\run.ps1 ...`，无需手动激活 `.venv`。

### 方式三：直接使用已有 Conda 环境

确认该环境使用 Python 3.10.x，再将示例路径替换为该环境解释器的绝对路径：

```powershell
.\install-conda.ps1 -PythonPath 'C:\path\to\conda\envs\course\python.exe'
.\run.ps1 -PythonPath 'C:\path\to\conda\envs\course\python.exe' doctor
```

此方式直接向所选环境安装依赖。后续每条 `.\run.ps1` 命令都要添加 `-PythonPath`；仅激活 Conda 不会改变脚本默认使用的 `.venv` 路径。

### Linux 用户

Linux 安装入口如下，当前发布包的 Linux 安装尚未实测，请提前完成下文同等功能验证：

```bash
bash install.sh /path/to/python3.10
.venv/bin/python -m course.cli doctor
```

后续将 `.\run.ps1` 替换为 `.venv/bin/python -m course.cli`。安装报错时参考 [故障排查](docs/troubleshooting.md)。

## 安装验证

使用本仓库 `.venv` 时，依次执行以下四项检查。每次 GUI 观察结束后关闭窗口，再执行下一条命令。

```powershell
# 1. 检查解释器、依赖和教学模型
.\run.ps1 doctor

# 2. 检查机器人资源和 GUI
.\run.ps1 play pd --seconds 10

# 3. 检查教学起点策略能加载并回放
.\run.ps1 play stand --checkpoint models/teaching_start.pt --seconds 10

# 4. 检查 Motrix 评估环境
.\run.ps1 evaluate --checkpoint models/teaching_start.pt `
  --modes stand --seconds 2 --conditions 5 `
  --output reports/local/preclass
```

若选择直接使用已有 Conda 环境，对应命令为：

```powershell
$coursePython = 'C:\path\to\conda\envs\course\python.exe'
.\run.ps1 -PythonPath $coursePython doctor
.\run.ps1 -PythonPath $coursePython play pd --seconds 10
.\run.ps1 -PythonPath $coursePython play stand --checkpoint models/teaching_start.pt --seconds 10
.\run.ps1 -PythonPath $coursePython evaluate --checkpoint models/teaching_start.pt `
  --modes stand --seconds 2 --conditions 5 `
  --output reports/local/preclass
```

| 检查 | 通过标准 |
|---|---|
| 环境检查 | `doctor` 的 `missing` 列表为空，`default_checkpoint_present` 为 `true`；`cuda_available=false` 不影响本课 |
| PD 回放 | GUI 显示完整机器人，仿真可以运行并正常关闭 |
| 策略回放 | `models/teaching_start.pt` 能加载、仿真能运行；教学起点的前后摆动属于预期现象 |
| 短评估 | 正常结束，生成 `reports/local/preclass/evaluation.json`，包含 5 组条件 |

`play pd` 是零动作 PD 诊断，默认位置增益乘 4；`play stand` 加载 RL 策略并输入零速度命令。GUI 使用 MuJoCo，短评估使用 Motrix，两者都要检查。这次 2 秒评估只验证环境功能，课前无需调参；完整性能评估按 [实验手册](docs/student_lab.md) 在课堂进行。

## 课前自查与课程文档

完成 [Day1 课前准备清单](docs/preclass_preparation.md)，保存 `doctor` 输出、一次 GUI 截图和短评估报告，并带好电源适配器及鼠标。遇到问题时保留完整命令和报错，交教师提前处理。

| 文档 | 用途 |
|---|---|
| [课前准备清单](docs/preclass_preparation.md) | 自查与带到课堂的材料 |
| [学生实验手册](docs/student_lab.md) | 课程目标、A/B/C 实验、完整评估、ONNX 导出与交付 |
| [仓库与模型说明](docs/repository_guide.md) | 文件用途、模型身份、控制接口和验证范围 |
| [故障排查](docs/troubleshooting.md) | 安装、GUI、训练和评估问题 |
| [来源与许可](NOTICE.md) | 代码、资产与模型的来源说明 |
