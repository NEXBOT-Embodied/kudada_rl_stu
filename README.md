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
| Python | 理解变量、列表、字典、循环、函数和导入；能读懂简单脚本 |
| JSON 配置 | 能找到一个参数、修改数值并保存，知道修改前要保留原文件 |
| 基础数学 | 理解平均值、二维向量、速度与位移；知道角度和弧度不同 |
| 机器人概念 | 初步了解关节角、关节速度、IMU、目标角和实际角 |
| 强化学习基础 | 理解智能体与环境、观测与动作、奖励与累计回报、策略、探索和回合；能区分训练与评估 |
| 实验记录 | 会保存配置、模型文件、运行命令和结果，避免覆盖旧结果 |

详细操作目标、入门概念和基础知识自测见 [课前准备清单](docs/preclass_preparation.md)。请课前阅读 [强化学习基础预习](docs/rl_basics.md) 并完成自测。PPO、GAE、奖励设计、观测历史和 ONNX 导出由课程讲授，不要求课前熟练掌握。有基础的学员建议安排 30–60 分钟复习与自测，再完成本页环境安装和功能验证，并浏览 [实验手册](docs/student_lab.md) 的课程目标。

## 获取仓库

已安装 Git 时，在 PowerShell 中执行：

```powershell
git clone https://github.com/NEXBOT-Embodied/kudada_rl_stu.git kudada_rl_stu_repo
cd kudada_rl_stu_repo
```

也可以下载 GitHub ZIP 或使用教师发放的学生包。完整解压后，在包含 `README.md`、`install.ps1` 和 `run.ps1` 的目录打开 PowerShell。建议使用简单路径，例如 `C:\course\kudada_rl_stu_repo`。

**下文所有命令都在学生仓库目录执行。**

## 环境安装

所有安装和运行统一使用**学生仓库根目录内的 `.venv`**，Python 为 **64 位 3.10.x**，推荐 3.10.21。安装脚本统一安装固定版本依赖、PyTorch 2.7.0 CPU 和本仓库的 Motrix 源码包，并执行依赖检查与 `doctor`。

默认使用国内镜像，直接运行安装脚本即可：

| 下载内容 | 默认镜像 |
|---|---|
| Python 依赖、pip 引导包及 Motrix 构建依赖 | USTC：`https://mirrors.ustc.edu.cn/pypi/simple` |
| PyTorch CPU | 南京大学：`https://mirrors.nju.edu.cn/pytorch/whl/cpu` |
| uv 自动下载的 Python 3.10.21 | 南京大学：`https://mirrors.nju.edu.cn/github-release/astral-sh/python-build-standalone` |

安装脚本也会把 `.venv` 内 pip 的默认源设为 USTC；使用自定义 `PackageIndex` 时改为对应源。

### Windows

已有 Python 时，可通过 USTC 安装 uv：

```powershell
py -m pip install uv --index-url https://mirrors.ustc.edu.cn/pypi/simple
```

也可按 [uv 安装说明](https://docs.astral.sh/uv/getting-started/installation/) 安装。重新打开 PowerShell，确认 `uv --version` 能运行，然后执行：

```powershell
.\install.ps1
```

首次执行使用 uv 创建 Python 3.10.21 的 `.venv`；再次执行会复用已有 `.venv` 并检查、安装依赖，无需手动激活。

如果已经有 64 位 Python 3.10，可在首次安装时指定基础解释器，仍由同一个脚本创建仓库内 `.venv`。将示例路径替换为自己的 `python.exe` 绝对路径：

```powershell
.\install.ps1 -PythonPath 'C:\path\to\python.exe'
```

`-PythonPath` 仅用于创建 `.venv`。已有 Conda 的 Python 3.10 也可以作为基础解释器，课程依赖始终安装到 `.venv`，运行时统一使用 `.\run.ps1 ...`。

### Linux

安装 uv 后执行同等入口。

```bash
bash install.sh
bash run.sh doctor
```

已有 64 位 Python 3.10 时，可用 `bash install.sh /path/to/python3.10` 创建 `.venv`。后续将文档中的 `.\run.ps1` 替换为 `bash run.sh`；启动脚本均自动进入仓库目录。

### 更新与重建环境

更新仓库后重新运行安装脚本即可。

下载依赖失败时可以分别切换 Python 依赖源与 PyTorch 镜像，例如：

```powershell
.\install.ps1 -PackageIndex https://pypi.tuna.tsinghua.edu.cn/simple `
  -TorchIndex https://mirror.sjtu.edu.cn/pytorch-wheels/cpu
```

Linux 对应 `bash install.sh '' https://pypi.tuna.tsinghua.edu.cn/simple https://mirror.sjtu.edu.cn/pytorch-wheels/cpu`；第一个参数是创建 `.venv` 时使用的基础 Python 路径。uv 的 Python 下载镜像可通过 `UV_PYTHON_INSTALL_MIRROR` 环境变量覆盖。更多问题见 [故障排查](docs/troubleshooting.md)。

## 安装验证

依次执行以下四项检查。每次 GUI 观察结束后关闭窗口，再执行下一条命令。

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

| 检查 | 通过标准 |
|---|---|
| 环境检查 | `python` 指向本仓库 `.venv`，`missing` 列表为空，`default_checkpoint_present` 为 `true`；CPU 版的 `cuda_available=false` 为预期结果 |
| PD 回放 | GUI 显示完整机器人，仿真可以运行并正常关闭 |
| 策略回放 | `models/teaching_start.pt` 能加载、仿真能运行；教学起点的前后摆动属于预期现象 |
| 短评估 | 正常结束，生成 `reports/local/preclass/evaluation.json`，包含 5 组条件 |

`play pd` 是零动作 PD 诊断，默认位置增益乘 4；`play stand` 加载 RL 策略并输入零速度命令。GUI 使用 MuJoCo，短评估使用 Motrix，两者都要检查。这次 2 秒评估只验证环境功能，课前无需调参；完整性能评估按 [实验手册](docs/student_lab.md) 在课堂进行。

## 课前自查与课程文档

完成 [Day1 课前准备清单](docs/preclass_preparation.md)，保存 `doctor` 输出、一次 GUI 截图和短评估报告，并带好电源适配器及鼠标。遇到问题时保留完整命令和报错，交教师提前处理。

| 文档 | 用途 |
|---|---|
| [课前准备清单](docs/preclass_preparation.md) | 自查与带到课堂的材料 |
| [强化学习基础预习](docs/rl_basics.md) | 交互过程、回报、探索、回合、训练与评估，以及知识自测 |
| [学生实验手册](docs/student_lab.md) | 课程目标、A/B/C 实验、完整评估、ONNX 导出与交付 |
| [仓库与模型说明](docs/repository_guide.md) | 文件用途、模型身份、控制接口和验证范围 |
| [故障排查](docs/troubleshooting.md) | 安装、GUI、训练和评估问题 |
| [来源与许可](NOTICE.md) | 代码、资产与模型的来源说明 |
