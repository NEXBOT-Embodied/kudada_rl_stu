# Day1 课前准备清单

课前准备目标：**每组至少一台电脑能够运行课程环境，每位学员会基本 Python、命令行和配置文件操作。**完成安装与功能检查后，策略训练、优化和完整评估按课堂安排进行。

## 1. 电脑与硬件

| 项目 | 准备要求 |
|---|---|
| 电脑 | 推荐64位 Windows 10/11；本课程已在 Windows 验证 |
| 内存 | 建议至少8GB，推荐16GB |
| 硬盘 | 建议预留10GB空闲空间，用于环境、模型和实验输出 |
| GPU | 无需专门准备，本课训练器使用CPU |
| 随身物品 | 电源适配器、鼠标；课堂训练时连接电源 |
| 网络 | 课前安装依赖需要联网；安装完成后可离线训练和仿真 |
| 机器人 | 学员无需自行购买或携带，由课程组织方统一安排 |

内存和磁盘数值是备课建议。Linux 用户应提前完成安装验证；当前发布包的 Linux 安装尚未实测。每组建议2–3人，至少准备一台已经通过检查的电脑。

## 2. 软件与课程文件

领取教师发放的学生仓库 ZIP，完整解压后再运行。建议放在简单路径，例如 `C:\course\kudada_rl_stu_repo`。

需要准备：

- **Python 3.10.x**：可以使用已有 Conda 环境，也可以通过 `uv` 创建课程环境；两种方式选一种。
- **PowerShell**：用于执行课程命令。
- **代码编辑器**：推荐 VS Code，用于阅读 Python、修改 JSON、查看报告。
- **浏览器**：用于阅读课程资料。

按照 [学生 README](../README.md#开始前) 安装依赖。PyTorch、Motrix、MuJoCo、ONNX Runtime 等由安装脚本统一安装，学员不需要自行挑选版本。排查安装问题时参考 [故障处理](troubleshooting.md)。

安装方式选一条即可：

```powershell
# 方式一：已安装 uv，创建本仓库 .venv 并安装依赖
.\install.ps1

# 方式二：已有 Python 3.10，创建本仓库 .venv
# 将示例路径替换为自己的 python.exe 绝对路径
.\install.ps1 -PythonPath C:\path\to\python.exe

# 方式三：直接给已有 Conda Python 3.10 环境安装依赖
# 将示例路径替换为自己的 Conda 环境解释器
.\install-conda.ps1 -PythonPath C:\path\to\conda\envs\course\python.exe
```

以下功能检查默认使用本仓库 `.venv`。若选择直接使用已有 Conda 环境，后续每条命令都要显式添加 `-PythonPath`，例如：

```powershell
.\run.ps1 -PythonPath C:\path\to\conda\envs\course\python.exe doctor
```

## 3. 安装后必须完成的功能检查

在学生仓库目录执行：

```powershell
# 检查环境
.\run.ps1 doctor

# 检查机器人资源和 GUI
.\run.ps1 play pd --seconds 10

# 检查教学起点策略能加载
.\run.ps1 play stand --checkpoint models/teaching_start.pt --seconds 10

# 检查 Motrix 评估环境
.\run.ps1 evaluate --checkpoint models/teaching_start.pt `
  --modes stand --seconds 2 --conditions 5 `
  --output reports/local/preclass
```

通过标准：

- `doctor` 的 `missing` 列表为空，`default_checkpoint_present` 为 `true`。`cuda_available=false` 不影响本课。
- GUI 能显示完整机器人，仿真可以运行；观察结束后正常关闭窗口。
- 短评估正常结束，生成 `reports/local/preclass/evaluation.json`，包含5组条件。
- 教学起点出现摆动属于预期现象，课前无需修改模型或控制参数。
- 这次2秒评估只检查功能，不能作为策略性能验收。

`play pd` 是零动作 PD 诊断，默认位置增益乘4；`play stand` 加载真实 RL 策略并输入零速度命令。GUI 使用 MuJoCo，短评估使用 Motrix；两者都要检查。

## 4. 知识准备

| 内容 | Day1前建议达到的程度 |
|---|---|
| 命令行 | 会进入目录、执行命令、识别文件路径，能保留完整报错 |
| Python | 理解变量、列表、循环、函数和导入；能读懂简单脚本 |
| JSON配置 | 能找到一个参数、修改数值并保存，知道修改前要保留原文件 |
| 基础数学 | 理解平均值、二维向量、速度与位移；知道角度和弧度不同 |
| 机器人概念 | 初步了解关节角、关节速度、IMU、目标角和实际角 |
| 实验记录 | 知道记录配置、模型文件、运行命令和结果，避免覆盖旧结果 |

**PPO、GAE、奖励设计、观测历史和 ONNX 导出不要求课前熟练掌握，由课程讲授。**建议安排30–60分钟预习，重点阅读 [README](../README.md) 和 [实验手册](student_lab.md)，了解课程将完成什么。

## 5. 带到课堂的材料

- 已安装好的课程环境及完整学生仓库。
- `doctor` 输出、一次 GUI 截图、短评估报告。
- 个人电脑配置和安装中遇到的问题。
- 对两个问题的初步想法：“零速度指令为什么仍可能晃动？”“怎样证明修改奖励确实带来改善？”

课前自查：

- [ ] 已完整解压学生仓库，能够找到 `run.ps1` 和 `models/teaching_start.pt`。
- [ ] 课程环境安装完成，`doctor` 没有缺失依赖。
- [ ] PD 与 RL 两种 GUI 检查都能够运行。
- [ ] Motrix 短评估已生成报告。
- [ ] 能打开并阅读 Python、JSON 和 Markdown 文件。
- [ ] 已保存检查结果和报错，准备好电源适配器及鼠标。

建议教师提前3–7天发放课程包，在开课前一天收齐环境检查结果，提前安排有安装问题的学员使用备用工位。
