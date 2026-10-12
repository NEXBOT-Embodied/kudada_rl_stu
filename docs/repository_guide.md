# 仓库与模型说明

安装和课前验证见 [README](../README.md)，课程操作见 [学生实验手册](student_lab.md)。

## 仓库内容

下表路径均相对学生仓库根目录。

| 路径 | 用途 |
|---|---|
| `models/teaching_start.pt` / `.onnx` | 可训练检查点与对应 ONNX，模型身份见下文及教师发布说明 |
| `models/manifest.json` | 教学起点的来源、模型哈希和限制 |
| `configs/` | B/C 同等预算、单变量实验配置 |
| `course/train.py` | 课堂 PPO：热启动 Actor/Critic，重新建立 Adam；HIM 架构时冻结其编码器 |
| `course/evaluate.py` | 5 条件配对评估及每步原始轨迹 |
| `course/runtime.py` | 模型加载、固定 `[1,924] → [1,11]` ONNX 导出 |
| `scratch_policy/` | 独立 Actor、兼容 HIM 演示模型与训练环境继承关系 |
| `kudada_motrix/`、`tinker_policy/env.py` | 机器人状态、观测和控制解码 |
| `vendor/` | 固定版本的 Motrix 环境源码，不含整个研发工作区 |
| `assets/kudada/`、`references/tinker/` | MJCF、mesh 与已有控制契约 YAML |
| `provenance/source_manifest.json` | 发布文件清单及 SHA256；不包含安装环境和实验输出 |
| `runs/`、`reports/` | 本地训练和评估输出，由命令生成，不纳入 Git 提交 |
| `docs/` | 课前清单、实验手册、本说明和故障排查 |

## 教学起点与模型结构

教学起点来自真实早期独立训练 CP550，有持续前后摆动和行走横移，供仿真分析与优化。它不是本次可直接上机的默认演示模型。教师另持 CP855 的部署演示模型，两份模型身份不可互换。

CP550 的普通 Actor 实际取历史末尾 5 帧（220 项）作为网络特征；输入接口仍接收完整 21 帧（924 项）。教师 CP855 演示模型还使用 21 帧的 HIM 感知编码器。查看检查点 `actor_config` 可以确认模型结构，不能只根据 ONNX 输入长度猜算法。

## 控制与 ONNX 接口

模型的 ONNX 接口是 float32 `[1,924] → [1,11]`，由 21 帧、每帧 44 项组成，最旧帧在前，启动历史补零。前 10 项用于关节动作，第 11 项在当前控制程序中被忽略。真实位置、真实速度和朝向误差不作为 Actor 输入；仿真可用它们算奖励、Critic 和评估指标。

`play pd` 是零动作 PD 诊断，默认位置增益乘 4；`play stand` 加载 RL 模型、输入零速度命令，使用机器人控制参数。二者对应不同测试。`play` 在 MuJoCo 中回放，`train` 和 `evaluate` 在 MotrixLab/MotrixSim 中执行，报告记录各自引擎。

## 验证范围与来源

仿真评估、导出校验和真机验证是三个不同的证据。教学起点供课堂分析，新课堂候选的仿真结果不能作为真机验证结论；CP855 的历史真机反馈也不能转用到其他模型。

候选通过课堂检查后交教师组织实机验证，按核对过的控制程序和参数加载，保留默认模型备份与回退方法。完整来源和许可见 [NOTICE](../NOTICE.md)。
