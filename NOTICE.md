# 来源与许可

本学生仓库由当前 Kudada/MotrixLab 工作区提取，仅保留两天站立优化课程运行所需的源码、机器人资源、控制契约和模型。

- 原始 Kudada：commit `b8a8e759473452f93afbd69a08f759a3e9af8964`。机器人资源和控制/观测设计的许可文件在 `licenses/kudada.txt`、`licenses/legged_gym.txt`、`licenses/rsl_rl.txt`。
- MotrixLab：commit `069b1bc4aa4f758786469009e38016518c21af70`，Apache-2.0。仅 vendor `motrix_env_core` 和 `motrix_env_motrixsim`，保留原版权头，完整许可在 `licenses/MotrixLab.txt`。当前上游这两个包的版本号为 `0.4.0b0`；仿真器二进制 `motrixsim` / `motrixsim-core` 为 `0.10.1`，两者不能混淆。
- `scratch_policy` 的 HIM Actor/独立 PPO 环境来自本工作区的独立策略研发，不装载厂商 ONNX，也不从厂商网络生成目标动作。
- `references/tinker/*.yaml` 是用户提供的机器人控制参数副本，用于保持部署接口。此文件不授予用户提供厂商资料的额外许可。
- `course` 是本次课程适配：精简训练与命令接口、配对评估、MuJoCo 可视化、固定 IO 导出。课堂热启动方式和原研究恢复方式的区别见学生手册。

模型与制备记录由教师发布清单绑定到 SHA256。ONNX 导出只证明接口和数值一致；当前一次课程候选的仿真或过去其他模型的硬件成功，不自动证明新候选可部署。

第三方依赖包保留各自许可。分发本仓库时保留 `LICENSE`、`licenses`、本文件和代码版权头。`provenance/source_manifest.json` 记录导入资源的来源、相对路径和哈希；它与最终发布完整性清单用途不同。
