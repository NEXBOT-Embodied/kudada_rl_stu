# 故障排查

|现象|检查与处理|
|---|---|
|启动脚本找不到 Python|运行 `install.ps1`（Linux 为 `bash install.sh`），检查本仓库 `.venv` 是否存在。|
|提示 `Install uv`|按 README 安装 uv 后重开终端，或向安装脚本提供已有 64 位 Python 3.10 的路径来创建 `.venv`。|
|提示 `No Python at`、`Invalid .venv` 或环境不完整|基础 Python 路径失效或 `.venv` 从其他电脑复制。先把 `.venv` 改名为 `.venv-backup`，再重新运行安装脚本。|
|Windows 禁止运行 PS1|按所在组织允许的方式给当前 PowerShell 进程设置执行策略，例如 `Set-ExecutionPolicy -Scope Process Bypass`。|
|Python 版本不支持|使用 64 位 3.10.x，推荐 3.10.21；Motrix 源码包限制 `==3.10.*`。已有不兼容 `.venv` 应先改名备份再重建。|
|依赖下载失败|默认使用 USTC，可用 `.\install.ps1 -PackageIndex https://pypi.tuna.tsinghua.edu.cn/simple` 切换到清华；Linux 为 `bash install.sh '' https://pypi.tuna.tsinghua.edu.cn/simple`。|
|PyTorch 下载失败|默认使用南京大学 CPU 镜像，可用 `.\install.ps1 -TorchIndex https://mirror.sjtu.edu.cn/pytorch-wheels/cpu` 切换到上海交大；Linux 第三个参数指定该镜像地址。|
|uv 下载 Python 失败|默认使用南京大学 Python 镜像，可设置 `UV_PYTHON_INSTALL_MIRROR` 切换镜像，或提供已有 Python 3.10 路径来创建 `.venv`。|
|没有 CUDA|课堂小规模训练用 CPU，先确认 `torch` 可以导入。`doctor` 显示没有 CUDA 本身不是失败。|
|Motrix 原生库导入错误|重新运行安装脚本，检查 Python/操作系统位数和系统运行库。保存完整 `doctor` 输出给教师。|
|GUI 黑屏或窗口打不开|先运行 `play ... --headless --seconds 2`，区分物理/模型失败与显示失败。使用远程桌面时优先看教师录制视频。|
|窗口一打开机器人倒地|确认用 `play stand --checkpoint/--onnx`，不是 PD 诊断；检查加载路径和哈希、模型/控制参数版本，保留失败记录。|
|ONNX shape 错误|必须是 float32 `[1,924] → [1,11]`，使用本仓库 export，而不是给动态 batch 文件改后缀。|
|`compare` 拒绝比较|三份评估条件不同；按相同 seed/conditions/时长/模式重新评估。不要通过修改 JSON 绕过条件校验。|
|训练奖励上升但模型更差|检查物理评估、保护、失败和回归。奖励改动可能改变指标含义；同一奖励数值也不等于相同行为。|
|缩短测试后“通过”|缩短评估是 smoke，不能当完整课程验收。完整站立/行走/启停分别为60/30/40秒。|
|改进没有达到目标|如实记录 A/B/C 各条件。先检查假设、奖励生效门限、预算和副作用，再设计下一次单变量实验。|

学生仓库运行代码不引用相邻 `kudada_rl_motrix`。`doctor` 中 Python 和已安装包应来自本仓库 `.venv`，Motrix 环境源码安装自本仓库 `vendor/`。模型来源记录中历史路径不作为运行依赖。
