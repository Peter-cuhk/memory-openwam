# CPFS 原生环境：Wan 训练与推理服务

本环境用于当前基础镜像及其配套的 **PPU 训练节点**。复用镜像的
Python 3.12.3、PPU PyTorch 2.9.0、torchvision 0.24.0、torchaudio 2.9.0、
Triton 3.5.0+gitcc5446cf 和 CUDA API 13.0 工具链。
镜像包含 PPU SDK，`nvidia-smi` 使用 HGML；这些版本号不代表原生 NVIDIA 环境。

新增依赖只安装到 `/mnt/cpfs/feiyang/code/OpenWAM/.venv`，不修改基础镜像。
`.venv` 通过符号链接复用上述四个 Python 包的代码，并复制其发行元数据；
不启用整个系统的 `site-packages`，避免混入系统旧版 Transformers/DeepSpeed。
PyTorch 的动态库继续使用基础镜像已有的库搜索路径。

## 激活与重建

每次登录和每个训练/部署任务都先执行：

```bash
cd /mnt/cpfs/feiyang/code/OpenWAM
source scripts/env.sh
```

此脚本激活 `.venv`，将 pip、uv、Hugging Face、ModelScope、编译器及
W&B 缓存/临时目录指向 `/mnt/cpfs/feiyang/cache`；运行输出在项目 `outputs/`。
W&B 默认离线，需要在线记录时在激活前设置 `WANDB_MODE=online`。

环境已经存在时无需重装。需要在相同基础镜像中重建时运行：

```bash
bash scripts/setup_env.sh
```

脚本使用系统已有的 `uv` 和 Python，不下载 Python/CUDA。依赖版本记录在
`requirements/native.txt`，基础计算栈的版本约束由镜像生成到
`.venv/base-constraints.txt`。PPU DeepSpeed 依赖基础镜像配置的软件源，
重建时保留该源及其网络访问能力。

不要用 `uv sync` 或无约束升级来替换链接的 PyTorch 栈。
该 `.venv` 依赖相同基础镜像、相同挂载路径及已有动态库环境，不能作为独立
NVIDIA 环境移植。无需创建新镜像；任务挂载 `/mnt/cpfs/feiyang/` 和 `/mnt/oss`。

## 检查

本次验证：105 个包的依赖一致性检查通过；训练 `torchrun` 配置解析、
`openwam-serve`/`scripts/deploy.py` CLI 通过；实际 WebSocket 服务的
ping/reset/error 通信通过（模型初始化使用桩，不代表真实模型推理）。
现有测试 126 项直接通过，另 1 项 checkpoint 测试在本地临时目录复测通过。
日志和基础栈清单位于 `outputs/environment-*`。

CPFS 注意：`test_save_checkpoint_excludes_vlm_backbone` 在 safetensors
张量仍持有 mmap 时删除临时目录，会报 `Directory not empty`；读写和权重
断言已通过。单独运行该测试时可用
`TMPDIR=/tmp python -m pytest -q tests/test_openwam_trainer.py::test_save_checkpoint_excludes_vlm_backbone`。
这仅改变测试产生的微型临时 checkpoint 位置，不改变依赖、下载和编译缓存。

```bash
source scripts/env.sh
uv pip check --python .venv/bin/python
python scripts/check_env.py
python scripts/train.py --cfg job --resolve
openwam-serve --help
```

在分配了设备的 PPU 节点增加：

```bash
nvidia-smi
python scripts/check_env.py --gpu
```

`--gpu` 要求设备可用，并在每个可见设备执行 bf16 矩阵乘法和反向传播。
这不替代真实 Wan 权重推理、DeepSpeed 多卡通信和完整训练验证。
DeepSpeed 安装时使用 `DS_BUILD_OPS=0`，按需编译的算子会在任务节点构建，
缓存位于挂载盘。默认 attention 使用项目已有的 PyTorch 路径，未添加
FlashAttention/Cosmos/机器人仿真器依赖。

## 训练

先准备 Wan 权重和 RoboTwin 数据，下面的路径需要替换为实际路径。
项目默认 YAML 中的 `/path/to/...` 是占位符。

```bash
cd /mnt/cpfs/feiyang/code/OpenWAM
source scripts/env.sh
python scripts/check_env.py --gpu
NPROC_PER_NODE=8 bash scripts/train.sh \
  model/video_backbone=wan22_ti2v_5b \
  model.video_backbone.model_path=/mnt/oss/path/to/Wan2.2-TI2V-5B \
  dataloader=robotwin \
  dataloader.dataset_dir=/mnt/oss/path/to/robotwin_data
```

`NPROC_PER_NODE` 设为任务实际分配的设备数。多机额外设置 `NNODES`、
`NODE_RANK`、`MASTER_ADDR` 和 `MASTER_PORT`。首次验证可用单卡并追加
`training.debug=true training.batch_size=1 training.dataset_num_workers=0`。

## 推理部署

训练输出目录需包含 checkpoint 和对应的配置/部署资源：

```bash
cd /mnt/cpfs/feiyang/code/OpenWAM
source scripts/env.sh
python scripts/check_env.py --gpu
openwam-serve \
  --ckpt-dir /mnt/cpfs/feiyang/code/OpenWAM/outputs/path/to/checkpoint_dir \
  --device cuda:0 --host 0.0.0.0 --port 8848 --compile-enabled false
```

首次在 PPU 上使用 eager 模式完成真实推理验证后，再测试项目默认的
`--compile-enabled true`。服务使用 WebSocket，默认端口 8848。
多实例入口及 checkpoint 结构见 [训练与部署说明](train-and-deploy.md)。

本次配置范围为默认 Wan 的运行依赖，未下载模型权重、训练数据或部署
checkpoint；未启动常驻服务或训练任务。开发机未加载设备驱动，设备执行
和多卡通信需在 PPU 训练节点检查。
