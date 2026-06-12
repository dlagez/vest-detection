# 反光背心检测

基于 YOLO 的安全装备检测系统，支持图片、视频和实时摄像头的反光背心/安全帽检测与可视化。

## 功能

- **图片推理** — 单张图片检测，输出标注图 + JSON
- **视频推理** — 逐帧处理，输出标注视频 + 逐帧 JSON 结果
- **实时摄像头** — 摄像头画面实时检测，按 `q` 退出
- **模型训练** — 提供 YOLO 训练脚本
- **REST API** — FastAPI 接口服务

## 目录结构

```
vest-detection/
├── configs/              # 配置文件
│   └── default.yaml
├── weights/              # 模型权重
│   └── yolo11m_safety.pt
├── data/                 # 测试数据
│   ├── images/
│   └── videos/
├── outputs/              # 推理输出
│   ├── images/           # 标注结果图片
│   ├── videos/           # 标注结果视频
│   └── json/             # 检测数据 JSON
├── src/vest_detection/   # 核心代码
│   ├── config.py
│   ├── detector.py
│   ├── visualizer.py
│   ├── pipelines/        # 图片/视频/摄像头推理管线
│   ├── api/              # FastAPI 服务
│   └── utils/            # 工具函数
├── scripts/              # 可执行脚本
│   ├── infer_image.py
│   ├── infer_video.py
│   ├── infer_camera.py
│   └── train_yolo.py
└── tests/
```

## 快速开始

### 1. 安装

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

### 2. 视频推理

```bash
python scripts/infer_video.py \
  --input data/videos/video.mp4 \
  --model weights/yolo11m_safety.pt \
  --conf 0.35
```

### 3. 图片推理

```bash
python scripts/infer_image.py \
  --input data/images/test.jpg \
  --model weights/yolo11m_safety.pt
```

### 4. 摄像头实时检测

```bash
python scripts/infer_camera.py --camera 0 --model weights/yolo11m_safety.pt
```

按 `q` 退出。

### 5. 启动 API 服务

```bash
uvicorn src.vest_detection.api.main:app --reload --host 0.0.0.0 --port 8000
```

### 6. 训练模型

```bash
python scripts/train_yolo.py \
  --data data.yaml \
  --epochs 100 \
  --batch 16 \
  --model yolo11m.pt
```

## 配置

编辑 `configs/default.yaml`：

```yaml
model:
  path: "weights/yolo11m_safety.pt"   # 模型路径
  confidence: 0.35                     # 置信度阈值
  iou: 0.5                             # NMS IoU 阈值
  imgsz: 640                           # 输入尺寸

classes:
  vest: ["safety_vest", "reflective_vest", "vest"]
  no_vest: ["no_safety_vest", "no_vest"]

output:
  save_json: true                      # 是否保存 JSON
  save_annotated: true                 # 是否保存标注图

video:
  enable_tracking: true                # 视频目标跟踪
```

## 检测类别

| 类别 | 说明 |
|------|------|
| vest | 穿着反光背心 |
| novest | 未穿反光背心 |
| person | 人员检测 |
| hat | 佩戴安全帽 |
| nohat | 未戴安全帽 |

## 输出格式

JSON 输出示例（视频）：

```json
[
  {
    "frame": 0,
    "count": 4,
    "detections": [
      {
        "class_name": "vest",
        "confidence": 0.87,
        "bbox": [100, 200, 300, 500]
      }
    ]
  }
]
```

## 依赖

- ultralytics >= 8.x
- supervision >= 0.28
- opencv-python
- pyyaml
- fastapi + uvicorn（API 服务）

## 运行测试

```bash
python -m pytest tests/ -v
```
