# 基于 SSVEP 的脑控机械臂系统（大创项目）

本项目实现了一套基于 **稳态视觉诱发电位（SSVEP）** 的脑机接口系统，通过脑电识别用户意图，控制机械臂完成**杯子定位、抓取与搬运**任务。

## 系统流程

```
SSVEP 视觉刺激 → 脑电采集(蓝牙 BLE) → FBCCA 频率识别 → 写入 eeg_trigger.json
                                                              ↓
杯子检测(YOLOv8) + 机械臂抓取 ← 读取 trigger 指令 ← 机械臂控制脚本
```

## 功能模块

| 文件 | 功能 |
|------|------|
| `ssvep_stim.py` | SSVEP 视觉刺激界面（PyQt5）+ BLE 脑电采集 + FBCCA 实时分类 |
| `robot_arm_control.py` | 读取脑电指令 + YOLO 杯子检测与标定 + HTTP 控制机械臂 |
| `cup_detect.py` | 独立的杯子检测与标定模块（YOLO + 卡尔曼滤波 + 椭圆拟合） |

## 脑电指令 → 机械臂动作映射

| 刺激频率 | 意图 | 动作 |
|---------|------|------|
| 10 Hz | 扫描 | 精确扫描杯子位置 |
| 11 Hz | 抓取 | 抓取杯子 |
| 12 Hz | 左边 | 机械臂左移 |
| 13 Hz | 右边 | 机械臂右移 |
| 14 / 15 Hz | 复位 | 机械臂复位 |

## 技术栈

`Python` · `PyQt5` · `FBCCA` · `BLE (bleak)` · `YOLOv8 (ultralytics)` · `OpenCV` · `scikit-learn` · `HTTP`

## 快速开始

### 环境要求

- Python 3.10+
- 摄像头（用于杯子检测）
- 支持 BLE 的脑电采集设备
- 机械臂（通过局域网 HTTP 控制）

### 安装依赖

```bash
pip install -r requirements.txt
```

### 运行

1. 启动 SSVEP 刺激与识别界面：

```bash
python ssvep_stim.py
```

2. 启动机械臂控制脚本（需先配置机械臂 IP，见 `robot_arm_control.py` 中的 `ARM_IP`）：

```bash
python robot_arm_control.py
```

3. （可选）单独运行杯子检测标定：

```bash
python cup_detect.py
```

> 说明：`yolov8s.pt`（YOLO 权重）已随仓库提供；`eeg_trigger.json` 为运行时自动生成的中间文件。

## 关键实现

- **FBCCA 分类**：5 子带滤波器组 + 谐波增强（含亚谐波）的典型相关分析，实时识别注视频率
- **杯子检测**：YOLOv8 目标检测 + 卡尔曼平滑 + 杯口椭圆拟合，解算抓取坐标
- **脑电采集**：BLE 蓝牙串流，异步解析脑电数据并送入分类器
