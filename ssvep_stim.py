import sys
import re
import time
import random
import os
#


import json
from collections import deque
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QLabel, QFrame, QPushButton,
    QHBoxLayout, QGridLayout
)
from PyQt5.QtCore import QThread, pyqtSignal, QTimer, Qt
from PyQt5.QtGui import QPainter, QPen, QColor
import math
import asyncio
from bleak import BleakClient
from scipy.linalg import svd
from scipy import signal

# -------------------- 核心配置 --------------------
DEVICE_ADDRESS   = "D8:3B:DA:74:7F:69"
CHAR_NOTIFY_UUID = "12345678-1234-1234-1234-123456789abd"
CHAR_WRITE_UUID  = "12345678-1234-1234-1234-123456789abe"

ORG_FS = 250
ACTIVE_CHANNELS = [0, 1, 4, 5]

FREQ_NAME_MAP = {
    10.0: "扫描",
    11.0: "抓取",
    12.0: "左边",
    13.0: "右边",
    14.0: "复位",
    15.0: "复位"
}
VALID_FREQS = list(FREQ_NAME_MAP.keys())

STIM_DURATION = 5.0
REFRESH_INTERVAL = 16
ANALYSIS_START_DELAY = 1.0
POST_STIM_EXTRA = 0.5
NUM_HARMS = 5
NUM_FBS = 5
USE_SUB_HARMONIC = True
CCA_LAMBDA = 0.1
COUNTDOWN = 3
STIM_SIZE_RATIO = 0.35

BG_COLOR_METABCI = [77, 77, 77]
INITIAL_BRIGHTNESS = 120
MIN_BRIGHTNESS = 90
MAX_BRIGHTNESS = 255
COLOR_SUCCESS = QColor(0, 255, 0)
COLOR_FAILED = QColor(255, 0, 0)
COLOR_NORMAL = QColor(200, 200, 200)

INNER_CUBE_OFFSET = {
    0: (-0.3, -0.3),
    1: (0.0, -0.3),
    2: (0.3, -0.3),
    3: (-0.3, 0.3),
    4: (0.0, 0.3),
    5: (0.3, 0.3)
}

ELLIPSE_WIDTH = 20
ELLIPSE_HEIGHT = 12

# 触发文件路径：与本脚本同目录，供 robot_arm_control.py 读取
TRIGGER_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "eeg_trigger.json")


# ★ 新增：写入触发文件函数
def write_trigger(freq: float, func_name: str):
    """
    将脑电识别结果写入共享队列文件
    robot_arm_control.py 每100ms读取一次此文件并执行对应机械臂动作
    """
    entry = {
        "freq": freq,
        "func": func_name,
        "time": time.time()
    }
    try:
        with open(TRIGGER_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        print(f"✅ 触发指令已写入: {freq}Hz ({func_name})")
    except Exception as e:
        print(f"❌ 写入触发文件失败: {e}")


# -------------------- 自定义3D立方体线框控件（完全不变）--------------------
# 自定义3D立方体线框控件，继承PyQt5的QFrame基础容器控件
# 每个SSVEP视觉刺激方块都是该类的一个实例，支持亮度调节、结果变色、3D绘制
class CubeWireframe(QFrame):
    # 构造函数：创建对象时自动执行，初始化实例属性
    # block_index：方块索引(0-5)，区分6个不同位置的立方体；parent：父控件，用于界面布局
    def __init__(self, block_index=0, parent=None):
        # 调用父类QFrame的构造函数，固定写法，初始化父类属性
        super().__init__(parent)
        # 初始化立方体亮度，使用全局配置的初始亮度值
        self.brightness = INITIAL_BRIGHTNESS
        # 初始化识别结果状态：0=正常/未识别 1=识别成功 2=识别失败
        self.result_state = 0
        # 保存当前立方体的索引，用于获取位置偏移、区分不同方块
        self.block_index = block_index
        # 设置控件背景透明，避免遮挡其他控件/背景，让6个立方体可重叠摆放
        self.setStyleSheet("background:transparent;")
    # 设置立方体亮度的方法，外部调用以实现闪烁效果
    # brightness：传入的目标亮度值(0-255)
    def set_brightness(self, brightness):
        # 保证亮度不低于全局配置的最小亮度，防止过暗看不见
        self.brightness = max(MIN_BRIGHTNESS, brightness)
        # 强制刷新控件，触发paintEvent重绘，让亮度变化立即显示
        self.update()

    # 设置识别结果状态的方法，外部调用以实现成功/失败变色
    # state：结果状态值(0/1/2)
    def set_result(self, state):
        # 保存结果状态
        self.result_state = state
        # 强制刷新控件，触发paintEvent重绘，让颜色变化立即显示
        self.update()

    # PyQt5控件绘制核心方法：自动调用（创建/刷新时），所有绘制逻辑都写在这里
    # event：绘制事件，包含绘制区域等信息，无需手动传参
    def paintEvent(self, event):
        # 先调用父类的paintEvent，保证基础绘制逻辑执行，固定写法
        super().paintEvent(event)
        # 创建画笔对象，用于绘制线条/图形，是PyQt5绘图的核心工具
        painter = QPainter(self)
        # 开启抗锯齿模式，让绘制的线条更平滑，避免锯齿状边缘，提升视觉效果
        painter.setRenderHint(QPainter.Antialiasing)

        # 获取当前控件的宽、高（像素），用于计算立方体的位置和大小
        w = self.width()
        h = self.height()
        # 计算控件的中心坐标(x/y)，让立方体绘制在控件正中间
        center_x = w // 2
        center_y = h // 2

        # 计算外立方体的整体大小：取宽高的最小值*0.4，保证立方体为正比例，不拉伸
        outer_size = min(w, h) * 0.4
        # 计算内立方体的整体大小：取宽高的最小值*0.25，比外立方体小，形成双层效果
        inner_size = min(w, h) * 0.25

        # 根据当前立方体索引，获取全局配置的位置偏移量(dx,dy)，默认(0,0)
        # 让6个立方体在界面上呈现不同位置（左上/中上/右上/左下/中下/右下）
        dx, dy = INNER_CUBE_OFFSET.get(self.block_index, (0, 0))
        # 将偏移比例转换为实际像素值，基于外立方体大小做适配，保证偏移比例统一
        offset_pixel_x = dx * (outer_size / 2)
        offset_pixel_y = dy * (outer_size / 2)

        # 内部函数：根据中心坐标、立方体大小，计算3D立方体的8个顶点坐标
        # cx/cy：立方体中心坐标；size：立方体整体大小
        def get_cube_vertices(cx, cy, size):
            # 计算立方体半长，用于确定顶点与中心的距离
            offset = int(size // 2)
            # 计算z轴偏移量（模拟3D效果的倾斜值），让立方体有立体视觉，不是纯平面
            z_offset = int(size // 3)
            # 转换为整数，避免像素坐标出现小数，导致绘制模糊
            cx = int(cx)
            cy = int(cy)
            # 返回8个顶点的(x,y)坐标：前4个是底面顶点，后4个是顶面顶点（带z轴偏移）
            return [
                (cx - offset, cy - offset), (cx + offset, cy - offset),  # 底面左上、右上
                (cx + offset, cy + offset), (cx - offset, cy + offset),  # 底面右下、左下
                (cx - offset + z_offset, cy - offset - z_offset),  # 顶面左上（3D偏移）
                (cx + offset + z_offset, cy - offset - z_offset),  # 顶面右上（3D偏移）
                (cx + offset + z_offset, cy + offset - z_offset),  # 顶面右下（3D偏移）
                (cx - offset + z_offset, cy + offset - z_offset),  # 顶面左下（3D偏移）
            ]

        # 生成外立方体的8个顶点坐标，中心为控件正中心
        outer_verts = get_cube_vertices(center_x, center_y, outer_size)
        # 生成内立方体的8个顶点坐标，中心带偏移量，与外立方体错开，形成双层3D效果
        inner_verts = get_cube_vertices(center_x + offset_pixel_x, center_y + offset_pixel_y, inner_size)

        # 计算内立方体的中心坐标（用于绘制中心椭圆）：8个顶点x/y坐标取平均值
        inner_center_x = sum(v[0] for v in inner_verts) / 8
        inner_center_y = sum(v[1] for v in inner_verts) / 8

        # 创建画笔样式对象，用于设置线条宽度、颜色
        pen = QPen()
        # 设置线条宽度为5像素，让立方体线条清晰可见
        pen.setWidth(5)
        # 根据结果状态设置画笔颜色
        if self.result_state == 1:
            # 状态1=识别成功，使用全局配置的绿色
            pen.setColor(COLOR_SUCCESS)
        elif self.result_state == 2:
            # 状态2=识别失败，使用全局配置的红色
            pen.setColor(COLOR_FAILED)
        else:
            # 状态0=正常/闪烁，使用当前亮度的灰色（RGB值相同为灰色，值越高越亮）
            pen.setColor(QColor(self.brightness, self.brightness, self.brightness))
        # 将设置好的画笔样式应用到绘图工具中，后续绘制均使用该样式
        painter.setPen(pen)

        # 内部函数：根据传入的8个顶点坐标，绘制立方体的12条线（构成3D线框）
        # verts：get_cube_vertices返回的8个顶点坐标列表
        def draw_cube(verts):
            # 绘制底面4条线：左上→右上→右下→左下→左上
            painter.drawLine(int(verts[0][0]), int(verts[0][1]), int(verts[1][0]), int(verts[1][1]))
            painter.drawLine(int(verts[1][0]), int(verts[1][1]), int(verts[2][0]), int(verts[2][1]))
            painter.drawLine(int(verts[2][0]), int(verts[2][1]), int(verts[3][0]), int(verts[3][1]))
            painter.drawLine(int(verts[3][0]), int(verts[3][1]), int(verts[0][0]), int(verts[0][1]))
            # 绘制顶面4条线：左上→右上→右下→左下→左上（带3D偏移）
            painter.drawLine(int(verts[4][0]), int(verts[4][1]), int(verts[5][0]), int(verts[5][1]))
            painter.drawLine(int(verts[5][0]), int(verts[5][1]), int(verts[6][0]), int(verts[6][1]))
            painter.drawLine(int(verts[6][0]), int(verts[6][1]), int(verts[7][0]), int(verts[7][1]))
            painter.drawLine(int(verts[7][0]), int(verts[7][1]), int(verts[4][0]), int(verts[4][1]))
            # 绘制上下底面连接的4条竖线，形成3D立方体的侧面
            painter.drawLine(int(verts[0][0]), int(verts[0][1]), int(verts[4][0]), int(verts[4][1]))
            painter.drawLine(int(verts[1][0]), int(verts[1][1]), int(verts[5][0]), int(verts[5][1]))
            painter.drawLine(int(verts[2][0]), int(verts[2][1]), int(verts[6][0]), int(verts[6][1]))
            painter.drawLine(int(verts[3][0]), int(verts[3][1]), int(verts[7][0]), int(verts[7][1]))

        # 绘制外立方体线框
        draw_cube(outer_verts)
        # 绘制内立方体线框，形成双层3D效果
        draw_cube(inner_verts)

        # 重新创建画笔，用于绘制中心椭圆：继承当前主颜色，调整线宽为1像素（更细）
        cp = QPen(pen.color())
        cp.setWidth(1)
        painter.setPen(cp)
        # 设置画刷颜色为当前主颜色，让椭圆填充为实心
        painter.setBrush(pen.color())
        # 绘制内立方体中心的椭圆：基于内立方体中心坐标，向左上偏移椭圆半宽/半高，保证椭圆居中
        # 椭圆大小使用全局配置的ELLIPSE_WIDTH（宽）和ELLIPSE_HEIGHT（高）
        painter.drawEllipse(int(inner_center_x - ELLIPSE_WIDTH / 2), int(inner_center_y - ELLIPSE_HEIGHT / 2),
                            ELLIPSE_WIDTH, ELLIPSE_HEIGHT)

# -------------------- FBCCA核心识别算法（完全不变）--------------------
def filterbank(data, fs, num_fbs=5):
    bands = [(6, 18), (14, 34), (30, 48), (46, 62), (58, 72)]
    if data.ndim == 1:
        data = data[np.newaxis, :]
    n_ch, n_s = data.shape
    fb_out = np.zeros((num_fbs, n_ch, n_s))
    for i, (low, high) in enumerate(bands[:num_fbs]):
        try:
            b, a = signal.butter(6, [low / (fs / 2), high / (fs / 2)], btype='band')
            fb_out[i] = signal.filtfilt(b, a, data, axis=-1)
        except:
            fb_out[i] = data
    return fb_out


def cca_reference(list_freqs, fs, num_smpls, num_harms=5, use_sub_harmonic=True):
    tidx = np.arange(0, num_smpls) / fs
    y_ref = []
    for freq in list_freqs:
        ref = []
        for harm in range(1, num_harms + 1):
            ref.append(np.sin(2 * np.pi * tidx * harm * freq))
            ref.append(np.cos(2 * np.pi * tidx * harm * freq))
        if use_sub_harmonic and freq >= 8:
            ref.append(np.sin(2 * np.pi * tidx * 0.5 * freq))
            ref.append(np.cos(2 * np.pi * tidx * 0.5 * freq))
        y_ref.append(np.array(ref))
    return np.array(y_ref)


def my_cca(input_x, input_y, lambda_=0.1):
    p = []
    x = input_x - np.mean(input_x, axis=1, keepdims=True)
    cov_xx = np.dot(x, x.T) + lambda_ * np.eye(x.shape[0])
    for y_t in input_y:
        try:
            y = y_t - np.mean(y_t, axis=1, keepdims=True)
            cov_yy = np.dot(y, y.T) + lambda_ * np.eye(y.shape[0])
            cov_xy = np.dot(x, y.T)
            inv_sqrt_xx = np.linalg.inv(np.sqrt(cov_xx))
            inv_sqrt_yy = np.linalg.inv(np.sqrt(cov_yy))
            t_mat = inv_sqrt_xx @ cov_xy @ inv_sqrt_yy
            s = svd(t_mat, compute_uv=False)
            p.append(s[0] if len(s) else 0)
        except:
            p.append(0)
    return np.array(p)


def fbcca_realtime(data, list_freqs, org_fs, num_harms=5, num_fbs=5, lambda_=0.1):
    n_ch, n_samp = data.shape
    win_len = int(org_fs * 0.2)
    if n_samp < win_len:
        return 0, 0.5
    step = int(org_fs * 0.1)
    votes = np.zeros(len(list_freqs))
    confs = []
    for start in range(0, n_samp - win_len + 1, step):
        win_data = data[:, start:start + win_len]
        fbdata = filterbank(win_data, org_fs, num_fbs)
        fb_coefs = [i ** -1.25 + 0.25 for i in range(1, num_fbs + 1)]
        y_ref = cca_reference(list_freqs, org_fs, win_len, num_harms, USE_SUB_HARMONIC)
        r = np.zeros((num_fbs, len(list_freqs)))
        for fb_i in range(num_fbs):
            r[fb_i] = my_cca(fbdata[fb_i], y_ref, lambda_)
        rho = np.dot(fb_coefs, r)
        best_idx = np.argmax(rho)
        votes[best_idx] += 1
        confs.append(abs(rho[best_idx]))
    recog_idx = np.argmax(votes)
    avg_conf = np.mean(confs) if len(confs) > 0 else 0.5
    return recog_idx, float(avg_conf)


# -------------------- BLE脑电采集线程（完全不变）--------------------
class BLEThread(QThread):
    data_signal = pyqtSignal(np.ndarray)
    status_signal = pyqtSignal(str)

    def __init__(self, device_address, notify_uuid, write_uuid, active_channels):
        super().__init__()
        self.device_address = device_address
        self.notify_uuid = notify_uuid
        self.write_uuid = write_uuid
        self.active_channels = active_channels
        self.running = False
        self.client = None
        self.loop = None
        self.data_pattern = re.compile(r'-?\d+\.?\d*')
        self._data_buffer = ""

    def _parse_line(self, line: str):
        if not line.startswith("Channel:"):
            return
        num_strs = self.data_pattern.findall(line)
        if len(num_strs) < max(self.active_channels) + 1:
            return
        vals = []
        for i in self.active_channels:
            try:
                vals.append(float(num_strs[i]))
            except (IndexError, ValueError):
                return
        if vals:
            self.data_signal.emit(np.array(vals, dtype=float))

    def _notification_handler(self, sender, data: bytearray):
        text = data.decode("utf-8", errors="ignore")
        self._data_buffer += text
        while "\n" in self._data_buffer:
            line, self._data_buffer = self._data_buffer.split("\n", 1)
            line = line.strip()
            if line:
                self._parse_line(line)

    async def _run_ble(self):
        self.status_signal.emit("正在连接BLE设备...")
        try:
            async with BleakClient(self.device_address) as client:
                self.client = client
                self.status_signal.emit("✅ BLE连接成功，开始接收数据")
                print(f"BLE已连接: {self.device_address}")
                await client.write_gatt_char(self.write_uuid, b"1")
                await client.start_notify(self.notify_uuid, self._notification_handler)
                while self.running:
                    await asyncio.sleep(0.01)
                await client.stop_notify(self.notify_uuid)
        except Exception as e:
            import traceback
            self.status_signal.emit(f"❌ BLE错误: {type(e).__name__}: {e}")
            print(f"BLE完整错误:\n{traceback.format_exc()}")

    def run(self):
        self.running = True
        if sys.platform == "win32":
            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        self.loop.run_until_complete(self._run_ble())

    def stop(self):
        self.running = False
        self.wait(1000)


# -------------------- 主界面（完全不变，仅run_recog末尾加一行）--------------------
class SSVEPRecognitionUI(QWidget):
    def __init__(self):
        super().__init__()
        self.is_fullscreen = True
        self.total_tests = 0
        self.success_count = 0
        self.target_idx = -1
        self.init_ui()
        self.init_data_buffers()
        self.init_timers()
        self.init_ble()

        self.pdm_integrator = np.zeros(len(VALID_FREQS), dtype=np.float32)
        self.last_brightness = np.full(len(VALID_FREQS), MIN_BRIGHTNESS)
        self.sample_counter = np.zeros(len(VALID_FREQS), dtype=int)

    def init_ui(self):
        self.setWindowTitle('SSVEP脑电识别系统 - 优化版（谐波增强+滑动FBCCA）')
        self.setStyleSheet(f'background-color: rgb({BG_COLOR_METABCI[0]},{BG_COLOR_METABCI[1]},{BG_COLOR_METABCI[2]});')

        self.main_layout = QVBoxLayout(self)
        self.main_layout.setContentsMargins(10, 10, 10, 10)
        self.main_layout.setSpacing(8)

        self.target_label = QLabel('目标：未设置', alignment=Qt.AlignCenter)
        self.target_label.setStyleSheet('color: yellow; font-size: 24px; font-weight: bold;')
        self.main_layout.addWidget(self.target_label)

        self.status_label = QLabel('状态：等待启动', alignment=Qt.AlignCenter)
        self.status_label.setStyleSheet('color: white; font-size: 18px;')
        self.main_layout.addWidget(self.status_label)

        ctrl_layout = QHBoxLayout()
        self.start_btn = QPushButton('开始测试')
        self.start_btn.clicked.connect(self.start_countdown)
        ctrl_layout.addWidget(self.start_btn)

        self.window_btn = QPushButton('切换模式')
        self.window_btn.clicked.connect(self.toggle_window_mode)
        ctrl_layout.addWidget(self.window_btn)
        self.main_layout.addLayout(ctrl_layout)

        self.stat_label = QLabel('次数：0 | 成功：0 | 正确率：0.0%', alignment=Qt.AlignCenter)
        self.stat_label.setStyleSheet('color: cyan; font-size: 19px;')
        self.main_layout.addWidget(self.stat_label)

        self.grid_layout = QGridLayout()
        self.grid_layout.setHorizontalSpacing(60)
        self.grid_layout.setVerticalSpacing(100)
        self.grid_layout.setContentsMargins(10, 10, 10, 10)
        self.main_layout.addLayout(self.grid_layout)

        self.result_label = QLabel('识别结果：无', alignment=Qt.AlignCenter)
        self.result_label.setStyleSheet('color: white; font-size: 18px;')
        self.main_layout.addWidget(self.result_label)

        self.conf_label = QLabel('置信度：0.000', alignment=Qt.AlignCenter)
        self.conf_label.setStyleSheet('color: #ccc; font-size: 17px;')
        self.main_layout.addWidget(self.conf_label)

        self.stim_blocks = []
        self.stim_labels = []
        screen = QApplication.primaryScreen().size()
        self.stim_size = int(min(screen.width(), screen.height()) * STIM_SIZE_RATIO)

        for idx, freq in enumerate(VALID_FREQS):
            container = QWidget()
            v_layout = QVBoxLayout(container)
            v_layout.setContentsMargins(0, 0, 0, 0)
            v_layout.setSpacing(4)

            name_label = QLabel(f"{freq:.1f}Hz: {FREQ_NAME_MAP[freq]}", alignment=Qt.AlignCenter)
            name_label.setStyleSheet('color: white; font-size: 15px; font-weight: bold;')
            v_layout.addWidget(name_label)
            self.stim_labels.append(name_label)

            block = CubeWireframe(block_index=idx)
            block.setFixedSize(self.stim_size, self.stim_size)
            v_layout.addWidget(block)
            self.stim_blocks.append(block)

            row = idx // 3
            col = idx % 3
            self.grid_layout.addWidget(container, row, col, alignment=Qt.AlignCenter)

        self.showNormal()
        self.resize(1100, 700)
        self.update_stats()

    def update_stats(self):
        acc = 0.0 if self.total_tests == 0 else self.success_count / self.total_tests * 100
        self.stat_label.setText(f'次数：{self.total_tests} | 成功：{self.success_count} | 正确率：{acc:.1f}%')

    def reset_block_states(self):
        for b in self.stim_blocks:
            b.set_result(0)

    def init_data_buffers(self):
        self.global_buf = deque(maxlen=ORG_FS * 10)
        self.stim_buf = deque()
        self.is_stim_running = False
        self.stim_start = None

    def init_timers(self):
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_stim)
        self.timer.setInterval(REFRESH_INTERVAL)

    def init_ble(self):
        self.ble_thread = BLEThread(
            DEVICE_ADDRESS, CHAR_NOTIFY_UUID, CHAR_WRITE_UUID, ACTIVE_CHANNELS
        )
        self.ble_thread.data_signal.connect(self.on_data)
        self.ble_thread.status_signal.connect(self.status_label.setText)
        try:
            self.ble_thread.start()
        except Exception as e:
            print(f"启动BLE线程失败: {e}")
            self.status_label.setText(f'错误：{e}')

    def toggle_window_mode(self):
        if self.is_fullscreen:
            self.showNormal()
            self.resize(1100, 700)
            self.window_btn.setText("切换全屏")
        else:
            self.showFullScreen()
            self.window_btn.setText("切换窗口")
        self.is_fullscreen = not self.is_fullscreen

    def on_data(self, vec):
        t = time.perf_counter()
        self.global_buf.append((t, vec))
        if self.is_stim_running:
            self.stim_buf.append((t, vec))

    def start_countdown(self):
        self.reset_block_states()
        self.result_label.setText('识别结果：无')
        self.conf_label.setText('置信度：0.000')
        self.start_btn.setEnabled(False)

        self.target_idx = random.randint(0, len(VALID_FREQS) - 1)
        self.target_label.setText(f'请注视：【 {self.target_idx + 1} 】')

        self.cd = COUNTDOWN
        self.status_label.setText(f'倒计时：{self.cd}')
        self.cdt = QTimer(self)

        def cdtick():
            self.cd -= 1
            if self.cd <= 0:
                self.cdt.stop()
                self.start_stim()
            else:
                self.status_label.setText(f'倒计时：{self.cd}')

        self.cdt.timeout.connect(cdtick)
        self.cdt.start(1000)

    def start_stim(self):
        self.stim_buf.clear()
        self.is_stim_running = True
        self.stim_start = time.perf_counter()
        self.pdm_integrator[:] = 0
        self.sample_counter[:] = 0
        self.timer.start()
        self.status_label.setText('采集中...')

    def update_stim(self):
        if not self.is_stim_running:
            return
        elapsed = time.perf_counter() - self.stim_start_pc_time
        if elapsed >= STIM_DURATION:
            self.timer.stop()
            self.is_stim_running = False
            self.status_label.setText('识别中...')
            QTimer.singleShot(int(POST_STIM_EXTRA * 1000), self.run_recog)
            return

        for i, freq in enumerate(VALID_FREQS):
            # 方波：一半时间最亮，一半时间最暗
            if math.sin(2 * math.pi * freq * elapsed) > 0:
                br = MAX_BRIGHTNESS
            else:
                br = MIN_BRIGHTNESS
            self.stim_blocks[i].set_brightness(int(br))

    def run_recog(self):
        try:
            if len(self.stim_buf) < 100:
                self.result_label.setText("⚠️ 数据太少，识别无效")
                self.status_label.setText("数据不足")
                self.start_btn.setEnabled(True)
                return

            ts = np.array([t for t, _ in self.stim_buf])
            mat = np.array([d for _, d in self.stim_buf]).T
            t0 = self.stim_start
            m = (ts >= t0 + ANALYSIS_START_DELAY) & (ts <= t0 + STIM_DURATION + POST_STIM_EXTRA)
            eeg = mat[:, m]

            if eeg.shape[1] < 50:
                self.result_label.setText("⚠️ 有效数据太短")
                self.status_label.setText("数据异常")
                self.start_btn.setEnabled(True)
                return

            eeg = eeg - np.mean(eeg, axis=1, keepdims=True)
            try:
                b, a = signal.iirnotch(50, 30, ORG_FS)
                eeg = signal.filtfilt(b, a, eeg, axis=1)
            except:
                pass
            eeg = signal.detrend(eeg, axis=1)

            recog_idx, conf = fbcca_realtime(eeg, VALID_FREQS, ORG_FS, NUM_HARMS, NUM_FBS, CCA_LAMBDA)
            f = VALID_FREQS[recog_idx]
            func_name = FREQ_NAME_MAP.get(f, "unknown")

            self.total_tests += 1
            success = (recog_idx == self.target_idx)
            if success:
                self.success_count += 1

            self.reset_block_states()
            if success:
                self.stim_blocks[self.target_idx].set_result(1)
                self.result_label.setText(f'✅ 识别正确：{f:.1f}Hz')
            else:
                self.stim_blocks[recog_idx].set_result(2)
                self.result_label.setText(f'❌ 识别错误：{f:.1f}Hz')

            self.conf_label.setText(f'置信度：{conf:.3f}')
            self.status_label.setText('完成')
            self.update_stats()

            # ★ 新增：将识别结果写入共享触发文件，通知机械臂控制程序执行动作
            write_trigger(f, func_name)

        except Exception as e:
            print("错误原因：", e)
            self.status_label.setText('识别失败')
            self.result_label.setText('识别结果：异常')
        self.start_btn.setEnabled(True)

    def keyPressEvent(self, e):
        if e.key() == Qt.Key_Escape:
            self.close()
        elif e.key() in (Qt.Key_Return, Qt.Key_Enter):
            if not self.is_stim_running and self.start_btn.isEnabled():
                self.start_countdown()

    def closeEvent(self, e):
        self.ble_thread.stop()
        e.accept()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    w = SSVEPRecognitionUI()
    w.show()
    sys.exit(app.exec_())
