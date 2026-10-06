import cv2
import numpy as np
import threading
import time
import os
import json
import requests

from ultralytics import YOLO
from collections import deque
from sklearn.linear_model import LinearRegression
from queue import Queue
# ==================== 频率 → 机械臂动作映射 ====================
FREQ_TO_ARM_CMD = {
    10.0: "SCAN",
    11.0: [
        '{"T": 1041, "x": -194.8066227, "y":3.885300056, "z": 16.36681748, "t": 1.475689518, "r":0.007669904, "g": 2.3}',
        '{"T": 1041, "x": -194.8066227, "y":3.885300056, "z": 16.36681748, "t": 1.475689518, "r":0.007669904, "g": 3.14}'
    ],
    12.0: [
        '{"T": 1041, "x": 219.6885415, "y": 211.0967891, "z": 48.28208358, "t": 1.036971013, "r":0.004601942, "g": 2.4}',
        '{"T": 1041, "x": 223.6845582, "y": 220.2793311, "z": -47.99468669, "t": 1.328427362, "r":0.004601942, "g": 2.3}',
        '{"T": 1041, "x": 223.6845582, "y": 220.2793311, "z": -47.99468669, "t": 1.328427362, "r":0.004601942, "g": 3.14}',
        '{"T": 1041, "x": 221.5298578, "y": 213.5206438, "z": 29.67438679, "t": 1.096796263, "r":0.004601942, "g": 3.14}',
        '{"T": 1041, "x": 348.1272432, "y": 0.534020922, "z": 213.7262062, "t": 0.007669904, "r":-0.016873789, "g": 3.141592654}'
    ],
    13.0: [
        '{"T": 1041, "x": 206.1116037, "y": -202.9738957, "z": 39.93113448, "t": 1.118271994, "r":0.003067962, "g": 2.3}',
        '{"T": 1041, "x": 201.7204837, "y": -206.0997711, "z": -55.55286232, "t": 1.443475921, "r":0.004601942, "g": 2.3}',
        '{"T": 1041, "x": 201.7204837, "y": -206.0997711, "z": -55.55286232, "t": 1.443475921, "r":0.004601942, "g": 3.14}',
        '{"T": 1041, "x": 204.6889202, "y": -209.1326514, "z": 15.60240973, "t": 1.201106957, "r":0.003067962, "g": 3.141592654}',
        '{"T": 1041, "x": 348.1272432, "y": 0.534020922, "z": 213.7262062, "t": 0.007669904, "r":-0.016873789, "g": 3.141592654}'
    ],
    14.0: [
        '{"T": 1041, "x": 348.1272432, "y": 0.534020922, "z": 213.7262062, "t": 0.007669904, "r":-0.016873789, "g": 3.141592654}'
    ],
    15.0: [
        '{"T": 1041, "x": 348.1272432, "y": 0.534020922, "z": 213.7262062, "t": 0.007669904, "r":-0.016873789, "g": 3.141592654}'
    ]
}

# ==================== 机械臂坐标线性回归模型 ====================
MEASURED_POINTS = np.array([
    [0, 0, -141.4200966, 188.2576348],
    [9, 0, -52.50078347, 188.5875597],
    [0, 9, -145.2925713, 101.3295243],
    [9, 9, -52.87733753, 98.20066506]
], dtype=np.float32)

real_xy = MEASURED_POINTS[:, :2]
arm_YX  = MEASURED_POINTS[:, 2:]

model_Y = LinearRegression()
model_Y.fit(real_xy[:, 0].reshape(-1, 1), arm_YX[:, 0])

model_X = LinearRegression()
model_X.fit(real_xy[:, 1].reshape(-1, 1), arm_YX[:, 1])

print(f"✅ 线性回归系数 - X→Y: 斜率={model_Y.coef_[0]:.6f}, 截距={model_Y.intercept_:.6f}")
print(f"✅ 线性回归系数 - Y→X: 斜率={model_X.coef_[0]:.6f}, 截距={model_X.intercept_:.6f}")

import traceback

# -------------------- 核心配置 --------------------
WINDOW_NAME = "Cup Detection & Calibration"
TRIGGER_QUEUE = Queue()
LOCK = threading.Lock()

ARM_IP = "192.168.142.46"
ARM_URL = f"http://{ARM_IP}/js"
ARM_TIMEOUT = 2.0

TRIGGER_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "eeg_trigger.json")
TRIGGER_MAX_AGE = 3.0

YOLO_MODEL_PATH = "yolov8s.pt"
VIDEO_SOURCE = 1
FRAME_WIDTH, FRAME_HEIGHT = 640, 480
FPS_TARGET = 30

# 工业级图像识别参数（与你给的版本完全一致）
MEM_WEIGHT = 0.1
BOTTOM_SMOOTH_LEN = 3
BOTTOM_MEM_WEIGHT = 0.1
TOP_SMOOTH_LEN = 3
KALMAN_PROCESS_NOISE = 1e-4
KALMAN_MEAS_NOISE = 1e-2

ELLIPSE_MIN_AREA = 50
ROI_UPPER_RATIO = 0.75
MIN_CONTOUR_POINTS = 5
FULL_ELLIPSE_FIT_ERROR_MAX = 0.25
FULL_ELLIPSE_COVERAGE_THRESH = 0.55
FULL_ELLIPSE_CENTER_OFFSET_MAX = 0.3
FULL_ELLIPSE_BOUNDING_BOX_RATIO = 0.4
ELLIPSE_MAJOR_AXIS_MIN_RATIO = 0.75

SMOOTH_LEN = 5
SMOOTH_ALPHA = 0.4
MISS_TIMEOUT = 0.3

debug_ellipse = False
real_points = np.array([[0,0],[9,0],[0,9],[9,9]], dtype=np.float32)

# -------------------- 全局状态 --------------------
class State:
    def __init__(self):
        self.origin = self.x_ref = self.y_ref = self.H = None
        self.calib_points = []
        self.calibrated = False
        self.calibration_prompted = False
        self.calib_step = 0
        self.plane_calib_count = 0
        self.mode = None

        # 平滑
        self.top_window = deque(maxlen=TOP_SMOOTH_LEN)
        self.bottom_window = deque(maxlen=BOTTOM_SMOOTH_LEN)
        self.prev_top = self.prev_bottom = None
        self.stable_window = deque(maxlen=15)
        self.second_buffer = deque(maxlen=30)
        self.last_second_time = time.time()

        # Kalman
        self.kalman = self._init_kalman()

        # 工业级椭圆平滑
        self.ellipse_buf = deque(maxlen=SMOOTH_LEN)
        self.smooth_ellipse = None
        self.last_detect_time = time.time()

        # YOLO 框平滑
        self.smooth_box = None
        self.box_alpha = 0.35

        # ROI 运动检测
        self.prev_roi = None

        # 预热扫描标记
        self.calib_done_time = None
        self.pre_scan_done = False

    def _init_kalman(self):
        kf = cv2.KalmanFilter(4, 2)
        kf.measurementMatrix = np.array([[1,0,0,0],[0,1,0,0]], np.float32)
        kf.transitionMatrix  = np.array([[1,0,1,0],[0,1,0,1],[0,0,1,0],[0,0,0,1]], np.float32)
        kf.processNoiseCov   = np.eye(4, dtype=np.float32) * KALMAN_PROCESS_NOISE
        kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * KALMAN_MEAS_NOISE
        kf.statePre = np.zeros((4,1), np.float32)
        return kf

state = State()

# -------------------- 工业级图像识别工具函数 --------------------
def limit_change(prev, new, max_delta):
    return prev + np.clip(new - prev, -max_delta, max_delta)

def smooth_yolo_box(state, x1, y1, x2, y2):
    new_box = np.array([x1, y1, x2, y2], dtype=float)
    if state.smooth_box is None:
        state.smooth_box = new_box
    else:
        state.smooth_box = (
            state.box_alpha * new_box +
            (1 - state.box_alpha) * state.smooth_box
        )
    return state.smooth_box.astype(int)

def is_full_ellipse(contour, ellipse):
    (cx, cy), (M, m), ang = ellipse
    a, b = M/2, m/2
    theta = np.radians(ang)
    pts = contour.reshape(-1,2).astype(np.float32)

    covered = 0
    errors = []

    for (px, py) in pts:
        x_rot = (px-cx)*np.cos(theta) + (py-cy)*np.sin(theta)
        y_rot = -(px-cx)*np.sin(theta) + (py-cy)*np.cos(theta)
        dist = abs((x_rot**2)/(a*a) + (y_rot**2)/(b*b) - 1)
        errors.append(dist)
        if dist <= FULL_ELLIPSE_FIT_ERROR_MAX:
            covered += 1

    coverage = covered / len(pts)
    avg_err = np.mean(errors)

    M_cnt = cv2.moments(contour)
    if M_cnt["m00"] == 0:
        return False, avg_err

    cnt_cx = M_cnt["m10"]/M_cnt["m00"]
    cnt_cy = M_cnt["m01"]/M_cnt["m00"]
    offset = np.sqrt((cnt_cx-cx)**2 + (cnt_cy-cy)**2)
    max_offset = min(a,b) * FULL_ELLIPSE_CENTER_OFFSET_MAX

    x,y,w,h = cv2.boundingRect(contour)
    bbox_area = w*h
    ellipse_area = np.pi*a*b
    bbox_ratio = ellipse_area/bbox_area if bbox_area>0 else 0

    if coverage>=FULL_ELLIPSE_COVERAGE_THRESH and offset<=max_offset and bbox_ratio>=FULL_ELLIPSE_BOUNDING_BOX_RATIO:
        return True, avg_err
    return False, avg_err

def detect_valid_ellipse_in_upper_roi(frame, x1, y1, x2, y2):
    h = y2 - y1
    upper_h = int(h * ROI_UPPER_RATIO)
    if upper_h <= 10:
        return None

    H, W = frame.shape[:2]
    x1c = max(0, min(x1, W-1))
    x2c = max(0, min(x2, W))
    y1c = max(0, min(y1, H-1))
    y2c = max(0, min(y1+upper_h, H))

    if x2c<=x1c or y2c<=y1c:
        return None

    roi = frame[y1c:y2c, x1c:x2c]
    if roi.size == 0:
        return None

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5,5), 1.2)
    edges = cv2.Canny(gray, 50, 150)

    contours,_ = cv2.findContours(edges, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        if len(cnt) < MIN_CONTOUR_POINTS:
            continue
        if cv2.contourArea(cnt) < ELLIPSE_MIN_AREA:
            continue

        try:
            ellipse = cv2.fitEllipse(cnt)
        except:
            continue

        (cx,cy),(M,m),ang = ellipse
        if M < m:
            M,m,ang = m,M,ang+90

        if M < (x2c-x1c)*ELLIPSE_MAJOR_AXIS_MIN_RATIO:
            continue

        ok,_ = is_full_ellipse(cnt, ((cx,cy),(M,m),ang))
        if ok:
            return (cx,cy),(M,m),ang

    return None

def calculate_bottom_center(rim_cx, rim_cy, ellipse):
    (cx,cy),(M,m),ang = ellipse
    a = M/2
    b = m/2
    ratio = np.clip(b/a, 0.0, 0.999)

    extend = np.clip(2.236*a*np.sqrt(1-ratio**2)*1.0, 10, 200)

    short_ang = (ang+90) % 360
    rad = np.radians(short_ang)
    dir_vec = np.array([np.cos(rad), np.sin(rad)])
    if dir_vec[1] < 0:
        dir_vec = -dir_vec
    dir_vec /= np.linalg.norm(dir_vec)

    bx = int(rim_cx + dir_vec[0]*extend)
    by = int(rim_cy + dir_vec[1]*extend)
    return bx, by

def update_smooth(raw):
    now = time.time()

    if raw is not None:
        state.last_detect_time = now
        (cx,cy),(M,m),ang = raw
        state.ellipse_buf.append((cx,cy,M,m,ang))

        arr = np.array(state.ellipse_buf)
        mean = arr.mean(axis=0)

        if state.smooth_ellipse is None:
            state.smooth_ellipse = mean
        else:
            prev = np.array(state.smooth_ellipse)
            mean[0] = limit_change(prev[0], mean[0], 4)
            mean[1] = limit_change(prev[1], mean[1], 4)
            mean[4] = limit_change(prev[4], mean[4], 3)

            state.smooth_ellipse = SMOOTH_ALPHA*mean + (1-SMOOTH_ALPHA)*prev
    else:
        if now - state.last_detect_time > MISS_TIMEOUT:
            state.smooth_ellipse = None
            state.ellipse_buf.clear()

    return state.smooth_ellipse
# ==================== 标定鼠标回调 ====================
def mouse_callback(event, x, y, flags, param):
    if state.calibrated:
        if not state.calibration_prompted:
            print("ℹ️ Calibration done. Press 'R' to reset.")
            state.calibration_prompted = True
        return
    if event != cv2.EVENT_LBUTTONDOWN:
        return

    if state.mode == "plane":
        if state.calib_points:
            lx, ly = state.calib_points[-1]
            if np.hypot(x - lx, y - ly) < 10:
                print("⚠️ 点太近，请重新点击")
                return
        state.calib_points.append([x, y])
        state.plane_calib_count = len(state.calib_points)
        print(f"📍 平面标定点{state.plane_calib_count}: ({x},{y})")

        if state.plane_calib_count == 4:
            try:
                calib_np = np.array(state.calib_points, np.float32)
                state.H, _ = cv2.findHomography(calib_np, real_points, cv2.RANSAC, 5.0)
                if state.H is None:
                    raise Exception("单应性矩阵计算失败")
                state.calibrated = True
                state.calib_step = 4
                state.calib_done_time = time.time()
                print("✅ 平面标定完成")
            except Exception as e:
                print(f"❌ {e}")
                state.calib_points = []
                state.plane_calib_count = 0
            finally:
                state.calib_points = []

    elif state.mode == "origin":
        state.origin = np.array([x, y], np.float32)
        state.calib_step = 1
        state.mode = None
        print(f"🟢 原点: ({x},{y}) → 按 'x' 标定X轴")

    elif state.mode == "x_axis":
        if state.origin is None:
            print("⚠️ 先标原点")
            return
        if np.hypot(x - state.origin[0], y - state.origin[1]) < 20:
            print("⚠️ 距原点过近")
            return
        state.x_ref = np.array([x, y], np.float32)
        state.calib_step = 2
        state.mode = None
        print(f"🔵 X轴: ({x},{y}) → 按 'y' 标定Y轴")

    elif state.mode == "y_axis":
        if state.origin is None or state.x_ref is None:
            print("⚠️ 先标原点和X轴")
            return
        if np.hypot(x - state.origin[0], y - state.origin[1]) < 20:
            print("⚠️ 距原点过近")
            return
        state.y_ref = np.array([x, y], np.float32)
        state.calibrated = True
        state.calib_step = 4
        state.mode = None
        state.calib_done_time = time.time()
        print(f"🟣 Y轴: ({x},{y}) 标定完成 ✅")


# ==================== 像素 → 实际坐标 ====================
def get_real_coordinates(pt):
    if state.calibrated and state.H is not None:
        r = cv2.perspectiveTransform(np.array([[pt]], dtype=np.float32), state.H)
        return float(r[0][0][0]), float(r[0][0][1])

    if state.origin is not None and state.x_ref is not None and state.y_ref is not None:
        vec = np.array(pt, np.float32) - state.origin
        xv = state.x_ref - state.origin
        xlen = np.linalg.norm(xv)
        if xlen < 1e-6:
            return 0.0, 0.0
        xu = xv / xlen

        yv = state.y_ref - state.origin
        ylen = np.linalg.norm(yv)
        if ylen < 1e-6:
            return float(np.dot(vec, xu) * 9 / xlen), 0.0
        yu = yv / ylen
        if np.dot(xu, yu) > 0.01:
            yu = np.array([-xu[1], xu[0]])

        return float(np.dot(vec, xu) * 9 / xlen), float(-np.dot(vec, yu) * 9 / ylen)

    return pt[0] / FRAME_WIDTH * 9, pt[1] / FRAME_HEIGHT * 9


# ==================== 机械臂抓取指令生成 ====================
def generate_grab_commands(real_x, real_y):
    arm_Y = model_Y.predict(np.array([[real_x]]))[0]
    arm_X = model_X.predict(np.array([[real_y]]))[0]

    arm_X2 = arm_Y - 3
    arm_Y2 = arm_X - 10

    arm_Y  = np.clip(arm_Y,  -200.0, 350.0)
    arm_X  = np.clip(arm_X,  -200.0, 350.0)
    arm_X2 = np.clip(arm_X2, -200.0, 350.0)
    arm_Y2 = np.clip(arm_Y2, -200.0, 350.0)

    cmd1 = f'{{"T":1041,"x":{arm_X+25:.6f},"y":{arm_Y:.6f},"z":6,"t":1,"r":0,"g":3.141592654}}'
    cmd2 = f'{{"T":1041,"x":{arm_Y2+35:.6f},"y":{arm_X2:.6f},"z":-85,"t":1.35,"r":0,"g":3.141592654}}'
    cmd3 = f'{{"T":1041,"x":{arm_Y2+35:.6f},"y":{arm_X2:.6f},"z":-85,"t":1.35,"r":0,"g":2.3}}'

    return [cmd1, cmd2, cmd3]
# ==================== WiFi通信 ====================
def send_control_cmd(cmd_list, freq=None):
    if not isinstance(cmd_list, list):
        cmd_list = [cmd_list]
    try:
        for idx, cmd in enumerate(cmd_list):
            if not cmd or not isinstance(cmd, str):
                print(f"⚠️ 跳过无效指令（第{idx+1}条）")
                continue
            url = f"http://{ARM_IP}/js?json={cmd}"
            resp = requests.get(url, timeout=ARM_TIMEOUT)
            print(f"📤 发送第{idx+1}条指令: {cmd}")
            print(f"📥 机械臂回复: {resp.text}")
            if idx < len(cmd_list) - 1:
                print(f"⌛ 延迟0.5秒...")
                time.sleep(0.5)
        return True
    except requests.exceptions.Timeout:
        print(f"❌ 请求超时（IP: {ARM_IP}）")
        return False
    except requests.exceptions.ConnectionError:
        print(f"❌ 无法连接机械臂（IP: {ARM_IP}）")
        return False
    except Exception as e:
        print(f"❌ 指令发送失败: {e}")
        traceback.print_exc()
        return False


def check_arm_connection():
    try:
        url = f"http://{ARM_IP}/js?json={{\"T\":105}}"
        resp = requests.get(url, timeout=ARM_TIMEOUT)
        print(f"✅ 机械臂WiFi连接成功！IP: {ARM_IP}")
        print(f"📊 机械臂状态: {resp.text[:100]}")
        return True
    except Exception as e:
        print(f"⚠️ 机械臂WiFi连接失败（IP: {ARM_IP}）: {e}")
        print("💡 请检查：1.机械臂是否开机 2.IP是否正确 3.是否在同一WiFi")
        return False

def read_triggers_from_file():
    if not os.path.exists(TRIGGER_FILE):
        return []
    triggers = []
    try:
        with open(TRIGGER_FILE, "r", encoding="utf-8") as f:
            lines = f.readlines()
        open(TRIGGER_FILE, "w").close()
        for line in lines:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                age = time.time() - entry.get("time", 0)
                if age < TRIGGER_MAX_AGE:
                    triggers.append((entry["freq"], entry.get("func", "unknown")))
                    print(f"📥 读取触发指令: {entry['freq']}Hz ({age:.2f}秒前写入)")
                else:
                    print(f"⏭️ 丢弃过期指令: {entry['freq']}Hz ({age:.2f}秒前写入，已超时)")
            except Exception as e:
                print(f"⚠️ 解析触发指令失败: {e}")
    except Exception as e:
        print(f"⚠️ 读取触发文件失败: {e}")
    return triggers

# ==================== YOLO & 摄像头 ====================
def load_yolo_model(model_path=YOLO_MODEL_PATH):
    try:
        model = YOLO(model_path)
        model.fuse()
        print(f"✅ YOLO loaded: {model_path}")
        return model
    except Exception as e:
        print(f"❌ YOLO load failed: {e}")
        return None


def init_camera():
    for backend in [cv2.CAP_DSHOW, cv2.CAP_MSMF]:
        cap = cv2.VideoCapture(VIDEO_SOURCE, backend)
        if not cap.isOpened():
            cap.release()
            continue

        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
        cap.set(cv2.CAP_PROP_FPS, FPS_TARGET)

        ret, f = cap.read()
        if ret and f is not None and f.size > 0:
            print("✅ 摄像头就绪")
            return cap

        cap.release()

    print("⚠️ 摄像头初始化失败")
    return None
# ==================== 工业级 collect_cup_coords（带 duration） ====================
def collect_cup_coords(cap, model, collect_duration=2.0, label="扫描"):
    """
    工业级彩色图像识别版本（完全替换旧版）
    支持 duration（预热扫描 / 触发扫描）
    """
    final_dx, final_dy = None, None
    start_time = time.time()
    print(f"\n📸 [{label}] 开始图像采集，持续 {collect_duration:.0f} 秒...")

    while time.time() - start_time < collect_duration:

        # -------------------- 读取帧 --------------------
        if cap is not None:
            ret, frame = cap.read()
            if not ret or frame is None or frame.size == 0:
                cap.release()
                cap = init_camera()
                frame = np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)
        else:
            frame = np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)

        # -------------------- 彩色增强（工业级） --------------------
        frame = cv2.convertScaleAbs(frame, alpha=1.05, beta=5)

        # -------------------- 标定轴显示 --------------------
        if state.origin is not None:
            cv2.circle(frame, tuple(state.origin.astype(int)), 6, (0,255,0), -1)
            cv2.putText(frame, "O", tuple(state.origin.astype(int)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
        if state.x_ref is not None:
            cv2.arrowedLine(frame, tuple(state.origin.astype(int)),
                            tuple(state.x_ref.astype(int)), (255,0,0), 2)
        if state.y_ref is not None:
            cv2.arrowedLine(frame, tuple(state.origin.astype(int)),
                            tuple(state.y_ref.astype(int)), (200,0,200), 2)

        # -------------------- YOLO 检测 --------------------
        if model is not None:
            try:
                results = model(frame, conf=0.5, verbose=False)
                boxes = results[0].boxes.xyxy.cpu().numpy()
                classes = results[0].boxes.cls.cpu().numpy()
            except:
                boxes = classes = np.array([])
        else:
            boxes = classes = np.array([])

        u = v = None
        has_cup = False

        # -------------------- 处理第一个杯子 --------------------
        if boxes.size > 0:
            for box, cls in zip(boxes, classes):
                if model.names[int(cls)] != "cup":
                    continue
                has_cup = True

                x1, y1, x2, y2 = map(int, box)

                # YOLO 框平滑
                x1, y1, x2, y2 = smooth_yolo_box(state, x1, y1, x2, y2)
                cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)

                # -------------------- ROI 运动检测 --------------------
                upper_h = int((y2 - y1) * ROI_UPPER_RATIO)
                if upper_h > 0:
                    H, W = frame.shape[:2]
                    x1c = max(0, min(x1, W-1))
                    x2c = max(0, min(x2, W))
                    y1c = max(0, min(y1, H-1))
                    y2c = max(0, min(y1+upper_h, H))

                    if x2c > x1c and y2c > y1c:
                        roi = frame[y1c:y2c, x1c:x2c]
                        if roi.size > 0:
                            if state.prev_roi is not None:
                                h, w = roi.shape[:2]
                                prev = cv2.resize(state.prev_roi, (w, h))
                                diff = cv2.absdiff(prev, roi)
                                motion = diff.mean()
                            else:
                                motion = 0
                            state.prev_roi = roi.copy()
                        else:
                            motion = 0
                    else:
                        motion = 0
                else:
                    motion = 0

                # -------------------- 椭圆检测（工业级） --------------------
                if motion > 8:
                    raw = None
                else:
                    raw = detect_valid_ellipse_in_upper_roi(frame, x1, y1, x2, y2)

                smooth = update_smooth(raw)

                if smooth is not None:
                    cx, cy, M, m, ang = smooth
                    gx = int(x1 + cx)
                    gy = int(y1 + cy)

                    cv2.ellipse(frame, (gx,gy),
                                (max(1,int(M/2)), max(1,int(m/2))),
                                ang, 0,360, (0,0,255),2)

                    # 杯底推算（工业级）
                    bx, by = calculate_bottom_center(cx, cy, ((cx,cy),(M,m),ang))
                    bx_g = x1 + bx
                    by_g = y1 + by

                    cv2.circle(frame, (bx_g,by_g), 6, (0,255,0), -1)

                    mid_x, mid_y = bx_g, by_g
                    state.last_detect_time = time.time()
                else:
                    # fallback
                    if state.prev_bottom is not None and time.time() - state.last_detect_time < 0.5:
                        mid_x, mid_y = state.prev_bottom
                    else:
                        mid_x, mid_y = (x1+x2)//2, y2

                # -------------------- 底部点平滑 --------------------
                state.bottom_window.append((mid_x, mid_y))
                avg_x = np.mean([p[0] for p in state.bottom_window])
                avg_y = np.mean([p[1] for p in state.bottom_window])

                if state.prev_bottom is None:
                    sm_x, sm_y = avg_x, avg_y
                else:
                    sm_x = avg_x*(1-BOTTOM_MEM_WEIGHT) + state.prev_bottom[0]*BOTTOM_MEM_WEIGHT
                    sm_y = avg_y*(1-BOTTOM_MEM_WEIGHT) + state.prev_bottom[1]*BOTTOM_MEM_WEIGHT

                sm_x, sm_y = int(sm_x), int(sm_y)
                state.prev_bottom = (sm_x, sm_y)

                # -------------------- Kalman --------------------
                if state.kalman is not None:
                    state.kalman.correct(np.array([[np.float32(sm_x)],
                                                   [np.float32(sm_y)]]))
                    kp = state.kalman.predict()
                    u, v = int(kp[0][0]), int(kp[1][0])
                else:
                    u, v = sm_x, sm_y

                # -------------------- 显示 & 坐标转换 --------------------
                if u is not None and v is not None:
                    cv2.circle(frame, (u,v), 5, (0,255,0), -1)
                    dx, dy = get_real_coordinates((u,v))
                    state.stable_window.append((dx,dy))
                    state.second_buffer.append((dx,dy))

                break  # 只处理第一个杯子

        # -------------------- 无杯子时清理状态 --------------------
        if not has_cup:
            state.prev_roi = None
            state.smooth_ellipse = None
            state.ellipse_buf.clear()
            state.prev_bottom = None
            state.bottom_window.clear()

        # -------------------- 每秒输出一次 --------------------
        if time.time() - state.last_second_time >= 1.0 and state.second_buffer:
            xs = np.array([c[0] for c in state.second_buffer])
            ys = np.array([c[1] for c in state.second_buffer])
            mx, my = np.median(xs), np.median(ys)
            mask = (np.abs(xs-mx)<1.0) & (np.abs(ys-my)<1.0)
            fx = xs[mask].mean() if mask.sum()>0 else mx
            fy = ys[mask].mean() if mask.sum()>0 else my
            final_dx, final_dy = fx, fy
            print(f"[{label}每秒输出] x={fx:.1f}cm y={fy:.1f}cm")
            state.second_buffer.clear()
            state.last_second_time = time.time()

        # -------------------- 显示 --------------------
        cv2.putText(frame, f"[{label}] 剩余 {collect_duration - (time.time()-start_time):.1f}s",
                    (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255),2)

        cv2.imshow(WINDOW_NAME, frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    # -------------------- 返回最终坐标 --------------------
    if final_dx is None:
        print(f"❌ [{label}] 未检测到有效杯子坐标")
    else:
        print(f"✅ [{label}] 完成，杯子坐标: x={final_dx:.1f}cm, y={final_dy:.1f}cm")

    return final_dx, final_dy


# ==================== 主函数 ====================
def main():
    print("\n" + "="*52)
    print("  SSVEP机械臂控制系统 v6.0-C（工业级彩色识别版）")
    print("="*52)

    check_arm_connection()

    if os.path.exists(TRIGGER_FILE):
        open(TRIGGER_FILE, "w").close()
        print(f"🗑️ 已清空旧触发文件")

    model = load_yolo_model()
    cap   = init_camera()

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL | cv2.WINDOW_GUI_EXPANDED)
    cv2.resizeWindow(WINDOW_NAME, FRAME_WIDTH, FRAME_HEIGHT)
    cv2.setMouseCallback(WINDOW_NAME, mouse_callback)

    print("""
    🧭 使用说明:
    - 'p' 平面标定（4点）
    - 'o' 原点   'x' X轴   'y' Y轴
    - 标定完成后自动预热扫描 5 秒
    - 10Hz：触发扫描 2 秒 → 抓取
    - 11~15Hz：执行预设动作
    - 'r' 重置标定
    - ESC 退出
    """)

    last_file_check = time.time()
    FILE_CHECK_INTERVAL = 0.1
    pre_scan_result = {"dx":None, "dy":None}

    try:
        while True:

            # -------------------- 自动预热扫描 --------------------
            if state.calibrated and not state.pre_scan_done:
                print("\n🌡️ 标定完成 → 开始预热扫描 5 秒")
                dx, dy = collect_cup_coords(cap, model, 5.0, "预热扫描")
                pre_scan_result["dx"] = dx
                pre_scan_result["dy"] = dy
                state.pre_scan_done = True

            # -------------------- 读取触发文件 --------------------
            if time.time() - last_file_check >= FILE_CHECK_INTERVAL:
                triggers = read_triggers_from_file()
                for freq, func in triggers:
                    TRIGGER_QUEUE.put((freq, func))
                last_file_check = time.time()

            # -------------------- 处理触发 --------------------
            while not TRIGGER_QUEUE.empty():
                freq, func = TRIGGER_QUEUE.get()

                if freq == 10.0:
                    print("\n🧠 收到 10Hz → 精确扫描 2 秒")
                    dx, dy = collect_cup_coords(cap, model, 2.0, "触发扫描")
                    if dx is None:
                        dx = pre_scan_result["dx"]
                        dy = pre_scan_result["dy"]
                    if dx is None:
                        print("❌ 坐标无效，跳过抓取")
                        continue
                    cmds = generate_grab_commands(dx, dy)
                    print(f"🚀 抓取坐标 ({dx:.1f}, {dy:.1f})")
                    send_control_cmd(cmds)

                elif freq in FREQ_TO_ARM_CMD:
                    print(f"🚀 执行 {freq}Hz 指令")
                    send_control_cmd(FREQ_TO_ARM_CMD[freq])

            # -------------------- 主循环显示 --------------------
            ret, frame = cap.read()
            if not ret:
                frame = np.zeros((FRAME_HEIGHT,FRAME_WIDTH,3),dtype=np.uint8)

            frame = cv2.convertScaleAbs(frame, alpha=1.05, beta=5)

            if state.calibrated:
                cv2.putText(frame, "Calibration: Locked", (50,50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0),2)
            else:
                cv2.putText(frame, "Calibration: Pending", (50,50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255),2)

            cv2.imshow(WINDOW_NAME, frame)

            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                break
            elif key == ord('p'):
                state.mode = "plane"
                state.calib_points.clear()
                state.plane_calib_count = 0
            elif key == ord('o'):
                state.mode = "origin"
            elif key == ord('x'):
                state.mode = "x_axis"
            elif key == ord('y'):
                state.mode = "y_axis"
            elif key == ord('r'):
                state.__init__()
                pre_scan_result = {"dx":None,"dy":None}

    except Exception as e:
        print("💥 主循环错误:", e)
        traceback.print_exc()
    finally:
        if cap:
            cap.release()
        cv2.destroyAllWindows()
        print("✅ 程序退出")


if __name__ == "__main__":
    main()
