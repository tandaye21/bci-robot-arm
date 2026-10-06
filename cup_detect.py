import cv2
import numpy as np
import time
from ultralytics import YOLO
from collections import deque
import traceback

# -------------------- 配置 --------------------
WINDOW_NAME = "Cup Detection & Calibration"
YOLO_MODEL_PATH = "yolov8s.pt"
VIDEO_SOURCE = 1
FRAME_WIDTH, FRAME_HEIGHT = 640, 480
FPS_TARGET = 30

# 平滑 & Kalman
MEM_WEIGHT           = 0.1
BOTTOM_SMOOTH_LEN    = 3
BOTTOM_MEM_WEIGHT    = 0.1
TOP_SMOOTH_LEN       = 3
KALMAN_PROCESS_NOISE = 1e-4
KALMAN_MEAS_NOISE    = 1e-2

# 标定
real_points   = np.array([[0,0],[9,0],[0,9],[9,9]], dtype=np.float32)
debug_ellipse = False

# 椭圆检测参数（工业级稳定版）
ELLIPSE_MIN_AREA              = 50
ROI_UPPER_RATIO               = 0.75
MIN_CONTOUR_POINTS            = 5
FULL_ELLIPSE_FIT_ERROR_MAX    = 0.25
FULL_ELLIPSE_COVERAGE_THRESH  = 0.55
FULL_ELLIPSE_CENTER_OFFSET_MAX= 0.3
FULL_ELLIPSE_BOUNDING_BOX_RATIO = 0.4
ELLIPSE_MAJOR_AXIS_MIN_RATIO  = 0.75

# 椭圆平滑参数
SMOOTH_LEN    = 5
SMOOTH_ALPHA  = 0.4
MISS_TIMEOUT  = 0.3

# -------------------- 全局状态 --------------------
class State:
    def __init__(self):
        # 标定相关
        self.origin = self.x_ref = self.y_ref = self.H = None
        self.calib_points = []
        self.calibrated = False
        self.calibration_prompted = False
        self.calib_step = 0
        self.plane_calib_count = 0
        self.mode = None

        # 原有平滑窗口
        self.top_window    = deque(maxlen=TOP_SMOOTH_LEN)
        self.bottom_window = deque(maxlen=BOTTOM_SMOOTH_LEN)
        self.prev_top = self.prev_bottom = None
        self.stable_window    = deque(maxlen=15)
        self.second_buffer    = deque(maxlen=30)
        self.last_second_time = time.time()

        # Kalman
        self.kalman = self._init_kalman()

        # 旧逻辑中使用的时间戳
        self.last_valid_ellipse_time = time.time()

        # 工业级稳定版：椭圆平滑
        self.ellipse_buf     = deque(maxlen=SMOOTH_LEN)
        self.smooth_ellipse  = None
        self.last_detect_time= time.time()

        # YOLO 框平滑
        self.smooth_box = None
        self.box_alpha  = 0.35

        # ROI 运动检测
        self.prev_roi = None

    def _init_kalman(self):
        kf = cv2.KalmanFilter(4, 2)
        kf.measurementMatrix   = np.array([[1,0,0,0],[0,1,0,0]], np.float32)
        kf.transitionMatrix    = np.array([[1,0,1,0],[0,1,0,1],[0,0,1,0],[0,0,0,1]], np.float32)
        kf.processNoiseCov     = np.eye(4, dtype=np.float32) * KALMAN_PROCESS_NOISE
        kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * KALMAN_MEAS_NOISE
        kf.statePre            = np.zeros((4,1), np.float32)
        return kf

    def reset(self):
        self.__init__()

state = State()

# -------------------- 工具函数（工业级稳定版识别） --------------------
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
    a, b = M / 2, m / 2
    theta = np.radians(ang)
    contour_points = contour.reshape(-1, 2).astype(np.float32)

    covered = 0
    fit_errors = []

    for (px, py) in contour_points:
        x_rot = (px - cx) * np.cos(theta) + (py - cy) * np.sin(theta)
        y_rot = -(px - cx) * np.sin(theta) + (py - cy) * np.cos(theta)
        dist = abs((x_rot**2)/(a**2) + (y_rot**2)/(b**2) - 1.0)
        fit_errors.append(dist)
        if dist <= FULL_ELLIPSE_FIT_ERROR_MAX:
            covered += 1

    coverage_ratio = covered / len(contour_points)
    avg_fit_error  = np.mean(fit_errors)

    M_cnt = cv2.moments(contour)
    if M_cnt["m00"] == 0:
        return False, avg_fit_error

    cnt_cx = M_cnt["m10"] / M_cnt["m00"]
    cnt_cy = M_cnt["m01"] / M_cnt["m00"]
    center_offset = np.sqrt((cnt_cx - cx)**2 + (cnt_cy - cy)**2)
    max_offset    = min(a, b) * FULL_ELLIPSE_CENTER_OFFSET_MAX

    x, y, w, h = cv2.boundingRect(contour)
    bbox_area   = w * h
    ellipse_area= np.pi * a * b
    bbox_ratio  = ellipse_area / bbox_area if bbox_area > 0 else 0

    if (coverage_ratio >= FULL_ELLIPSE_COVERAGE_THRESH and
        center_offset <= max_offset and
        bbox_ratio >= FULL_ELLIPSE_BOUNDING_BOX_RATIO):
        return True, avg_fit_error

    return False, avg_fit_error

def detect_valid_ellipse_in_upper_roi(frame, x1, y1, x2, y2):
    h = y2 - y1
    upper_h = int(h * ROI_UPPER_RATIO)
    if upper_h <= 10:
        return None

    h_frame, w_frame = frame.shape[:2]
    x1_c = max(0, min(x1, w_frame-1))
    x2_c = max(0, min(x2, w_frame))
    y1_c = max(0, min(y1, h_frame-1))
    y2_c = max(0, min(y1+upper_h, h_frame))

    if x2_c <= x1_c or y2_c <= y1_c:
        return None

    roi = frame[y1_c:y2_c, x1_c:x2_c]
    if roi.size == 0:
        return None

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 1.2)
    edges = cv2.Canny(gray, 50, 150)

    contours, _ = cv2.findContours(edges, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        if len(cnt) < MIN_CONTOUR_POINTS:
            continue
        if cv2.contourArea(cnt) < ELLIPSE_MIN_AREA:
            continue

        try:
            ellipse = cv2.fitEllipse(cnt)
        except:
            continue

        (cx, cy), (M, m), ang = ellipse
        if M < m:
            M, m, ang = m, M, ang + 90

        if M < (x2_c - x1_c) * ELLIPSE_MAJOR_AXIS_MIN_RATIO:
            continue

        ok, _ = is_full_ellipse(cnt, ellipse)
        if ok:
            return (cx, cy), (M, m), ang

    return None

def calculate_bottom_center(rim_cx, rim_cy, ellipse):
    (cx, cy), (M, m), ang = ellipse
    a = M / 2
    b = m / 2
    ratio = np.clip(b / a, 0.0, 0.999)

    extend = np.clip(2.236 * a * np.sqrt(1 - ratio**2) * 1.0, 10, 200)

    short_ang = (ang + 90) % 360
    rad = np.radians(short_ang)
    dir_vec = np.array([np.cos(rad), np.sin(rad)])
    if dir_vec[1] < 0:
        dir_vec = -dir_vec
    dir_vec /= np.linalg.norm(dir_vec)

    bx = int(rim_cx + dir_vec[0] * extend)
    by = int(rim_cy + dir_vec[1] * extend)
    return bx, by

def update_smooth(raw):
    now = time.time()

    if raw is not None:
        state.last_detect_time = now
        (cx, cy), (M, m), ang = raw
        state.ellipse_buf.append((cx, cy, M, m, ang))

        arr  = np.array(state.ellipse_buf)
        mean = arr.mean(axis=0)

        if state.smooth_ellipse is None:
            state.smooth_ellipse = mean
        else:
            prev = np.array(state.smooth_ellipse)
            mean[0] = limit_change(prev[0], mean[0], 4)
            mean[1] = limit_change(prev[1], mean[1], 4)
            mean[4] = limit_change(prev[4], mean[4], 3)

            state.smooth_ellipse = (
                SMOOTH_ALPHA * mean +
                (1 - SMOOTH_ALPHA) * prev
            )
    else:
        if now - state.last_detect_time > MISS_TIMEOUT:
            state.smooth_ellipse = None
            state.ellipse_buf.clear()

    return state.smooth_ellipse

# -------------------- 标定 --------------------
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
                print("⚠️ 点太近，请重新点击"); return
        state.calib_points.append([x, y])
        state.plane_calib_count = len(state.calib_points)
        print(f"📍 平面标定点{state.plane_calib_count}: ({x},{y})")
        if state.plane_calib_count == 4:
            try:
                calib_np = np.array(state.calib_points, np.float32)
                state.H, _ = cv2.findHomography(calib_np, real_points, cv2.RANSAC, 5.0)
                if state.H is None: raise Exception("单应性矩阵计算失败")
                state.calibrated = True; state.calib_step = 4
                print("✅ 平面标定完成")
            except Exception as e:
                print(f"❌ {e}"); state.calib_points = []; state.plane_calib_count = 0
            finally:
                state.calib_points = []

    elif state.mode == "origin":
        state.origin = np.array([x, y], np.float32)
        state.calib_step = 1; state.mode = None
        print(f"🟢 原点: ({x},{y})  → 按 'x' 标定X轴")

    elif state.mode == "x_axis":
        if state.origin is None: print("⚠️ 先标原点"); return
        if np.hypot(x - state.origin[0], y - state.origin[1]) < 20:
            print("⚠️ 距原点过近"); return
        state.x_ref = np.array([x, y], np.float32)
        state.calib_step = 2; state.mode = None
        print(f"🔵 X轴: ({x},{y})  → 按 'y' 标定Y轴")

    elif state.mode == "y_axis":
        if state.origin is None or state.x_ref is None:
            print("⚠️ 先标原点和X轴"); return
        if np.hypot(x - state.origin[0], y - state.origin[1]) < 20:
            print("⚠️ 距原点过近"); return
        state.y_ref = np.array([x, y], np.float32)
        state.calibrated = True; state.calib_step = 4; state.mode = None
        print(f"🟣 Y轴: ({x},{y})  标定完成 ✅")

def get_real_coords(pt):
    if state.calibrated and state.H is not None:
        r = cv2.perspectiveTransform(np.array([[pt]], dtype=np.float32), state.H)
        return float(r[0][0][0]), float(r[0][0][1])
    if state.origin is not None and state.x_ref is not None and state.y_ref is not None:
        vec    = np.array(pt, np.float32) - state.origin
        xv     = state.x_ref - state.origin
        xlen   = np.linalg.norm(xv)
        if xlen < 1e-6: return 0.0, 0.0
        xu     = xv / xlen
        yv     = state.y_ref - state.origin
        ylen   = np.linalg.norm(yv)
        if ylen < 1e-6: return float(np.dot(vec, xu) * 9 / xlen), 0.0
        yu     = yv / ylen
        if np.dot(xu, yu) > 0.01:
            yu = np.array([-xu[1], xu[0]])
        return float(np.dot(vec, xu) * 9 / xlen), float(-np.dot(vec, yu) * 9 / ylen)
    return pt[0] / FRAME_WIDTH * 9, pt[1] / FRAME_HEIGHT * 9

# -------------------- 摄像头 & YOLO --------------------
def load_yolo():
    try:
        m = YOLO(YOLO_MODEL_PATH); m.fuse()
        print(f"✅ YOLO loaded: {YOLO_MODEL_PATH}"); return m
    except Exception as e:
        print(f"❌ YOLO failed: {e}"); return None

def init_camera():
    for backend in [cv2.CAP_DSHOW, cv2.CAP_MSMF]:
        cap = cv2.VideoCapture(VIDEO_SOURCE, backend)
        if not cap.isOpened(): cap.release(); continue
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
        cap.set(cv2.CAP_PROP_FPS, FPS_TARGET)
        ret, f = cap.read()
        if ret and f is not None and f.size > 0:
            print(f"✅ 摄像头就绪"); return cap
        cap.release()
    print("⚠️ 摄像头初始化失败"); return None

# -------------------- 主循环 --------------------
def run(cap, model):
    global debug_ellipse
    while True:
        # 读帧
        if cap is not None:
            ret, frame = cap.read()
            if not ret or frame is None or frame.size == 0:
                cap.release(); cap = init_camera()
                frame = np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)
        else:
            frame = np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)

        # 图像增强
        frame = cv2.convertScaleAbs(frame, alpha=1.05, beta=5)

        # 标定轴显示
        if state.origin is not None:
            cv2.circle(frame, tuple(state.origin.astype(int)), 6, (0,255,0), -1)
            cv2.putText(frame, "O", tuple(state.origin.astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
        if state.x_ref is not None:
            cv2.arrowedLine(frame, tuple(state.origin.astype(int)), tuple(state.x_ref.astype(int)), (255,0,0), 2)
            cv2.putText(frame, "X", tuple(state.x_ref.astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 2)
        if state.y_ref is not None:
            cv2.arrowedLine(frame, tuple(state.origin.astype(int)), tuple(state.y_ref.astype(int)), (200,0,200), 2)
            cv2.putText(frame, "Y", tuple(state.y_ref.astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200,0,200), 2)

        # YOLO 检测
        if model is not None:
            try:
                results = model(frame, conf=0.5, verbose=False)
                boxes   = results[0].boxes.xyxy.cpu().numpy()
                classes = results[0].boxes.cls.cpu().numpy()
            except:
                boxes = classes = np.array([])
        else:
            boxes = classes = np.array([])

        u, v = None, None
        has_cup = False

        if boxes.size > 0:
            for box, cls in zip(boxes, classes):
                if model.names[int(cls)] != "cup":
                    continue
                has_cup = True

                x1, y1, x2, y2 = map(int, box)

                # YOLO 框平滑
                x1, y1, x2, y2 = smooth_yolo_box(state, x1, y1, x2, y2)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # ROI 运动检测
                upper_h = int((y2 - y1) * ROI_UPPER_RATIO)
                if upper_h > 0:
                    h_frame, w_frame = frame.shape[:2]
                    x1_c = max(0, min(x1, w_frame-1))
                    x2_c = max(0, min(x2, w_frame))
                    y1_c = max(0, min(y1, h_frame-1))
                    y2_c = max(0, min(y1+upper_h, h_frame))

                    if x2_c > x1_c and y2_c > y1_c:
                        roi = frame[y1_c:y2_c, x1_c:x2_c]
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

                # 运动大时暂停更新椭圆
                if motion > 8:
                    raw = None
                else:
                    raw = detect_valid_ellipse_in_upper_roi(frame, x1, y1, x2, y2)

                smooth = update_smooth(raw)

                if smooth is not None:
                    cx, cy, M, m, ang = smooth
                    gx = int(x1 + cx)
                    gy = int(y1 + cy)

                    cv2.ellipse(frame, (gx, gy),
                                (max(1, int(M/2)), max(1, int(m/2))),
                                ang, 0, 360, (0, 0, 255), 2)
                    cv2.circle(frame, (gx, gy), 4, (0, 255, 255), -1)

                    bx, by = calculate_bottom_center(cx, cy, ((cx, cy), (M, m), ang))
                    bx_g = x1 + bx
                    by_g = y1 + by

                    cv2.line(frame, (gx, gy), (bx_g, by_g), (0, 255, 0), 2)
                    cv2.circle(frame, (bx_g, by_g), 6, (0, 255, 0), -1)

                    # 这里的 (bx_g, by_g) 就是工业级稳定版杯底像素坐标
                    mid_x, mid_y = bx_g, by_g
                    state.last_valid_ellipse_time = time.time()
                else:
                    # 椭圆失败时回退策略：用上一帧或框底中心
                    if state.prev_bottom is not None and \
                            time.time() - state.last_valid_ellipse_time < 0.5:
                        mid_x, mid_y = state.prev_bottom
                    else:
                        mid_x, mid_y = (x1 + x2) // 2, y2

                # 底部点平滑（沿用你原来的 bottom_window + exp_smooth 思路）
                state.bottom_window.append((mid_x, mid_y))
                avg_x = np.mean([p[0] for p in state.bottom_window])
                avg_y = np.mean([p[1] for p in state.bottom_window])

                if state.prev_bottom is None:
                    sm_x, sm_y = avg_x, avg_y
                else:
                    sm_x = avg_x * (1 - BOTTOM_MEM_WEIGHT) + state.prev_bottom[0] * BOTTOM_MEM_WEIGHT
                    sm_y = avg_y * (1 - BOTTOM_MEM_WEIGHT) + state.prev_bottom[1] * BOTTOM_MEM_WEIGHT

                sm_x, sm_y = int(sm_x), int(sm_y)
                state.prev_bottom = (sm_x, sm_y)

                # Kalman
                if state.kalman is not None:
                    state.kalman.correct(np.array([[np.float32(sm_x)],
                                                   [np.float32(sm_y)]]))
                    kp = state.kalman.predict()
                    u, v = int(kp[0][0]), int(kp[1][0])
                else:
                    u, v = sm_x, sm_y

                # 杯底点 + 坐标显示
                if u is not None and v is not None:
                    cv2.circle(frame, (u, v), 5, (0,255,0), -1)
                    cv2.putText(frame, f"Bottom({u},{v})", (u+8, v+6),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
                    dx, dy = get_real_coords((u, v))
                    state.stable_window.append((dx, dy))
                    state.second_buffer.append((dx, dy))
                    dx_s = np.mean([c[0] for c in state.stable_window])
                    dy_s = np.mean([c[1] for c in state.stable_window])
                    cv2.putText(frame, f"x={dx_s:.1f}cm y={dy_s:.1f}cm",
                                (u+10, max(20, v-15)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

                break  # 只处理第一个杯子

        # 如果本帧没有杯子，清理部分状态，避免误判
        if not has_cup:
            state.prev_roi = None
            state.smooth_ellipse = None
            state.ellipse_buf.clear()
            state.prev_bottom = None
            state.bottom_window.clear()

        # 每秒输出
        if time.time() - state.last_second_time >= 1.0 and state.second_buffer:
            xs = np.array([c[0] for c in state.second_buffer])
            ys = np.array([c[1] for c in state.second_buffer])
            mx, my = np.median(xs), np.median(ys)
            mask = (np.abs(xs-mx)<1.0) & (np.abs(ys-my)<1.0)
            fx = xs[mask].mean() if mask.sum()>0 else mx
            fy = ys[mask].mean() if mask.sum()>0 else my
            print(f"[每秒输出] x={fx:.1f}cm y={fy:.1f}cm")
            state.second_buffer.clear()
            state.last_second_time = time.time()

        # 状态栏
        calib_label = "Calibration: Locked ✅" if state.calibrated else "Calibration: Pending"
        calib_color = (0,255,0) if state.calibrated else (0,0,255)
        cv2.putText(frame, calib_label, (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, calib_color, 2)
        step_text = {0:"未开始(按o/x/y或p)",1:"已标原点→按x",2:"已标X轴→按y",3:"已标Y轴",4:"完成✅"}
        cv2.putText(frame, f"标定: {step_text.get(state.calib_step,'')}",
                    (50,80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2)
        cv2.putText(frame, "P:平面  O/X/Y:轴系  R:重置  D:调试  ESC:退出",
                    (6, FRAME_HEIGHT-8), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180,180,180), 1)

        cv2.imshow(WINDOW_NAME, frame)
        cv2.setMouseCallback(WINDOW_NAME, mouse_callback, param=frame)

        key = cv2.waitKey(1) & 0xFF
        if   key == 27:        break
        elif key == ord('p'):  state.mode="plane";  state.calib_points.clear(); state.plane_calib_count=0; print("📌 平面标定：请点击4个点")
        elif key == ord('o'):  state.mode="origin"; print("📌 点击设置原点")
        elif key == ord('x'):  state.mode="x_axis"; print("📌 点击设置X轴端点")
        elif key == ord('y'):  state.mode="y_axis"; print("📌 点击设置Y轴端点")
        elif key == ord('r'):  state.reset();       print("🔄 标定已重置")
        elif key == ord('d'):
            debug_ellipse = not debug_ellipse
            print(f"调试模式: {'开' if debug_ellipse else '关'}")

# -------------------- 入口 --------------------
def main():
    model = load_yolo()
    cap   = init_camera()
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL | cv2.WINDOW_GUI_EXPANDED)
    cv2.resizeWindow(WINDOW_NAME, FRAME_WIDTH, FRAME_HEIGHT)
    try:
        run(cap, model)
    except Exception as e:
        print(f"💥 {e}"); traceback.print_exc()
    finally:
        if cap: cap.release()
        cv2.destroyAllWindows()
        print("✅ 退出")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"💥 {e}"); traceback.print_exc(); input("按回车退出...")
