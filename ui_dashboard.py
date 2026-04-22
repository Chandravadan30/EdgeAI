import cv2
import psutil
from datetime import datetime

COLOR_BG = (10, 15, 20)
COLOR_PANEL = (18, 24, 30)
COLOR_CYAN = (255, 229, 0)
COLOR_WHITE = (230, 245, 255)
COLOR_GREEN = (80, 255, 120)
COLOR_YELLOW = (0, 220, 255)
COLOR_RED = (40, 40, 255)


def draw_transparent_rect(frame, pt1, pt2, color, alpha=0.18):
    overlay = frame.copy()
    cv2.rectangle(overlay, pt1, pt2, color, -1)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)


def draw_panel(frame, x, y, w, h, border_color=COLOR_CYAN, alpha=0.12):
    draw_transparent_rect(frame, (x, y), (x + w, y + h), COLOR_PANEL, alpha)
    cv2.rectangle(frame, (x, y), (x + w, y + h), border_color, 2)


def put_text(frame, text, x, y, color=COLOR_WHITE, scale=0.55, thickness=1):
    cv2.putText(
        frame,
        text,
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thickness,
        cv2.LINE_AA
    )


def draw_header(frame, width):
    draw_panel(frame, 10, 10, width - 20, 55, border_color=COLOR_CYAN, alpha=0.15)
    put_text(frame, "EDGEAI", 25, 45, COLOR_CYAN, 0.95, 2)
    put_text(frame, "Jetson Orin Nano Developer Kit", 200, 45, COLOR_WHITE, 0.6, 1)
    put_text(frame, "SYSTEM ONLINE", width - 220, 45, COLOR_GREEN, 0.65, 2)


def draw_live_feed_frame(frame, x, y, w, h):
    draw_panel(frame, x, y, w, h, border_color=COLOR_CYAN, alpha=0.06)
    put_text(frame, "LIVE ANALYSIS", x + 12, y + 28, COLOR_CYAN, 0.6, 2)

    corner = 18
    thickness = 2
    color = COLOR_CYAN

    cv2.line(frame, (x + 8, y + 40), (x + 8 + corner, y + 40), color, thickness)
    cv2.line(frame, (x + 8, y + 40), (x + 8, y + 40 + corner), color, thickness)

    cv2.line(frame, (x + w - 8, y + 40), (x + w - 8 - corner, y + 40), color, thickness)
    cv2.line(frame, (x + w - 8, y + 40), (x + w - 8, y + 40 + corner), color, thickness)

    cv2.line(frame, (x + 8, y + h - 8), (x + 8 + corner, y + h - 8), color, thickness)
    cv2.line(frame, (x + 8, y + h - 8), (x + 8, y + h - 8 - corner), color, thickness)

    cv2.line(frame, (x + w - 8, y + h - 8), (x + w - 8 - corner, y + h - 8), color, thickness)
    cv2.line(frame, (x + w - 8, y + h - 8), (x + w - 8, y + h - 8 - corner), color, thickness)


def draw_alert_panel(frame, x, y, w, h, is_anomaly, severity, score):
    border = COLOR_RED if is_anomaly else COLOR_GREEN
    draw_panel(frame, x, y, w, h, border_color=border, alpha=0.12)
    put_text(frame, "ALERT PANEL", x + 12, y + 28, border, 0.6, 2)

    if is_anomaly:
        put_text(frame, "STATUS: ANOMALY DETECTED", x + 12, y + 62, COLOR_RED, 0.58, 2)
        put_text(frame, f"SEVERITY: {severity}", x + 12, y + 92, COLOR_YELLOW, 0.55, 2)
        put_text(frame, f"SCORE: {score:.3f}", x + 12, y + 122, COLOR_WHITE, 0.55, 1)
    else:
        put_text(frame, "STATUS: NORMAL", x + 12, y + 62, COLOR_GREEN, 0.58, 2)
        put_text(frame, "SEVERITY: NONE", x + 12, y + 92, COLOR_WHITE, 0.55, 1)
        put_text(frame, f"SCORE: {score:.3f}", x + 12, y + 122, COLOR_WHITE, 0.55, 1)


def draw_status_panel(frame, x, y, w, h, fps):
    draw_panel(frame, x, y, w, h, border_color=COLOR_CYAN, alpha=0.12)
    put_text(frame, "EDGE DEVICE STATUS", x + 12, y + 28, COLOR_CYAN, 0.6, 2)

    cpu = psutil.cpu_percent()
    ram = psutil.virtual_memory().percent

    lines = [
        "DEVICE: Jetson Orin Nano Dev Kit",
        "MODEL: Motion-Based Anomaly Detector",
        "CAMERA: Connected",
        f"FPS: {fps:.2f}",
        f"CPU: {cpu:.1f}%",
        f"RAM: {ram:.1f}%"
    ]

    for i, line in enumerate(lines):
        put_text(frame, line, x + 12, y + 58 + i * 24, COLOR_WHITE, 0.5, 1)


def draw_risk_panel(frame, x, y, w, h, score):
    draw_panel(frame, x, y, w, h, border_color=COLOR_CYAN, alpha=0.12)
    put_text(frame, "RISK / CONFIDENCE", x + 12, y + 28, COLOR_CYAN, 0.6, 2)

    bar_x = x + 15
    bar_y = y + 55
    bar_w = w - 30
    bar_h = 22

    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), COLOR_WHITE, 1)

    normalized = min(score * 4.0, 1.0)
    fill_w = int(bar_w * normalized)

    fill_color = COLOR_GREEN
    if score > 0.06:
        fill_color = COLOR_YELLOW
    if score > 0.12:
        fill_color = COLOR_RED

    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h), fill_color, -1)

    put_text(frame, f"Current Score: {score:.3f}", x + 15, y + 105, COLOR_WHITE, 0.5, 1)


def draw_event_log(frame, x, y, w, h, events):
    draw_panel(frame, x, y, w, h, border_color=COLOR_GREEN, alpha=0.10)
    put_text(frame, "EVENT HISTORY", x + 12, y + 28, COLOR_GREEN, 0.6, 2)

    recent = events[-6:]
    for i, event in enumerate(recent):
        put_text(frame, event, x + 12, y + 58 + i * 24, COLOR_WHITE, 0.45, 1)


def draw_timestamp(frame):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    put_text(frame, now, 20, frame.shape[0] - 15, COLOR_WHITE, 0.5, 1)


def draw_detection_box(frame, bbox, score, severity):
    x, y, w, h = bbox
    if w <= 0 or h <= 0:
        return

    color = COLOR_RED if severity in ["HIGH", "CRITICAL"] else COLOR_YELLOW
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

    label = f"ANOMALY DETECTED | {score:.3f} | {severity}"
    label_w = max(260, len(label) * 8)
    top_y = max(0, y - 30)
    cv2.rectangle(frame, (x, top_y), (x + label_w, y), color, -1)
    put_text(frame, label, x + 6, y - 8, (0, 0, 0), 0.45, 1)


def draw_dashboard(frame, is_anomaly, severity, score, bbox, fps, event_history):
    height, width = frame.shape[:2]

    draw_header(frame, width)

    live_x, live_y, live_w, live_h = 20, 80, 840, 470
    right_x = 880

    draw_live_feed_frame(frame, live_x, live_y, live_w, live_h)
    draw_alert_panel(frame, right_x, 80, 370, 150, is_anomaly, severity, score)
    draw_status_panel(frame, right_x, 250, 370, 190, fps)
    draw_risk_panel(frame, right_x, 460, 370, 110, score)
    draw_event_log(frame, 20, 570, 1230, 130, event_history)
    draw_timestamp(frame)

    if is_anomaly:
        draw_detection_box(frame, bbox, score, severity)

    put_text(frame, "Press Q to quit | Press S to save screenshot", 20, height - 40, COLOR_WHITE, 0.52, 1)

    return frame
