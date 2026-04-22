import os
import csv
import cv2
import time
import argparse
import numpy as np
from datetime import datetime

from ui_dashboard import draw_dashboard


def ensure_directories():
    os.makedirs("results", exist_ok=True)
    os.makedirs("results/screenshots", exist_ok=True)
    os.makedirs("assets", exist_ok=True)


def current_time_string():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def file_time_string():
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def init_csv_log(path="results/logs.csv"):
    if not os.path.exists(path):
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "label", "score", "severity", "x", "y", "w", "h"])


def append_csv_log(label, score, severity, bbox, path="results/logs.csv"):
    x, y, w, h = bbox
    with open(path, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([current_time_string(), label, f"{score:.3f}", severity, x, y, w, h])


def save_screenshot(frame, prefix="capture"):
    path = f"results/screenshots/{prefix}_{file_time_string()}.png"
    cv2.imwrite(path, frame)
    print(f"[INFO] Screenshot saved: {path}")
    return path


class MotionAnomalyDetector:
    def __init__(self, min_area=1800, anomaly_threshold=0.06, history=300, var_threshold=20):
        self.min_area = min_area
        self.anomaly_threshold = anomaly_threshold
        self.bg = cv2.createBackgroundSubtractorMOG2(
            history=history,
            varThreshold=var_threshold,
            detectShadows=True
        )

    def detect(self, frame):
        h, w = frame.shape[:2]
        frame_area = h * w

        fg_mask = self.bg.apply(frame)

        _, thresh = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)
        kernel = np.ones((5, 5), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        thresh = cv2.dilate(thresh, kernel, iterations=2)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        largest_area = 0
        largest_bbox = (0, 0, 0, 0)

        for contour in contours:
            area = cv2.contourArea(contour)
            if area > self.min_area and area > largest_area:
                x, y, bw, bh = cv2.boundingRect(contour)
                largest_area = area
                largest_bbox = (x, y, bw, bh)

        score = largest_area / frame_area

        if score > 0.20:
            severity = "CRITICAL"
        elif score > 0.12:
            severity = "HIGH"
        elif score > 0.08:
            severity = "MEDIUM"
        elif score > self.anomaly_threshold:
            severity = "LOW"
        else:
            severity = "NORMAL"

        is_anomaly = score > self.anomaly_threshold

        return is_anomaly, score, severity, largest_bbox, thresh


def parse_args():
    parser = argparse.ArgumentParser(description="EdgeAI - Docker anomaly detection project")
    parser.add_argument("--source", type=str, default="0", help="Camera index like 0, 1 or video path")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--save-video", action="store_true")
    return parser.parse_args()


def open_source(source):
    if source.isdigit():
        return cv2.VideoCapture(int(source))
    return cv2.VideoCapture(source)


def main():
    args = parse_args()

    ensure_directories()
    init_csv_log()

    cap = open_source(args.source)
    if not cap.isOpened():
        print("[ERROR] Could not open camera/video source.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    detector = MotionAnomalyDetector(
        min_area=1800,
        anomaly_threshold=0.06,
        history=300,
        var_threshold=20
    )

    event_history = []
    prev_time = time.time()
    last_auto_save = 0
    auto_save_gap = 3.0

    writer = None
    if args.save_video:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(
            "results/output_video.mp4",
            fourcc,
            20.0,
            (args.width, args.height)
        )

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[INFO] End of stream or camera read failed.")
            break

        frame = cv2.resize(frame, (args.width, args.height))

        is_anomaly, score, severity, bbox, _ = detector.detect(frame)

        now = time.time()
        fps = 1.0 / (now - prev_time) if now > prev_time else 0.0
        prev_time = now

        if is_anomaly and bbox[2] > 0 and bbox[3] > 0:
            event_text = f"{current_time_string()} | anomaly | {score:.3f} | {severity}"

            if not event_history or event_history[-1] != event_text:
                event_history.append(event_text)
                append_csv_log("anomaly", score, severity, bbox)

            if now - last_auto_save > auto_save_gap:
                temp_frame = frame.copy()
                draw_dashboard(temp_frame, is_anomaly, severity, score, bbox, fps, event_history)
                save_screenshot(temp_frame, prefix="anomaly")
                last_auto_save = now
        else:
            event_text = f"{current_time_string()} | normal | {score:.3f} | NORMAL"
            if not event_history or event_history[-1] != event_text:
                event_history.append(event_text)

        if len(event_history) > 30:
            event_history = event_history[-30:]

        output = frame.copy()
        draw_dashboard(output, is_anomaly, severity, score, bbox, fps, event_history)

        cv2.imshow("EdgeAI", output)

        if writer is not None:
            writer.write(output)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("s"):
            save_screenshot(output, prefix="manual")

    cap.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
