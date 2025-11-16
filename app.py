from flask import Flask, render_template, request
import cv2
import numpy as np
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
OUTPUT_FOLDER = "static/output"

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["OUTPUT_FOLDER"] = OUTPUT_FOLDER

# Ensure folders exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)


# ---------------------------
# Lane Detection Pipeline
# ---------------------------
def grayscale(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def gaussian_blur(img, kernel_size=5):
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def canny(img, low_thresh=50, high_thresh=150):
    return cv2.Canny(img, low_thresh, high_thresh)

def region_of_interest(img):
    height, width = img.shape[:2]
    polygon = np.array([[
        (int(0.1 * width), height),
        (int(0.45 * width), int(0.6 * height)),
        (int(0.55 * width), int(0.6 * height)),
        (width, height)
    ]], np.int32)

    mask = np.zeros_like(img)
    cv2.fillPoly(mask, polygon, 255)
    return cv2.bitwise_and(img, mask)

def draw_lines(img, lines):
    if lines is None:
        return

    left, right = [], []

    for line in lines:
        for x1, y1, x2, y2 in line:
            slope = (y2 - y1) / (x2 - x1 + 1e-6)

            # Ignore almost horizontal or insane lines
            if abs(slope) < 0.5 or abs(slope) > 5:
                continue

            if slope < 0:
                left.append((x1, y1, x2, y2))
            else:
                right.append((x1, y1, x2, y2))

    def make(points):
        if len(points) == 0:
            return None

        xs, ys = [], []
        for x1, y1, x2, y2 in points:
            xs += [x1, x2]
            ys += [y1, y2]

        # Fit slope + intercept
        m, b = np.polyfit(xs, ys, 1)

        y_bottom = img.shape[0]
        y_top = int(img.shape[0] * 0.6)

        x_bottom = int((y_bottom - b) / m)
        x_top = int((y_top - b) / m)

        return x_bottom, y_bottom, x_top, y_top

    left_lane = make(left)
    right_lane = make(right)

    if left_lane:
        cv2.line(img, (left_lane[0], left_lane[1]), (left_lane[2], left_lane[3]), (255, 0, 0), 8)
    if right_lane:
        cv2.line(img, (right_lane[0], right_lane[1]), (right_lane[2], right_lane[3]), (255, 0, 0), 8)


def process_image(img_path, out_path):
    image = cv2.imread(img_path)
    gray = grayscale(image)
    blur = gaussian_blur(gray)
    edges = canny(blur)
    roi = region_of_interest(edges)
    lines = cv2.HoughLinesP(roi, 2, np.pi/180, 50, minLineLength=40, maxLineGap=100)

    line_img = np.zeros_like(image)
    draw_lines(line_img, lines)

    output = cv2.addWeighted(image, 0.8, line_img, 1, 0)
    cv2.imwrite(out_path, output)


def process_video(video_path, out_path):
    cap = cv2.VideoCapture(video_path)

    # FIX: browser-compatible codec
    fourcc = cv2.VideoWriter_fourcc(*"avc1")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        fps = 30  # fallback FPS

    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        gray = grayscale(frame)
        blur = gaussian_blur(gray)
        edges = canny(blur)
        roi = region_of_interest(edges)
        lines = cv2.HoughLinesP(roi, 2, np.pi/180, 50, minLineLength=40, maxLineGap=100)

        line_img = np.zeros_like(frame)
        draw_lines(line_img, lines)

        output = cv2.addWeighted(frame, 0.8, line_img, 1, 0)
        out.write(output)

    cap.release()
    out.release()


# ---------------------------
# Flask Routes
# ---------------------------
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/upload_image", methods=["POST"])
def upload_image():
    file = request.files["image"]
    filename = secure_filename(file.filename)

    upload_path = os.path.join(UPLOAD_FOLDER, filename)
    output_path = os.path.join(OUTPUT_FOLDER, "output_" + filename)

    file.save(upload_path)
    process_image(upload_path, output_path)

    return render_template("index.html",
                           input_image=f"/{upload_path}",
                           output_image=f"/{output_path}")


@app.route("/upload_video", methods=["POST"])
def upload_video():
    file = request.files["video"]
    filename = secure_filename(file.filename)

    upload_path = os.path.join(UPLOAD_FOLDER, filename)
    output_path = os.path.join(OUTPUT_FOLDER, "output_" + filename)

    file.save(upload_path)
    process_video(upload_path, output_path)

    return render_template("index.html",
                           input_video=f"/{upload_path}",
                           output_video=f"/{output_path}")


if __name__ == "__main__":
    app.run(debug=True)
