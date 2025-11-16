# 🚘 Lane Detection Dashboard
A simple Flask-based web application for lane detection in images and videos using OpenCV.
Users can upload images or videos, and the app processes them to display lane-marked output.

## ⭐ Features
- Upload images and detect road lanes
- Upload videos and process them frame-by-frame
- Clean visual dashboard for input/output results
- Hough Transform-based lane detection pipeline
- Works locally in any Python environment

## 📸 Demo
- Image Processing
 Upload → Detect Lanes → View Output
- Video Processing
 Upload → Real-time frame processing → Render output video

## 🚀 Technologies Used
- Python
- Flask
- OpenCV
- HTML/CSS
- NumPy

## 📦 Installation
### 1️⃣ Clone the Repository
```bash
git clone https://github.com/your-username/lane-detection-dashboard.git
cd lane-detection-dashboard
```

### Install Requirements
Create a requirements.txt:
- flask
- opencv-python
- numpy


### Then install:
```bash

pip install -r requirements.txt

▶️ Run the Flask App
python app.py
```

### Open in your browser:
```bash
http://127.0.0.1:5000/
```

### 📁 Project Structure
static/
 ├── uploads/   → User uploaded files
 ├── output/    → Processed output files
templates/
 └── index.html
app.py

🧠 Lane Detection Pipeline

The project uses:

✔ Canny Edge Detection
✔ Gaussian Blur
✔ Polygon ROI Masking
✔ Probabilistic Hough Lines
✔ Slope-based left/right lane separation

🔧 Future Improvements

Use Deep Learning (YOLOv8, UNet) for lane detection

Add smoothing over frames

Deploy using Docker

Add progress bar for video processing

🤝 Contributing

Pull requests are welcome!

📜 License

MIT License
