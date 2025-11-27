# Lane Detection Dashboard

A web-based application for detecting lane lines in images and videos using computer vision techniques. Built with Flask and OpenCV.

## 🚀 Features

- **Image Analysis**: Upload an image to detect and visualize lane lines.
- **Video Analysis**: Process video files to track lanes in real-time.
- **Side-by-Side Comparison**: View original media alongside the processed output with detected lanes overlaid.
- **Responsive UI**: Clean and modern dashboard interface.

## 🛠️ Tech Stack

- **Backend**: Python, Flask
- **Computer Vision**: OpenCV, NumPy
- **Frontend**: HTML5, CSS3

## ⚙️ Installation

1.  **Clone the repository** (if applicable) or navigate to the project directory.

2.  **Install dependencies**:
    Ensure you have Python installed, then run:
    ```bash
    pip install -r requirements.txt
    ```

## ▶️ Usage

1.  **Run the application**:
    ```bash
    python app.py
    ```

2.  **Open your browser**:
    Navigate to `http://127.0.0.1:5000/`.

3.  **Upload Media**:
    - Use the **Image Analysis** section to upload a road image.
    - Use the **Video Analysis** section to upload a driving video.
    - Click "Process" and wait for the results.

## 📂 Project Structure

```
Lane Detection Project/
├── app.py                # Main Flask application and lane detection logic
├── requirements.txt      # Python dependencies
├── static/               # Static assets (CSS, uploads, outputs)
│   ├── style.css
│   ├── uploads/          # Temporary storage for uploaded files
│   └── output/           # Storage for processed files
└── templates/
    └── index.html        # Main dashboard HTML template
```

## 🧠 How It Works

The lane detection pipeline follows these steps:
1.  **Grayscale Conversion**: Simplifies the image for processing.
2.  **Gaussian Blur**: Reduces noise.
3.  **Canny Edge Detection**: Identifies edges in the image.
4.  **Region of Interest (ROI)**: Focuses on the road area, masking out the sky and surroundings.
5.  **Hough Transform**: Detects line segments within the ROI.
6.  **Line Drawing**: Extrapolates and draws the left and right lane lines on the original image.
