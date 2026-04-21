---
title: AI Haircut Stylist
emoji: ✂️
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 7860
---
# AI Haircut Stylist & Face Structure Identifier

An incredibly accurate, AI-powered stylist that uses Computer Vision to mathematically deduce your face shape and Large Language Models (LLMs) to recommend the most flattering haircuts.

## 🚀 Concept & Features
- **Mathematical Face Analysis:** Uses OpenCV to extract contour geometry, calculating proportions (Length-to-Width ratio, Jawline Taper) to accurately classify your face shape (Oval, Round, Square, Diamond, Oblong, or Heart).
- **Computer Vision Pipeline:** Includes dynamic Canny Edge detection, Haar Cascades for eye tracking, and contour smoothing.
- **Gender Detection:** Integrated a lightweight Caffe Deep Neural Network (DNN) model to classify gender from the detected facial crop.
- **AI Stylist via Groq API:** Uses Llama 3 (via Groq for lightning-fast inference) as an expert master stylist. It receives your exact face shape and gender to output personalized, trending haircut recommendations that balance your facial proportions.
- **Interactive UI:** A highly polished, premium custom frontend using Vanilla JS with an intuitive Drag-and-Drop image upload.

## 🔍 Deep Dive: Image Processing Pipeline

The app uses classic computer vision techniques instead of a heavy neural network for face shape classification. Here is the step-by-step pipeline:

1. **Preprocessing**: The uploaded image is converted to Grayscale and passed through a `cv2.GaussianBlur` to reduce noise and unnecessary details.
2. **Dynamic Edge Detection**: We compute the median pixel intensity of the image and calculate custom upper and lower thresholds based on the median to feed into `cv2.Canny`. This ensures the edges are dynamically captured regardless of lighting conditions.
3. **Morphological Dilation**: Small gaps in the detected edges are closed using `cv2.dilate` with a rectangular kernel, ensuring facial outlines run continuously.
4. **Contour Extraction & Smoothing**: `cv2.findContours` identifies the largest, most prominent shape structure. It is then smoothed mathematically using `cv2.approxPolyDP` to form a geometric bounding representation of the jawline and face.
5. **Geometry & Eye Tracking**: The absolute face length ($H$) and width ($W$) are extracted via bounding boxes. We utilize Haar Cascades (`haarcascade_eye.xml`) to calculate specific forehead/upper-face width.
6. **Shape Classification Logic**: 
    - **Jawline Taper Metric**: We measure the area of the smoothed contour divided by the rectangular bounding box area ($W \times H$).
    - **Length-to-Width Ratio**: A high ratio ($>1.35$) helps differentiate Oblong, Oval, and Diamond shapes, whereas broad ratios assist in finding Square, Round, and Heart shapes based on the Jawline Taper Extent.

## 🛠️ Technology Stack
- **Backend**: Python, FastAPI, Uvicorn 
- **AI & Vision**: OpenCV, Groq API (Llama 3 70b-versatile)
- **Frontend**: HTML5, Vanilla CSS, JavaScript
- **Deployment**: Localhost (Waitress/Uvicorn & HTTP Server)

## ⚙️ Installation & Usage

### 1. Prerequisites
- Python 3.10+
- A [Groq API Key](https://console.groq.com/) for the AI recommendations.

### 2. Setup the Backend
Navigate to the `backend` directory and install the requirements:
```bash
cd backend
pip install -r requirements.txt
```

Create a `.env` file in the root backend directory (or use the provided `.env.example`) and add your Groq API Key:
```env
GROQ_API_KEY=gsk_your_api_key_here
```

Start the FastAPI server:
```bash
python -m uvicorn main:app --reload --host 127.0.0.1 --port 8000
```

### 3. Setup the Frontend
Open a new terminal, navigate to the `frontend` directory, and start a local static server:
```bash
cd frontend
python -m http.server 5500
```

### 4. Try it out!
Visit `http://localhost:5500` in your web browser. Drag and drop a clear photo of your face, and the AI will analyze it to provide stylist recommendations!
