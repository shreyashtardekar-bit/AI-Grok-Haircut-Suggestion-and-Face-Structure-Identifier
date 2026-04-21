import os
import cv2
import numpy as np
import requests
import base64
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv

# Load Environment Variables from .env file
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

gender_list = ['Male', 'Female']
gender_net = None

def get_gender(face_img):
    global gender_net
    if gender_net is None:
        proto_path = os.path.join(os.path.dirname(__file__), "models", "gender_deploy.prototxt")
        model_path = os.path.join(os.path.dirname(__file__), "models", "gender_net.caffemodel")
        if os.path.exists(proto_path) and os.path.exists(model_path):
            gender_net = cv2.dnn.readNetFromCaffe(proto_path, model_path)
        else:
            return "Unknown (Model missing)"
            
    blob = cv2.dnn.blobFromImage(face_img, 1.0, (227, 227), (78.4263377603, 87.7689143744, 114.895847746), swapRB=False)
    gender_net.setInput(blob)
    preds = gender_net.forward()
    return gender_list[preds[0].argmax()]

app = FastAPI(title="AI Haircut Stylist API")

# Allow requests from the frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # For production, restrict this to the frontend URL!
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def get_haircut_recommendations(face_shape: str, gender: str) -> str:
    """
    Constructs a prompt securely and requests recommendations from the Groq API.
    """
    if not GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY is not configured in .env")

    prompt = (
        f"The user is {gender} with a {face_shape} face shape determined by an OpenCV bounding box analysis. "
        "Act as an expert master hair stylist. Recommend the top 3 best haircuts for this exact "
        "gender and face shape. For each cut, explain exactly why it balances their facial proportions and "
        "provide brief styling advice. Explicitly include any trending haircuts that would ideally suit them. "
        "Only output styling advice and trends, do not add filler intros."
    )

    url = "https://api.groq.com/openai/v1/chat/completions"
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {GROQ_API_KEY}"
    }

    payload = {
        "messages": [
            {"role": "system", "content": "You are a master hair stylist with expert knowledge in facial geometry."},
            {"role": "user", "content": prompt}
        ],
        "model": "llama-3.3-70b-versatile",
        "temperature": 0.5
    }

    response = requests.post(url, headers=headers, json=payload, timeout=30)
    response.raise_for_status() 
    data = response.json()
    return data['choices'][0]['message']['content']


def analyze_image(image_bytes: bytes):
    """
    Processes the uploaded image using classic CV algorithms to deduce Face Shape
    and returns a base64 encoded visual proof image.
    """
    file_bytes = np.asarray(bytearray(image_bytes), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    
    if img is None:
        raise ValueError("Invalid image format.")

    original_vis = img.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Dynamic Canny Edge Detection
    median_val = np.median(blurred)
    sigma = 0.33
    lower_thresh = int(max(0, (1.0 - sigma) * median_val))
    upper_thresh = int(min(255, (1.0 + sigma) * median_val))
    edges = cv2.Canny(blurred, lower_thresh, upper_thresh)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    edges = cv2.dilate(edges, kernel, iterations=1)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        raise ValueError("No prominent contours detected. Try a photo with a clearer background.")
        
    largest_contour = max(contours, key=cv2.contourArea)

    epsilon = 0.01 * cv2.arcLength(largest_contour, True)
    smoothed_contour = cv2.approxPolyDP(largest_contour, epsilon, True)

    x, y, w, h = cv2.boundingRect(smoothed_contour)
    
    # Mathematical Aspect Ratio & Advanced Geometry Logic
    upper_width = w * 0.75 # fallback
    try:
        eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        roi_gray = gray[max(0, y):y+h, max(0, x):x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 4)
        if len(eyes) >= 2:
            eyes = sorted(eyes, key=lambda e: e[0])
            left_eye_edge = eyes[0][0]
            right_eye_edge = eyes[-1][0] + eyes[-1][2]
            upper_width = float(right_eye_edge - left_eye_edge)
    except Exception:
        pass

    face_length = float(h)
    face_width = float(w)
    
    length_to_width = face_length / max(1.0, face_width)
    upper_to_width = upper_width / max(1.0, face_width)
    contour_area = cv2.contourArea(smoothed_contour)
    lower_jaw_taper_metric = contour_area / max(1.0, float(w * h)) # extent

    # Absolute strict identification rules
    if length_to_width > 1.35:
        if upper_to_width < 0.52:
            face_shape = "Diamond"
        else:
            if lower_jaw_taper_metric > 0.8:
                face_shape = "Oblong"
            else:
                face_shape = "Oval"
    else:
        if upper_to_width > 0.65 and lower_jaw_taper_metric < 0.78:
            face_shape = "Heart"
        elif upper_to_width < 0.52:
            face_shape = "Diamond"
        elif lower_jaw_taper_metric > 0.82:
            face_shape = "Square"
        else:
            face_shape = "Round"
            
    # Gender Prediction
    face_crop = img[max(0, y):y+h, max(0, x):x+w]
    gender = "Unknown"
    if face_crop.shape[0] > 0 and face_crop.shape[1] > 0:
        gender = get_gender(face_crop)
            
    # Visual Proof
    cv2.drawContours(original_vis, [smoothed_contour], -1, (0, 255, 0), 2)
    cv2.rectangle(original_vis, (x, y), (x+w, y+h), (0, 0, 255), 3)

    # Convert to Base64 to send to Frontend
    _, buffer = cv2.imencode('.jpg', original_vis)
    base64_img = base64.b64encode(buffer).decode('utf-8')
    
    return face_shape, gender, base64_img

@app.post("/api/analyze")
async def analyze_face(file: UploadFile = File(...)):
    """
    Endpoint that accepts an image and returns face shape + AI styling advice.
    """
    try:
        contents = await file.read()
        
        # 1. OpenCV Analysis
        try:
            face_shape, gender, b64_img = analyze_image(contents)
        except ValueError as ve:
            return JSONResponse(status_code=400, content={"error": str(ve)})

        # 2. LLM Stylist Call
        try:
            recommendations = get_haircut_recommendations(face_shape, gender)
        except Exception as e:
            return JSONResponse(status_code=500, content={"error": f"Groq API Error: {str(e)}", "shape": face_shape, "gender": gender, "image": b64_img})
            
        return JSONResponse(content={
            "shape": face_shape,
            "gender": gender,
            "image_base64": b64_img,
            "recommendations": recommendations,
            "logic_summary": (
                "Advanced Geometric Extraction mapping Forehead Width (via Cascades) vs Jawline Taper & Extent Metric. "
                "Calculates absolute proportions to conclusively classify Diamond, Heart, Oval, Oblong, Round, or Square. "
                "Gender detection via pre-trained OpenCV DNN Caffe model."
            )
        })
        
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"Server Error: {str(e)}"})

# Health check route
@app.get("/health")
def read_root():
    return {"status": "ok", "message": "AI Stylist Backend is running!"}

# Mount the frontend directory to serve the static HTML/JS/CSS files at the root
frontend_path = os.path.join(os.path.dirname(__file__), "..", "frontend")
if os.path.exists(frontend_path):
    app.mount("/", StaticFiles(directory=frontend_path, html=True), name="frontend")
