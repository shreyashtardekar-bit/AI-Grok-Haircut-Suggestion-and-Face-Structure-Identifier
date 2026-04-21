# AI Haircut Stylist & Face Structure Identifier

An incredibly accurate, AI-powered stylist that uses Computer Vision to mathematically deduce your face shape and Large Language Models (LLMs) to recommend the most flattering haircuts.

## 🚀 Concept & Features
- **Mathematical Face Analysis:** Uses OpenCV to extract contour geometry, calculating proportions (Length-to-Width ratio, Jawline Taper) to accurately classify your face shape (Oval, Round, Square, Diamond, Oblong, or Heart).
- **Computer Vision Pipeline:** Includes dynamic Canny Edge detection, Haar Cascades for eye tracking, and contour smoothing.
- **Gender Detection:** Integrated a lightweight Caffe Deep Neural Network (DNN) model to classify gender from the detected facial crop.
- **AI Stylist via Groq API:** Uses Llama 3 (via Groq for lightning-fast inference) as an expert master stylist. It receives your exact face shape and gender to output personalized, trending haircut recommendations that balance your facial proportions.
- **Interactive UI:** A highly polished, premium custom frontend using Vanilla JS with an intuitive Drag-and-Drop image upload.

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
