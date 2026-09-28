from fastapi import FastAPI
from fastapi.responses import HTMLResponse, FileResponse
from pathlib import Path
import pickle
import numpy as np

app = FastAPI(title="Iris SVM API")

BASE_DIR = Path(__file__).resolve().parent


# ==============================
# LOAD MODEL
# ==============================

MODEL_PATH = BASE_DIR / "model.pkl"

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)


# ==============================
# ĐỌC HTML
# ==============================

def read_html(filename):
    path = BASE_DIR / filename

    with open(path, "r", encoding="utf-8") as f:
        return f.read()


# ==============================
# CSS
# ==============================

@app.get("/style.css")
async def css():
    return FileResponse(BASE_DIR / "style.css")


# ==============================
# HOME
# ==============================

@app.get("/", response_class=HTMLResponse)
async def home():
    return read_html("index.html")


@app.get("/ui", response_class=HTMLResponse)
async def ui():
    return read_html("index.html")


# ==============================
# PREDICT
# ==============================

@app.get("/predict", response_class=HTMLResponse)
async def predict():
    return read_html("predict.html")


@app.get("/predict.html", response_class=HTMLResponse)
async def predict_html():
    return read_html("predict.html")


# ==============================
# VISUALIZATION
# ==============================

@app.get("/visualization", response_class=HTMLResponse)
async def visualization():
    return read_html("visualization.html")


@app.get("/visualization.html", response_class=HTMLResponse)
async def visualization_html():
    return read_html("visualization.html")


# ==============================
# SPECIES
# ==============================

@app.get("/species", response_class=HTMLResponse)
async def species():
    return read_html("species.html")


@app.get("/species.html", response_class=HTMLResponse)
async def species_html():
    return read_html("species.html")


# ==============================
# API PREDICT
# ==============================

@app.post("/api/predict")
async def api_predict(data: dict):

    try:

        features = np.array([[
            float(data["sepal_length"]),
            float(data["sepal_width"]),
            float(data["petal_length"]),
            float(data["petal_width"])
        ]])

        prediction = model.predict(features)[0]

        return {
            "success": True,
            "prediction": str(prediction)
        }

    except Exception as e:

        return {
            "success": False,
            "error": str(e)
        }


# ==============================
# HEALTH
# ==============================

@app.get("/health")
async def health():

    return {
        "status": "ok",
        "message": "Iris SVM API is running"
    }