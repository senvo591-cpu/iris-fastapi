from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import joblib

app = FastAPI(title="Iris SVM API")

templates = Jinja2Templates(directory=".")

app.mount("/static", StaticFiles(directory="."), name="static")

model = joblib.load("model.pkl")


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {"request": request}
    )


@app.get("/predict", response_class=HTMLResponse)
def predict_page(request: Request):
    return templates.TemplateResponse(
        "predict.html",
        {"request": request}
    )


@app.get("/visualization", response_class=HTMLResponse)
def visualization_page(request: Request):
    return templates.TemplateResponse(
        "visualization.html",
        {"request": request}
    )


@app.get("/species", response_class=HTMLResponse)
def species_page(request: Request):
    return templates.TemplateResponse(
        "species.html",
        {"request": request}
    )


@app.get("/api")
def api_status():
    return {"message": "Iris SVM API is running"}