from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path
import joblib
import numpy as np

app = FastAPI(
    title="IRIS AI",
    description="Iris Flower Classification using SVM",
    version="1.0"
)

BASE_DIR = Path(__file__).resolve().parent


# =========================================================
# LOAD MODEL
# =========================================================

MODEL_PATH = BASE_DIR / "svm_model.pkl"
MODEL_PATH_2 = BASE_DIR / "model.pkl"


def load_model():
    """
    Đọc mô hình SVM.
    Nếu svm_model.pkl không tồn tại thì thử model.pkl.
    """

    if MODEL_PATH.exists():
        return joblib.load(MODEL_PATH)

    if MODEL_PATH_2.exists():
        return joblib.load(MODEL_PATH_2)

    # Nếu chưa có model thì tự huấn luyện
    from sklearn import datasets
    from sklearn.model_selection import train_test_split
    from sklearn.svm import SVC
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    iris = datasets.load_iris()

    X = iris.data
    y = iris.target

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    model = make_pipeline(
        StandardScaler(),
        SVC(kernel="rbf", probability=True)
    )

    model.fit(X_train, y_train)

    joblib.dump(model, MODEL_PATH)
    joblib.dump(model, MODEL_PATH_2)

    return model


model = load_model()


# =========================================================
# TÊN CÁC LOÀI HOA
# =========================================================

SPECIES = {
    0: "Iris Setosa",
    1: "Iris Versicolor",
    2: "Iris Virginica"
}


SPECIES_INFO = {
    "Iris Setosa": {
        "description":
            "Setosa là một trong ba loài hoa Iris trong bộ dữ liệu. "
            "Loài này thường có kích thước cánh hoa nhỏ hơn rõ rệt "
            "so với Versicolor và Virginica.",
        "image":
            "https://commons.wikimedia.org/wiki/Special:Redirect/file/Iris_setosa.JPG"
    },

    "Iris Versicolor": {
        "description":
            "Versicolor có kích thước trung gian trong bộ dữ liệu Iris. "
            "Các đặc trưng về chiều dài và chiều rộng cánh hoa "
            "thường nằm giữa Setosa và Virginica.",
        "image":
            "https://commons.wikimedia.org/wiki/Special:Redirect/file/Iris_versicolor.jpg"
    },

    "Iris Virginica": {
        "description":
            "Virginica thường có cánh hoa lớn hơn hai nhóm còn lại. "
            "Đặc trưng về petal length và petal width giúp mô hình "
            "phân biệt nhóm này trong dữ liệu Iris.",
        "image":
            "https://commons.wikimedia.org/wiki/Special:Redirect/file/Iris_virginica_-_NRCS.jpg"
    }
}


# =========================================================
# CSS
# =========================================================

@app.get("/style.css")
async def style():
    return FileResponse(
        BASE_DIR / "style.css",
        media_type="text/css"
    )


# =========================================================
# HOME
# =========================================================

@app.get("/", response_class=HTMLResponse)
async def home():

    file_path = BASE_DIR / "index.html"

    return FileResponse(
        file_path,
        media_type="text/html"
    )


# =========================================================
# PREDICT - GET
# =========================================================

@app.get("/predict", response_class=HTMLResponse)
async def predict_page():

    file_path = BASE_DIR / "predict.html"

    return FileResponse(
        file_path,
        media_type="text/html"
    )


# =========================================================
# PREDICT - POST
# =========================================================

@app.post("/predict", response_class=HTMLResponse)
async def predict(
    request: Request,
    sepal_length: float = Form(...),
    sepal_width: float = Form(...),
    petal_length: float = Form(...),
    petal_width: float = Form(...)
):

    values = [
        sepal_length,
        sepal_width,
        petal_length,
        petal_width
    ]

    # -----------------------------------------
    # KIỂM TRA DỮ LIỆU
    # -----------------------------------------

    if any(value <= 0 for value in values):

        return HTMLResponse(
            content=f"""
            <html>
            <head>
                <meta http-equiv="refresh" content="0; url=/predict">
            </head>
            </html>
            """
        )

    # -----------------------------------------
    # DỰ ĐOÁN
    # -----------------------------------------

    X = np.array([values])

    prediction = model.predict(X)[0]

    species = SPECIES[int(prediction)]

    # -----------------------------------------
    # XÁC SUẤT
    # -----------------------------------------

    confidence = None

    try:
        probability = model.predict_proba(X)[0]
        confidence = float(np.max(probability) * 100)
    except Exception:
        confidence = None

    image = SPECIES_INFO[species]["image"]
    description = SPECIES_INFO[species]["description"]

    # -----------------------------------------
    # TRẢ VỀ HTML
    # -----------------------------------------

    html = f"""
    <!DOCTYPE html>

    <html lang="vi">

    <head>

        <meta charset="UTF-8">

        <meta name="viewport"
              content="width=device-width, initial-scale=1.0">

        <title>Dự đoán - IRIS AI</title>

        <link rel="stylesheet"
              href="/style.css">

    </head>

    <body>

        <header class="navbar">

            <div class="nav-container">

                <a href="/" class="brand">

                    <div class="brand-flower">
                        🌸
                    </div>

                    <div>
                        <strong>IRIS AI</strong>

                        <span>
                            SVM FLOWER CLASSIFICATION
                        </span>
                    </div>

                </a>

                <nav>

                    <a href="/">
                        Home
                    </a>

                    <a href="/predict" class="active">
                        Predict
                    </a>

                    <a href="/visualization">
                        Visualization
                    </a>

                    <a href="/species">
                        Species
                    </a>

                </nav>

            </div>

        </header>


        <main class="page">

            <section class="page-header">

                <span class="eyebrow">
                    SVM PREDICTION
                </span>

                <h1>
                    Kết quả dự đoán
                </h1>

                <p>
                    Mô hình Support Vector Machine đã phân tích
                    bốn thông số hình thái của mẫu hoa.
                </p>

            </section>


            <section class="result-layout">

                <div class="result-card">

                    <div class="result-image">

                        <img
                            src="{image}"
                            alt="{species}"
                        >

                    </div>

                    <div class="result-content">

                        <span class="result-label">
                            PREDICTED SPECIES
                        </span>

                        <h2>
                            {species}
                        </h2>

                        <p>
                            {description}
                        </p>

                        {
                            f'''
                            <div class="confidence">
                                Độ tin cậy:
                                <strong>{confidence:.2f}%</strong>
                            </div>
                            '''
                            if confidence is not None else ""
                        }

                    </div>

                </div>


                <div class="input-summary">

                    <h3>
                        Thông số đầu vào
                    </h3>

                    <div class="summary-grid">

                        <div>
                            <span>Sepal Length</span>
                            <strong>{sepal_length} cm</strong>
                        </div>

                        <div>
                            <span>Sepal Width</span>
                            <strong>{sepal_width} cm</strong>
                        </div>

                        <div>
                            <span>Petal Length</span>
                            <strong>{petal_length} cm</strong>
                        </div>

                        <div>
                            <span>Petal Width</span>
                            <strong>{petal_width} cm</strong>
                        </div>

                    </div>

                    <a
                        href="/predict"
                        class="btn primary full"
                    >
                        ✦ Dự đoán mẫu khác
                    </a>

                </div>

            </section>

        </main>

    </body>

    </html>
    """

    return HTMLResponse(content=html)


# =========================================================
# VISUALIZATION
# =========================================================

@app.get("/visualization", response_class=HTMLResponse)
async def visualization():

    file_path = BASE_DIR / "visualization.html"

    return FileResponse(
        file_path,
        media_type="text/html"
    )


# =========================================================
# SPECIES
# =========================================================

@app.get("/species", response_class=HTMLResponse)
async def species():

    file_path = BASE_DIR / "species.html"

    return FileResponse(
        file_path,
        media_type="text/html"
    )


# =========================================================
# HEALTH CHECK
# =========================================================

@app.get("/health")
async def health():

    return {
        "status": "online",
        "model": "SVM",
        "project": "IRIS AI"
    }