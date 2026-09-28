import joblib

from sklearn.datasets import load_iris

from sklearn.model_selection import train_test_split

from sklearn.pipeline import make_pipeline

from sklearn.preprocessing import StandardScaler

from sklearn.svm import SVC

from sklearn.metrics import (
    accuracy_score,
    classification_report
)


# =====================================================
# LOAD DATA
# =====================================================

iris = load_iris()

X = iris.data
y = iris.target


# =====================================================
# CHIA DỮ LIỆU
# =====================================================

X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,

    test_size=0.2,

    random_state=42,

    stratify=y
)


# =====================================================
# TẠO MÔ HÌNH SVM
# =====================================================

model = make_pipeline(

    StandardScaler(),

    SVC(
        kernel="rbf",
        probability=True
    )

)


# =====================================================
# HUẤN LUYỆN
# =====================================================

model.fit(
    X_train,
    y_train
)


# =====================================================
# ĐÁNH GIÁ
# =====================================================

y_pred = model.predict(X_test)

accuracy = accuracy_score(
    y_test,
    y_pred
)


print("=" * 55)

print("IRIS SVM MODEL")

print("=" * 55)

print()

print(
    f"Accuracy: {accuracy:.4f}"
)

print()

print(
    "Classification Report:"
)

print(
    classification_report(
        y_test,
        y_pred,
        target_names=iris.target_names
    )
)


# =====================================================
# LƯU MODEL
# =====================================================

joblib.dump(
    model,
    "svm_model.pkl"
)


joblib.dump(
    model,
    "model.pkl"
)


print()

print("=" * 55)

print("Đã lưu mô hình:")

print("svm_model.pkl")

print("model.pkl")

print("=" * 55)