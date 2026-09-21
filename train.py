from sklearn import datasets
from sklearn.svm import SVC
import joblib

# Load dữ liệu Iris
iris = datasets.load_iris()

X = iris.data
y = iris.target

# Tạo mô hình SVM
model = SVC(kernel="linear")

# Huấn luyện mô hình
model.fit(X, y)

# Lưu mô hình
joblib.dump(model, "svm_model.pkl")

print("Model saved!")