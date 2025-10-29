from flask import Flask, render_template, request, send_file
import subprocess, joblib, os, io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error, r2_score

app = Flask(__name__)

# Nếu models chưa có thì chạy train_models.py
if not os.path.exists("models") or not all(
    os.path.exists(f"models/{m}.pkl") for m in ["logistic", "tree", "knn", "svm"]
):
    print("⚠️ Chưa có model, đang huấn luyện...")
    subprocess.run(["python", "train_models.py"], check=True)
    print("✅ Huấn luyện xong, tiếp tục load model...")

# Load models + scaler + encoders
models = {
    "Logistic Regression": joblib.load("models/logistic.pkl"),
    "Decision Tree": joblib.load("models/tree.pkl"),
    "KNN": joblib.load("models/knn.pkl"),
    "SVM": joblib.load("models/svm.pkl"),
}
scaler = joblib.load("models/scaler.pkl")
gender_encoder = joblib.load("models/gender_encoder.pkl")
smoking_encoder = joblib.load("models/smoking_history_encoder.pkl")

# Đọc dữ liệu để đánh giá
data = pd.read_csv("diabetes.csv")
for col in ["gender", "smoking_history"]:
    le = joblib.load(f"models/{col}_encoder.pkl")
    data[col] = le.transform(data[col])

X = data.drop("diabetes", axis=1)
y = data["diabetes"]
feature_names = list(X.columns)

# Metrics
model_scores = {}
X_scaled = scaler.transform(X)
for name, model in models.items():
    y_pred = model.predict(X_scaled)
    acc = accuracy_score(y, y_pred)
    mae = mean_absolute_error(y, y_pred)
    mse = mean_squared_error(y, y_pred)
    r2 = r2_score(y, y_pred)
    model_scores[name] = {"ACC": acc, "MAE": mae, "MSE": mse, "R2": r2}


def _encoder_mapping(le):
    # Trả về dict label -> mã số
    return {cls: int(i) for i, cls in enumerate(le.classes_)}


@app.route("/", methods=["GET", "POST"])
def index():
    probs = {}
    trace = None  # thông tin từng bước để hiển thị
    if request.method == "POST":
        # 1) Lấy input ở dạng gốc (raw)
        raw = {
            "gender": request.form["gender"],
            "age": float(request.form["age"]),
            "hypertension": int(request.form["hypertension"]),
            "heart_disease": int(request.form["heart_disease"]),
            "smoking_history": request.form["smoking_history"],
            "bmi": float(request.form["bmi"]),
            "HbA1c_level": float(request.form["HbA1c_level"]),
            "blood_glucose_level": float(request.form["blood_glucose_level"]),
        }

        # 2) Encode categorical theo LabelEncoder đã lưu
        enc_map = {
            "gender": _encoder_mapping(gender_encoder),
            "smoking_history": _encoder_mapping(smoking_encoder),
        }
        gender_val = int(gender_encoder.transform([raw["gender"]])[0])
        smoking_val = int(smoking_encoder.transform([raw["smoking_history"]])[0])

        encoded_row = [
            gender_val,
            raw["age"],
            raw["hypertension"],
            raw["heart_disease"],
            smoking_val,
            raw["bmi"],
            raw["HbA1c_level"],
            raw["blood_glucose_level"],
        ]

        # 3) Chuẩn hoá bằng MinMaxScaler đã fit trên tập train
        row_np = np.array([encoded_row])
        row_scaled = scaler.transform(row_np)[0].tolist()

        scaler_info = {
            "data_min_": scaler.data_min_.tolist(),
            "data_max_": scaler.data_max_.tolist(),
            "scale_": scaler.scale_.tolist(),
            "min_": scaler.min_.tolist(),
        }

        # 4) Dự đoán từng model
        for name, model in models.items():
            prob = float(model.predict_proba(row_np if name == "Decision Tree" else scaler.transform(row_np))[0][1]) \
                   if hasattr(model, "predict_proba") else 0.0
            pred = int(model.predict(row_np if name == "Decision Tree" else scaler.transform(row_np))[0])
            # Lưu ý: tất cả các model đều đã train trên dữ liệu đã scale.
            # Tuy nhiên quyết định chuẩn: dùng row_scaled cho nhất quán
            prob = float(model.predict_proba([row_scaled])[0][1]) if hasattr(model, "predict_proba") else 0.0
            pred = int(model.predict([row_scaled])[0])
            probs[name] = (pred, prob)

        # 5) Tạo trace để hiển thị "tường bước"
        trace = {
            "raw": raw,
            "encoders": enc_map,
            "encoded_row": {fn: val for fn, val in zip(feature_names, encoded_row)},
            "scaler_info": {fn: {"min": mn, "max": mx, "scale": sc, "min_offset": mi}
                            for fn, mn, mx, sc, mi in zip(
                                feature_names,
                                scaler_info["data_min_"],
                                scaler_info["data_max_"],
                                scaler_info["scale_"],
                                scaler_info["min_"],
                            )},
            "scaled_row": {fn: val for fn, val in zip(feature_names, row_scaled)},
            "threshold": 0.5,
        }

    return render_template(
        "index.html",
        results=probs,
        metrics=model_scores,
        feature_names=feature_names,
        trace=trace
    )


@app.route("/accuracy.png")
def accuracy_chart():
    fig, ax = plt.subplots()
    accs = {name: vals["ACC"] for name, vals in model_scores.items()}
    ax.bar(accs.keys(), [v * 100 for v in accs.values()])
    plt.ylabel("Accuracy (%)")
    plt.title("So sánh Accuracy giữa các mô hình")
    plt.ylim(0, 100)
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    return send_file(buf, mimetype="image/png")


@app.route("/chart.png")
def chart():
    """Vẽ biểu đồ xác suất cho input giả định"""
    sample = np.array([[gender_encoder.transform(["Male"])[0],
                        40, 0, 0,
                        smoking_encoder.transform(["never"])[0],
                        25, 5.5, 120]])
    sample_scaled = scaler.transform(sample)

    results = {}
    for name, model in models.items():
        prob = model.predict_proba(sample_scaled)[0][1]
        results[name] = prob

    fig, ax = plt.subplots()
    ax.bar(results.keys(), results.values())
    plt.ylabel("Xác suất mắc bệnh")
    plt.title("So sánh xác suất dự đoán (sample input)")
    plt.ylim(0, 1)
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    return send_file(buf, mimetype="image/png")


if __name__ == "__main__":
    app.run(debug=True)
