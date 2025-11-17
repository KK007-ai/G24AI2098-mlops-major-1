# app.py
import joblib
import numpy as np
from PIL import Image
from flask import Flask, request, render_template_string

app = Flask(__name__)

# Load model (savedmodel.pth must exist in application root)
DATA = joblib.load("savedmodel.pth")
MODEL = DATA['model']

HTML = """
<!doctype html>
<title>Olivetti Face Classifier</title>
<h1>Upload a face image (will be converted to 64x64 grayscale)</h1>
<form method=post enctype=multipart/form-data>
  <input type=file name=file>
  <input type=submit value=Upload>
</form>
{% if pred is not none %}
  <h2>Predicted class: {{ pred }}</h2>
{% endif %}
"""

def preprocess_image(file_stream):
    img = Image.open(file_stream).convert('L')  # grayscale
    img = img.resize((64,64))
    arr = np.array(img, dtype=np.float32) / 255.0  # scale to 0-1
    flat = arr.flatten()
    return flat.reshape(1, -1)

@app.route("/", methods=["GET", "POST"])
def index():
    pred = None
    if request.method == "POST":
        uploaded = request.files.get('file')
        if not uploaded:
            pred = "No file uploaded"
        else:
            try:
                vec = preprocess_image(uploaded.stream)
                label = MODEL.predict(vec)[0]
                pred = int(label)
            except Exception as e:
                pred = f"Error: {e}"
    return render_template_string(HTML, pred=pred)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
