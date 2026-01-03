<<<<<<< HEAD
from flask import Flask, render_template, request, jsonify
import model  # ML model file

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/classifier')
def classifier():
    return render_template('classifier.html')

@app.route('/dataset')
def dataset():
    return render_template('dataset.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.is_json:
        data = request.get_json()
        user_message = data.get("message", "")
        prediction = model.predict(user_message)
        return jsonify({"prediction": prediction})
    else:
        # fallback for normal form submission
        user_message = request.form['message']
        prediction = model.predict(user_message)
        return render_template('classifier.html', prediction=prediction, user_message=user_message)

if __name__ == "__main__":
    app.run(debug=True)
=======
from flask import Flask, render_template, request, jsonify
import model  # ML model file

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/classifier')
def classifier():
    return render_template('classifier.html')

@app.route('/dataset')
def dataset():
    return render_template('dataset.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.is_json:
        data = request.get_json()
        user_message = data.get("message", "")
        prediction = model.predict(user_message)
        return jsonify({"prediction": prediction})
    else:
        # fallback for normal form submission
        user_message = request.form['message']
        prediction = model.predict(user_message)
        return render_template('classifier.html', prediction=prediction, user_message=user_message)

if __name__ == "__main__":
    app.run(debug=True)
>>>>>>> 24f5cb8 (Initial commit: Flask spam classifier app)
