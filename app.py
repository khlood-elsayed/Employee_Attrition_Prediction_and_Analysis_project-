from flask import Flask, request, jsonify, render_template
import pandas as pd
import pickle

# إنشاء التطبيق
app = Flask(__name__)

# map للتحويل من Yes/No إلى 1/0
yes_no_map = {"Yes": 1, "No": 0}

# تحميل الموديل
with open("final_modell.pkl", "rb") as f:
    model = pickle.load(f)

# الصفحة الرئيسية
@app.route('/')
def home():
    return render_template('index.html')  # ضع index.html في فولدر templates

# API للتنبؤ
@app.route('/predict', methods=['POST'])
def predict():
    try:
        v1 = yes_no_map[request.form['Work-Life Balance_Good']]
        v2 = yes_no_map[request.form['Marital Status_Married']]
        v3 = yes_no_map[request.form['Marital Status_Single']]
        v4 = yes_no_map[request.form['Job Level_Senior']]
        v5 = yes_no_map[request.form['Remote Work_Yes']]

        new_data = pd.DataFrame([{
            'Work-Life Balance_Good': v1,
            'Marital Status_Married': v2,
            'Marital Status_Single': v3,
            'Job Level_Senior': v4,
            'Remote Work_Yes': v5
        }])

        prediction = model.predict(new_data)[0]
        prediction_text = "stayed" if prediction == 1 else "Not stayed"

        return jsonify({"prediction": prediction_text})
    except Exception as e:
        print(e)
        return jsonify({"prediction": "Error: Unable to get prediction."})

if __name__ == "__main__":
    app.run(debug=True)
