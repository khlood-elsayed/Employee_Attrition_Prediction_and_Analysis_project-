from flask import Flask, request, jsonify, render_template
import pandas as pd
import pickle
import google.generativeai as genai

app = Flask(__name__)

# Yes/No mapping
yes_no_map = {"Yes": 1, "No": 0}

# Load ML Model
with open("final_modell.pkl", "rb") as f:
    model = pickle.load(f)

# Gemini Config
genai.configure(api_key="AIzaSyCPUOBCz9tm62EDABZ30x67cLlllgFl8wM")
chat_model = genai.GenerativeModel("gemini-2.0-flash")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Map Yes/No inputs
        v1 = yes_no_map[request.form['Work-Life Balance_Good']]
        v2 = yes_no_map[request.form['Marital Status_Married']]
        v3 = yes_no_map[request.form['Marital Status_Single']]
        v4 = yes_no_map[request.form['Job Level_Senior']]
        v5 = yes_no_map[request.form['Remote Work_Yes']]

        # Prepare dataframe
        new_data = pd.DataFrame([{
            'Work-Life Balance_Good': v1,
            'Marital Status_Married': v2,
            'Marital Status_Single': v3,
            'Job Level_Senior': v4,
            'Remote Work_Yes': v5
        }])

        # Predict
        prediction = model.predict(new_data)[0]
        prediction_text = "stayed" if prediction == 1 else "Not stayed"
        return jsonify({"prediction": prediction_text})

    except Exception as e:
        print(e)
        return jsonify({"prediction": "Error: Unable to get prediction."})

@app.route('/chat', methods=['POST'])
def chat():
    try:
        user_message = request.json.get("message")
        if not user_message:
            return jsonify({"response": "Please send a message."})

        prompt = f"""
        You are an HR assistant. 
        If the user asks 'Will the employee stay or leave?', give a short, decisive answer like 
        'Leaning towards stayed' or 'Leaning towards not stayed', 
        and add 1-2 sentences explaining why based on HR factors (like work-life balance, seniority, remote work). 
        For all other HR questions, answer in a short, clear, HR-advisor style explanation (under 50 words). 
        User asked: {user_message}
        """

        response = chat_model.generate_content(prompt)
        return jsonify({"response": response.text})

    except Exception as e:
        print(e)
        return jsonify({"response": "Error: Unable to get response from Gemini API."})

if __name__ == "__main__":
    app.run(debug=True)
