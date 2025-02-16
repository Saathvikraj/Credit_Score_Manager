from flask import Flask, render_template, request
import pandas as pd
import joblib

# Load the saved model
model = joblib.load("credit_scoring_model.pkl")

# Initialize Flask app
app = Flask(__name__)

# Home route
@app.route("/")
def home():
    return render_template("index.html")

# Prediction route
@app.route("/predict", methods=["POST"])
def predict():
    # Get input data from the form
    input_data = {
        "checking_account": request.form["checking_account"],
        "duration": int(request.form["duration"]),
        "credit_history": request.form["credit_history"],
        "purpose": request.form["purpose"],
        "credit_amount": int(request.form["credit_amount"]),
        "savings_account": request.form["savings_account"],
        "employment": request.form["employment"],
        "installment_rate": int(request.form["installment_rate"]),
        "personal_status": request.form["personal_status"],
        "debtors": request.form["debtors"],
        "residence": int(request.form["residence"]),
        "property": request.form["property"],
        "age": int(request.form["age"]),
        "other_installment_plans": request.form["other_installment_plans"],
        "housing": request.form["housing"],
        "existing_credits": int(request.form["existing_credits"]),
        "job": request.form["job"],
        "dependents": int(request.form["dependents"]),
        "telephone": request.form["telephone"],
        "foreign_worker": request.form["foreign_worker"]
    }

    # Convert input to DataFrame
    input_df = pd.DataFrame([input_data])

    # One-hot encode categorical variables
    input_df = pd.get_dummies(input_df)

    # Add missing columns with a value of 0
    for col in model.feature_names_in_:
        if col not in input_df.columns:
            input_df[col] = 0

    # Reorder columns to match training data
    input_df = input_df[model.feature_names_in_]

    # Make a prediction
    prediction = model.predict(input_df)[0]

    # Map prediction to human-readable labels
    if prediction == 1:
        result = "Good Credit Risk"
    else:
        result = "Bad Credit Risk"

    # Render the result in the template
    return render_template("index.html", prediction=result)

# Run the app
if __name__ == "__main__":
    app.run(debug=True)