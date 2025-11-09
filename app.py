# app.py - Optimized Version
from flask import Flask, request
import pandas as pd
import joblib

app = Flask(__name__)

# Load model & scaler ONCE at startup
print("Loading model and scaler...")
model = joblib.load("fraud_model.pkl")
scaler = joblib.load("scaler.pkl")
print("✅ Model loaded successfully!")

def safe_float(x):
    """Convert input to float, return 0.0 if invalid"""
    try:
        return float(str(x).strip())
    except:
        return 0.0

@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        # Get form inputs
        time = safe_float(request.form.get("time", 0))
        v1 = safe_float(request.form.get("v1", 0))
        v2 = safe_float(request.form.get("v2", 0))
        v3 = safe_float(request.form.get("v3", 0))
        v4 = safe_float(request.form.get("v4", 0))
        amount = safe_float(request.form.get("amount", 0))
        
        # Create dataframe
        input_data = pd.DataFrame({
            'Time': [time],
            'V1': [v1],
            'V2': [v2],
            'V3': [v3],
            'V4': [v4],
            'Amount': [amount]
        })
        
        # Scale and predict
        input_scaled = scaler.transform(input_data)
        prediction = model.predict(input_scaled)[0]
        
        # Show result
        result = "🚨 FRAUD DETECTED!" if prediction == 1 else "✅ LEGITIMATE"
        color = "red" if prediction == 1 else "green"
        
        return f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Result</title>
            <style>
                body {{ font-family: Arial; padding: 20px; background: #f5f5f5; margin: 0; }}
                .container {{ background: white; padding: 20px; border-radius: 8px; max-width: 400px; margin: 20px auto; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
                .result {{ color: {color}; font-size: 28px; font-weight: bold; text-align: center; margin: 20px 0; }}
                .details {{ background: #f9f9f9; padding: 15px; border-radius: 5px; font-size: 14px; }}
                button {{ background: #007bff; color: white; padding: 12px 24px; border: none; border-radius: 5px; cursor: pointer; font-size: 16px; width: 100%; margin-top: 15px; }}
                button:hover {{ background: #0056b3; }}
            </style>
        </head>
        <body>
            <div class="container">
                <h2>Prediction Result</h2>
                <div class="result">{result}</div>
                <div class="details">
                    <strong>Transaction Details:</strong><br>
                    Time: {time}<br>
                    V1: {v1}<br>
                    V2: {v2}<br>
                    V3: {v3}<br>
                    V4: {v4}<br>
                    Amount: ${amount}
                </div>
                <button onclick="window.location.href='/'">Check Another</button>
            </div>
        </body>
        </html>
        """
    
    # GET request - show form
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Fraud Detection</title>
        <style>
            body { font-family: Arial; padding: 20px; background: #f5f5f5; margin: 0; }
            .container { background: white; padding: 25px; border-radius: 8px; max-width: 400px; margin: 20px auto; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
            h2 { color: #333; margin-top: 0; }
            label { font-weight: bold; color: #555; font-size: 14px; }
            input[type="text"] { width: 100%; padding: 10px; margin: 5px 0 15px 0; border: 1px solid #ddd; border-radius: 4px; box-sizing: border-box; font-size: 14px; }
            input[type="submit"] { background: #28a745; color: white; padding: 12px; border: none; border-radius: 5px; cursor: pointer; font-size: 16px; width: 100%; font-weight: bold; }
            input[type="submit"]:hover { background: #218838; }
            .info { background: #e7f3ff; padding: 10px; border-radius: 5px; font-size: 13px; margin-bottom: 20px; border-left: 3px solid #007bff; }
        </style>
    </head>
    <body>
        <div class="container">
            <h2>🔒 Credit Card Fraud Detection</h2>
            <div class="info">Enter transaction details below</div>
            <form method="POST">
                <label>Time:</label>
                <input type="text" name="time" placeholder="100000" value="100000">
                
                <label>V1:</label>
                <input type="text" name="v1" placeholder="-1.359807" value="-1.359807">
                
                <label>V2:</label>
                <input type="text" name="v2" placeholder="-0.072781" value="-0.072781">
                
                <label>V3:</label>
                <input type="text" name="v3" placeholder="2.536347" value="2.536347">
                
                <label>V4:</label>
                <input type="text" name="v4" placeholder="1.378155" value="1.378155">
                
                <label>Amount ($):</label>
                <input type="text" name="amount" placeholder="149.62" value="149.62">
                
                <input type="submit" value="🔍 Check Transaction">
            </form>
        </div>
    </body>
    </html>
    """

if __name__ == "__main__":
    print("\n🚀 Starting Flask app...")
    print("📍 Open browser: http://127.0.0.1:5000\n")
    app.run(debug=True, port=5000, threaded=True)
