from flask import Flask, render_template, request
import numpy as np
import joblib

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input


app = Flask(__name__)


# ==========================================
# Load scaler
# ==========================================

scaler = joblib.load("scaler.pkl")


# ==========================================
# Rebuild the ANN architecture
# ==========================================

model = Sequential([
    Input(shape=(13,)),
    Dense(1000, activation="relu"),
    Dense(500, activation="relu"),
    Dense(250, activation="relu"),
    Dense(1, activation="linear")
])


# Load trained weights
model.load_weights("model_weights.weights.h5")


# ==========================================
# Home page
# ==========================================

@app.route("/")
def home():

    return render_template("index.html")


# ==========================================
# Prediction
# ==========================================

@app.route("/predict", methods=["POST"])
def predict():

    try:

        # Numerical features
        longitude = float(request.form["longitude"])
        latitude = float(request.form["latitude"])
        housing_median_age = float(
            request.form["housing_median_age"]
        )
        total_rooms = float(request.form["total_rooms"])
        total_bedrooms = float(
            request.form["total_bedrooms"]
        )
        population = float(request.form["population"])
        households = float(request.form["households"])
        median_income = float(
            request.form["median_income"]
        )

        ocean_proximity = request.form["ocean_proximity"]


        # ======================================
        # One-Hot Encoding
        # ======================================

        ocean_less_1h = 0
        ocean_inland = 0
        ocean_island = 0
        ocean_near_bay = 0
        ocean_near_ocean = 0


        if ocean_proximity == "<1H OCEAN":
            ocean_less_1h = 1

        elif ocean_proximity == "INLAND":
            ocean_inland = 1

        elif ocean_proximity == "ISLAND":
            ocean_island = 1

        elif ocean_proximity == "NEAR BAY":
            ocean_near_bay = 1

        elif ocean_proximity == "NEAR OCEAN":
            ocean_near_ocean = 1


        # ======================================
        # 13 features used during training
        # ======================================

        features = np.array([[

            longitude,
            latitude,
            housing_median_age,
            total_rooms,
            total_bedrooms,
            population,
            households,
            median_income,

            ocean_less_1h,
            ocean_inland,
            ocean_island,
            ocean_near_bay,
            ocean_near_ocean

        ]], dtype=float)


        # Scale input
        features_scaled = scaler.transform(features)


        # ANN prediction
        prediction = model.predict(
            features_scaled,
            verbose=0
        )


        price = float(prediction[0][0])


        return render_template(
            "index.html",
            prediction_text=f"${price:,.0f}",
            submitted=request.form
        )


    except Exception as error:

        return render_template(
            "index.html",
            error_text=str(error),
            submitted=request.form
        )


# ==========================================
# Run Flask
# ==========================================

if __name__ == "__main__":

    app.run(debug=True)