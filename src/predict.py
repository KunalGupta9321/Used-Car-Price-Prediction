
# ============================================================
# Used Car Price Prediction Using Machine Learning Regression
# Interactive Prediction Script
# ============================================================

import joblib
import pandas as pd

from pathlib import Path


# 1. Define Project Paths

PROJECT_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = PROJECT_DIR / "models" / "used_car_price_model.joblib"


# 2. Load Trained Model

print("\n============================================")
print("       USED CAR PRICE PREDICTION SYSTEM")
print("============================================")

try:
    model = joblib.load(MODEL_PATH)
    print("\nModel loaded successfully!")

except FileNotFoundError:
    print("\nError: Trained model not found!")
    print("Please run the model training script first.")
    raise SystemExit


# 3. Get Input from User

print("\nEnter the following car details:\n")

try:
    year = int(input("Manufacturing Year: "))

    km_driven = float(input("Kilometres Driven: "))

    fuel = input("Fuel Type (Diesel/Petrol/CNG/LPG): ").strip()

    seller_type = input(
        "Seller Type (Individual/Dealer/Trustmark Dealer): "
    ).strip()

    transmission = input(
        "Transmission (Manual/Automatic): "
    ).strip()

    owner = input(
        "Owner Type (First Owner/Second Owner/Third Owner/"
        "Fourth & Above Owner/Test Drive Car): "
    ).strip()

    mileage = float(input("Mileage: "))

    engine = float(input("Engine Capacity (CC): "))

    max_power = float(input("Maximum Power (BHP): "))

    torque = float(input("Torque (Nm): "))

    seats = float(input("Number of Seats: "))

    manufacturer = input("Manufacturer: ").strip()

    mileage_unit = input("Mileage Unit (kmpl or km/kg): ").strip()


    # 4. Create Input DataFrame

    car_data = pd.DataFrame([{
        "year": year,
        "km_driven": km_driven,
        "fuel": fuel,
        "seller_type": seller_type,
        "transmission": transmission,
        "owner": owner,
        "mileage": mileage,
        "engine": engine,
        "max_power": max_power,
        "torque": torque,
        "seats": seats,
        "manufacturer": manufacturer,
        "mileage_unit": mileage_unit
    }])


    # 5. Generate Prediction

    predicted_price = model.predict(car_data)[0]


    # 6. Display Prediction

    print("\n============================================")
    print("             PREDICTION RESULT")
    print("============================================")

    print(f"\nEstimated Selling Price: ₹{predicted_price:,.2f}")

    print("\n============================================")
    print("Prediction completed successfully!")
    print("Note: This is an estimated market price.")
    print("============================================")


except ValueError:
    print("\nInvalid input! Please enter valid numeric values.")

except Exception as error:
    print("\nPrediction failed:", error)

