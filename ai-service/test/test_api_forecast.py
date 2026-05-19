import sys
import os
import json
from fastapi.testclient import TestClient

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from main import app

def run_test():
    print("\nMemulai Uji Coba Model 2 Forecasting Engine...")
    
    # Payload dummy merepresentasikan data historis transaksi user
    payload_forecast = {
        "lag1_total_expense": 4200000.0,
        "lag2_total_expense": 3900000.0,
        "lag3_total_expense": 4100000.0,
        "roll3_mean_expense": 4066666.6,
        "roll6_mean_expense": 3950000.0,
        "roll3_std_expense": 152752.5,
        "lag1_monthly_income": 7500000.0,
        "lag1_savings_rate": 0.44,
        "lag1_expense_growth": 0.07,
        "bulan_sin": -0.5,
        "bulan_cos": 0.866,
        "persona_id": 1.0
    }

    with TestClient(app) as client:
        response = client.post("/predict/forecast", json=payload_forecast)
        
        print("\n=======================================================")
        print("DEBUG OUTPUT JSON RESMI DARI FASTAPI (.keras):")
        print("=======================================================")
        
        if response.status_code == 200:
            print(json.dumps(response.json(), indent=2))
            print("=======================================================")
            print("STATUS: PASS (200 OK)")
        else:
            print(f"ERROR {response.status_code}: {response.text}")
            print("=======================================================")

if __name__ == "__main__":
    run_test()