from fastapi.testclient import TestClient
from main import app
import json

def run_test():
    print("\n🚀 Memulai Uji Coba Simulasi What-If Lab...")
    
    # Skenario: User bergaji 10jt ingin kredit barang 4.5jt selama 6 bulan
    payload = {
        "user_profile": {
            "age": 28,
            "total_income": 10000000,
            "monthly_expenses": 5000000,
            "current_savings": 20000000,
            "has_emergency_fund": 1,
            "emergency_fund_months": 4,
            "has_kpr": 0,
            "has_vehicle_credit": 0,
            "pinjol_active": 0,
            "total_debt": 0,
            "credit_card_utilization": 0.10,
            "financial_literacy_score": 80,
            "employment_type": "permanent",
            "city_tier": "tier_1",
            "paylater_usage_history": "never",
            "impulse_spending_tendency": "low"
        },
        "simulation": {
            "item_price": 4500000,
            "available_cash": 5000000,
            "paylater_interest_rate": 0.02,
            "paylater_tenor_months": 6
        }
    }

    # GUNAKAN 'with' AGAR PROSES BOOTING (LOAD .KERAS) BERJALAN
    with TestClient(app) as client:
        # Tembak API
        response = client.post("/predict/whatif", json=payload)
        
        print("\n=======================================================")
        print("🔴 DEBUG OUTPUT JSON RESMI DARI FASTAPI (.keras):")
        print("=======================================================")
        
        if response.status_code == 200:
            print(json.dumps(response.json(), indent=2))
            print("=======================================================")
            print("✅ STATUS: PASS (200 OK)")
        else:
            print(f"❌ ERROR {response.status_code}: {response.text}")
            print("=======================================================")

if __name__ == "__main__":
    run_test()