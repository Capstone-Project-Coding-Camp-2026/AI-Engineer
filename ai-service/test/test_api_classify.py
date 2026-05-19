from fastapi.testclient import TestClient
from main import app
import json

def run_test():
    print("\nMemulai Uji Coba Model 1 NLP...")
    
    # Payload super sederhana untuk Model 1
    payload_nlp = {
        "desc": "Beli kopi starbucks dan roti bakar"
    }

    with TestClient(app) as client:
        # Tembak API Classify
        response = client.post("/predict/classify", json=payload_nlp)
        
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