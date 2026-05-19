from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import numpy as np

from app.core.config import model_registry

router = APIRouter(prefix="/predict", tags=["Forecasting Engine"])

# Skema Request sesuai dengan 12 FEATURE_COLS dari notebook kamu
class ForecastRequest(BaseModel):
    lag1_total_expense: float
    lag2_total_expense: float
    lag3_total_expense: float
    roll3_mean_expense: float
    roll6_mean_expense: float
    roll3_std_expense: float
    lag1_monthly_income: float
    lag1_savings_rate: float
    lag1_expense_growth: float
    bulan_sin: float
    bulan_cos: float
    persona_id: float

@router.post("/forecast")
def predict_future_expense(req: ForecastRequest):
    if model_registry.forecast_model is None or model_registry.forecast_metadata is None:
        raise HTTPException(status_code=500, detail="Model Forecasting belum siap di server.")
    
    try:
        metadata = model_registry.forecast_metadata
        
        # Mengambil parameter StandardScaler (Mendukung penamaan versi keras 2 maupun config tfjs)
        mean = np.array(metadata.get("scaler_mean") or metadata.get("mean"), dtype=np.float32)
        scale = np.array(metadata.get("scaler_scale") or metadata.get("std"), dtype=np.float32)
        y_scale = float(metadata.get("y_scale", 1000000.0))
        
        # Susun array 2D dengan urutan yang BENAR sesuai FEATURE_COLS saat training
        feature_values = [
            req.lag1_total_expense, req.lag2_total_expense, req.lag3_total_expense,
            req.roll3_mean_expense, req.roll6_mean_expense, req.roll3_std_expense,
            req.lag1_monthly_income, req.lag1_savings_rate, req.lag1_expense_growth,
            req.bulan_sin, req.bulan_cos, req.persona_id
        ]
        
        x = np.array([feature_values], dtype=np.float32)
        
        # 1. Transformasi data menggunakan parameter StandardScaler asli
        x_scaled = (x - mean) / scale
        
        # 2. Prediksi menggunakan model biner .keras (Compile=False)
        pred_scaled = model_registry.forecast_model.predict(x_scaled, verbose=0).ravel()[0]
        
        # 3. Kembalikan skala dari Juta ke Rupiah penuh (Inverse Target Scale)
        predicted_rupiah = float(pred_scaled * y_scale)
        
        return {
            "success": True,
            "predicted_total_expense": round(predicted_rupiah, 2),
            "currency": "IDR",
            "model_metrics": metadata.get("metrics", {})
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gagal melakukan prediksi Forecasting: {str(e)}")