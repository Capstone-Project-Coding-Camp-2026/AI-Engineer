import json
import os
import tensorflow as tf
from fastapi import FastAPI
from contextlib import asynccontextmanager

from app.core.config import model_registry
from app.routers import whatif, classify, forecast  # Import router baru kita

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# =========================================================
# 🛠️ HARDCORE PATCH: Mencegah Bug Keras 3
# =========================================================
original_dense_init = tf.keras.layers.Dense.__init__
def patched_dense_init(self, *args, **kwargs):
    kwargs.pop('quantization_config', None)
    original_dense_init(self, *args, **kwargs)
tf.keras.layers.Dense.__init__ = patched_dense_init
# =========================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("[BOOTING] Memuat semua model biner AI ke memori server...")
    try:
        # --- LOAD MODEL 1 (NLP CLASSIFY) ---
        nlp_keras = os.path.join(BASE_DIR, "models", "classify", "model_classify.keras")
        nlp_json = os.path.join(BASE_DIR, "models", "classify", "metadata.json")
        
        model_registry.classify_model = tf.keras.models.load_model(nlp_keras, compile=False)
        with open(nlp_json, "r") as f:
            model_registry.classify_metadata = json.load(f)

        print("[READY] Model 1 (NLP Classification) berhasil dimuat!")

        # --- LOAD MODEL 3 (WHAT-IF LAB) ---
        wi_keras = os.path.join(BASE_DIR, "models", "whatif", "model_whatif.keras")
        wi_json = os.path.join(BASE_DIR, "models", "whatif", "metadata.json")
        
        model_registry.whatif_model = tf.keras.models.load_model(wi_keras, compile=False)
        with open(wi_json, "r") as f:
            model_registry.whatif_metadata = json.load(f)
        print("[READY] Model 3 (What-If Lab) berhasil dimuat!")

       # --- LOAD MODEL 2 ---
        fc_keras = os.path.join(BASE_DIR, "models", "forecast", "model_forecast.keras")
        fc_json = os.path.join(BASE_DIR, "models", "forecast", "metadata.json")
        model_registry.forecast_model = tf.keras.models.load_model(fc_keras, compile=False)
        with open(fc_json, "r") as f:
            model_registry.forecast_metadata = json.load(f)
        print("[READY] Model 2 (Forecasting Engine) Berhasil Dimuat.")
        
    except Exception as e:
        print(f"[CRITICAL ERROR] Gagal memuat model biner AI: {e}")
    
    yield
    print("[SHUTDOWN] Membersihkan resource memori server...")

app = FastAPI(title="FinTime Dedicated AI Engine", lifespan=lifespan)

# Daftarkan Router
app.include_router(whatif.router)
app.include_router(classify.router)
app.include_router(forecast.router)

@app.get("/health")
def health():
    return {"status": "healthy", "engine": "FastAPI Monolith Inference Server"}