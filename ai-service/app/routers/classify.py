from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import numpy as np
import re

from app.core.config import model_registry

router = APIRouter(prefix="/predict", tags=["NLP Classification"])

class ClassifyRequest(BaseModel):
    desc: str

@router.post("/classify")
def predict_category(req: ClassifyRequest):
    if model_registry.classify_model is None or model_registry.classify_metadata is None:
        raise HTTPException(status_code=500, detail="Model Classify belum siap.")
    
    try:
        # 1. AMBIL KAMUS VOCAB
        vocab = model_registry.classify_metadata.get("vocab")
        if not vocab:
            raise ValueError("Kamus 'vocab' tidak ditemukan di metadata.json!")
        
        # 2. AMBIL DIMENSI INPUT LANGSUNG DARI MODEL (Anti-Gagal)
        # Model akan memberitahu kita ukuran pastinya (dalam hal ini 22454)
        expected_dim = model_registry.classify_model.input_shape[-1] 
        
        # 3. CUSTOM VECTORIZER
        # Buat array 2D dengan ukuran presisi sesuai permintaan model
        input_vector = np.zeros((1, expected_dim), dtype=np.float32)
        
        # Bersihkan teks
        clean_text = req.desc.lower()
        words = re.findall(r'\b\w+\b', clean_text)
        
        # Cocokkan kata dengan kamus
        for word in words:
            if word in vocab:
                idx = vocab[word]
                # Pastikan index tidak melebihi batas kolom (Safety check)
                if idx < expected_dim:
                    input_vector[0, idx] = 1.0
                
        # 4. PREDIKSI
        predictions = model_registry.classify_model.predict(input_vector, verbose=0)
        predicted_index = int(np.argmax(predictions[0]))
        confidence_score = float(np.max(predictions[0]))
        
        # 5. AMBIL NAMA KELAS
        class_names = model_registry.classify_metadata.get("class_names", [])
        if predicted_index < len(class_names):
            predicted_category = class_names[predicted_index]
        else:
            predicted_category = f"Category_{predicted_index}"

        return {
            "success": True,
            "description": req.desc,
            "predicted_category": predicted_category,
            "confidence": confidence_score
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gagal melakukan prediksi NLP: {str(e)}")