class AIModelRegistry:
    def __init__(self):
        
        # Model 3: WhatIf Lab
        self.whatif_model = None
        self.whatif_metadata = None
        
        # Model 1: NLP Classification
        self.classify_model = None
        self.classify_metadata = None
        
        # Model 2: Forecasting 
        self.forecast_model = None
        self.forecast_metadata = None

# Single instance untuk digunakan di seluruh aplikasi
model_registry = AIModelRegistry()