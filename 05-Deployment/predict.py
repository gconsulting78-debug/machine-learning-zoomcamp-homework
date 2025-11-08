import pickle
from fastapi import FastAPI
import uvicorn

app = FastAPI(title="lead_conversion")

with open('pipeline_v1.bin', 'rb') as f_in:
   pipeline1 = pickle.load(f_in)


def predict_single(customer):
    result = pipeline1.predict_proba(customer)[0, 1]
    return float(result)

from typing import Dict, Any

@app.post("/predict")
def predict(customer: Dict[str, Any]):
    prob = predict_single(customer)


    return {
        "conversion_probability": prob,
        "converted": bool(prob >= 0.5)
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9696)