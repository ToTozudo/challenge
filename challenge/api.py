import fastapi
import pandas as pd

from challenge.model import DelayModel

app = fastapi.FastAPI()

@app.get("/health", status_code=200)
async def get_health() -> dict:
    return {
        "status": "OK"
    }

@app.post("/predict", status_code=200)
async def post_predict() -> dict:
    data = pd.read_csv('../data/data.csv')
    model = DelayModel()
    features, _ = model.preprocess(
        data)  # Asegúrate de que el preprocesamiento solo devuelva features si no necesita target
    prediction = model.predict(features)
    return {"prediction": prediction}