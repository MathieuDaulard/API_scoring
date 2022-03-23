from scoring_code import Scoring_model
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

#create the application
app = FastAPI(
    title = "Solvabilite Classifier API",
    version = 1.0,
    description = "Simple API to make predict of client solvancy."
)

#creating the classifier
classifier = Scoring_model("LightGBMModel.joblib")


@app.post("/")
async def get_prediction(id_client:str):
    id_client = int(id_client)
    if id_client not in classifier.get_id():
        raise HTTPException(status_code = 446, detail = str(id_client) + " non trouvé")
    result_classification = classifier.make_prediction(id_client)
    return result_classification