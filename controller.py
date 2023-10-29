from fastapi import APIRouter, HTTPException, status
from models import Prediction_Input
from models import Prediction_Output

from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle


MAX_LENGTH = 100
PADDING_TYPE = 'post'
TRUNC_TYPE = 'post'
TOKENIZER_PATH = 'tokenizer.pkl'
MODEL_PATH = 'model.h5'

# Load the tokenizer from the file
tokenizer_filename = TOKENIZER_PATH
with open(tokenizer_filename, 'rb') as f:
    tokenizer = pickle.load(f)

# Load Tensorflow model
model = keras.models.load_model(MODEL_PATH)
print(model.summary())

router = APIRouter()

preds = []

@router.get('/ml')
def get_preds():
    return preds

@router.post('/ml', status_code=status.HTTP_201_CREATED, response_model=Prediction_Output)
def predict(pred_input: Prediction_Input):

    sequences = tokenizer.texts_to_sequences(pred_input.text_input)
    padded = pad_sequences(sequences, maxlen=MAX_LENGTH, padding=PADDING_TYPE, truncating=TRUNC_TYPE)
    prediction_f = model.predict(padded)

    prediction_dict = {"id": str(pred_input.id), "text_input":str(pred_input.text_input) , "pred": float(prediction_f[0,0])}
    preds.append(prediction_dict)

    return prediction_dict

@router.put('/ml/{task_id}', status_code=203)
def update_predict(id : int , pred_input: Prediction_Input):

    for index, item in enumerate(preds):
        if item.id == id:
            preds[index] = pred_input
            return pred_input
    else:
        HTTPException(status_code=404,detail="Task not found")

@router.delete('/ml/{task_id}')
def delete_predict(id:int):
    for index, item in enumerate(preds):
        if item.id == id:
            preds.pop(index)
            return {"message": "Task deleted"}
    else:
        HTTPException(status_code=404,detail="Task not found")
