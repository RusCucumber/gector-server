import json, traceback
from typing import Tuple
from flask import Flask, jsonify, request

from utils.helpers import normalize
from gector.gec_model import GecBERTModel

app = Flask(__name__)
app.config.from_pyfile("resources/config.cfg")

# GECToR 呼び出し処理
def load_gector(
    max_len=None, min_len=None, lowercase_tokens=None, iterations=None,
    special_tokens_fix=None, min_error_probability=None, **kwargs
) -> GecBERTModel:
    with open(app.config["DEFAULT_GECTOR_PARAM_PATH"]) as f:
        default_params = json.load(f)
        params = default_params["model_params"]

    if isinstance(max_len, int):
        params["max_len"] = max_len
    if isinstance(min_len, int):
        params["min_len"] = min_len
    if isinstance(lowercase_tokens, bool):
        params["lowercase_tokens"] = lowercase_tokens
    if isinstance(iterations, int):
        params["iterations"] = iterations
    if isinstance(special_tokens_fix, bool):
        params["special_tokens_fix"] = special_tokens_fix
    if isinstance(min_error_probability, float):
        params["min_error_probability"] = min_error_probability

    model = GecBERTModel(**params)

    return model    

def check_params(params: dict) -> Tuple[list, int, bool, dict]:
    if "input_text" in params:
        input_text = params["input_text"]
    else:
        raise IndexError()
    
    with open(app.config["DEFAULT_GECTOR_PARAM_PATH"]) as f:
        default_params = json.load(f)
        batch_size = default_params["batch_size"]
        to_normalize = default_params["to_normalize"]

    if "batch_size" in params:
        batch_size = params["batch_size"]
    if "to_normalize" in params:
        to_normalize = params["to_normalize"]

    if "model_params" in params:
        model_params = params["model_params"]
    else:
        model_params = {}

    return input_text, batch_size, to_normalize, model_params

def predict(
    model: GecBERTModel, input_text: list, batch_size: int, to_normalize: bool
) -> Tuple[list, dict]:
    predictions = []
    history = []

    cnt_corrections = 0
    batch = []
    for sent in input_text:
        batch.append(sent.split())
        if len(batch) == batch_size:
            preds, cnt, hist = model.handle_batch(batch)
            predictions.extend(preds)
            history.extend(hist)
            
            cnt_corrections += cnt
            batch = []
    if batch:
        preds, cnt, hist = model.handle_batch(batch)
        predictions.extend(preds)
        history.extend(hist)
        cnt_corrections += cnt

    result_lines = [" ".join(x) for x in predictions]
    if to_normalize:
        result_lines = [normalize(line) for line in result_lines]

    return result_lines, history, cnt_corrections

@app.route("/")
def indiex():
    return "index"

@app.route("/gector", methods=["GET", "POST"])
def gector():
    response = {"status": "NG"}

    if request.method == "POST":
        if request.headers["Content-type"] != "application/json":
            response["error"] = {
                "msg": "Unsupported Media Type"
            }
            return jsonify(response)
        
        params = request.get_json()

        try:
            input_text, batch_size, to_normalize, model_params = check_params(params)
        except IndexError:
            response["error"] = {
                "msg": "No \"input_text\" Parameter"
            }
            return jsonify(response)
        except Exception as e:
            response["error"] = {
                "msg": f"Unexpected Error",
                "traceback": traceback.format_exc(e)
            }

        try:
            model = load_gector(**model_params)
            result_lines, history, cnt_corrections = predict(model, input_text, batch_size, to_normalize)
        except Exception as e:
            response["error"] = {
                "msg": f"Unexpected Error",
                "traceback": traceback.format_exc(e)
            }
        else:
            response["status"] = "OK"
            response["result"] = {
                "result": result_lines,
                "n_correction": cnt_corrections,
                "history": history
            }

        return jsonify(response)
    else:
        return jsonify(response)

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0")