import json, traceback, argparse
from typing import Tuple
from flask import Flask, jsonify, request

from utils.helpers import normalize
from gector.gec_model import GecBERTModel

app = Flask(__name__)
app.config.from_pyfile("resources/config.cfg")

# GECToR 呼び出し処理
def load_gector(params_path: str) -> GecBERTModel:
    with open(params_path) as f: # TODO: fix hard programming
        default_params = json.load(f)
        params = default_params["model_params"]
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

    return input_text, batch_size, to_normalize

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

    model = load_gector("resources/default_params.json")

    if request.method == "POST":
        if request.headers["Content-type"] != "application/json":
            response["error"] = {
                "msg": "Unsupported Media Type"
            }
            return jsonify(response)
        
        params = request.get_json()

        try:
            input_text, batch_size, to_normalize = check_params(params)
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
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--params_path", 
        help="Path to the model parameter json file.", 
        required=False, 
        default="resources/default_params.json",
        type=str
    )
    parser.add_argument(
        "--debug", 
        help="If true, run flask as a debug mode.",
        required=False,
        default=True,
        type=bool
    )
    parser.add_argument(
        "--host",
        help="Host of flask app",
        required=False,
        default="0.0.0.0",
        type=str
    )
    parser.add_argument(
        "--port",
        help="Port number of flask app",
        required=False,
        default=5000,
        type=int
    )

    args = parser.parse_args()

    # TODO: uwsgi での動作確認
    model = load_gector(args.params_path)
    app.run(debug=args.debug, host=args.host, port=args.port)