from flask import Flask, request, jsonify
from transformers import PegasusForConditionalGeneration, PegasusTokenizer
import torch

app = Flask(__name__)

model_name = "google/pegasus-xsum"
tokenizer = PegasusTokenizer.from_pretrained(model_name)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = PegasusForConditionalGeneration.from_pretrained(model_name).to(device)

@app.route('/text-summarization', methods=["POST"])
def summarize():
    if request.method == "POST":
        input_text = request.data.decode("utf-8")  # Get the plain text input

        min_length = request.headers.get("Min-Length")
        max_length = request.headers.get("Max-Length")

        if not min_length or not max_length:
            return jsonify({"error": "Please provide Min-Length and Max-Length headers"}), 400

        min_length = int(min_length)
        max_length = int(max_length)

        if input_text:
            input_text = "summarize: " + input_text

            tokenized_text = tokenizer.encode(input_text, return_tensors='pt', max_length=512).to(device)
            summary_ = model.generate(tokenized_text, min_length=min_length, max_length=max_length)
            summary = tokenizer.decode(summary_[0], skip_special_tokens=True)

            return jsonify({"summary": summary})

        return jsonify({"error": "Missing input text"}), 400

if __name__ == '__main__':
    app.run()
