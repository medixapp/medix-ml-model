from flask import Flask, request, jsonify
import numpy as np
import os
from transformers import AutoTokenizer, TFAutoModelForQuestionAnswering

# Predefined Context
with open('Context.txt', 'r') as file :
    context = file.read()

# Run the download model first
# Load the Model and Tokenizer from the local storage.
model_checkpoint = os.path.join(os.getcwd(), 'model')
tokenizer_checkpoint = os.path.join(os.getcwd(), 'tokenizer')
model = TFAutoModelForQuestionAnswering.from_pretrained(model_checkpoint)
tokenizer = AutoTokenizer.from_pretrained(tokenizer_checkpoint)

# App initialization
app  = Flask(__name__)

# Endpoint
@app.route('/answer', methods=['POST'])
def index():
    
    # Get the request
    question = request.json['question']
    
    # Inference
    inputs = tokenizer(question, context, return_tensors="np")
    outputs = model(inputs)
    
    # Get the maximum probabilities for start and end positions.
    start_position = np.argmax(outputs.start_logits[0])
    end_position = np.argmax(outputs.end_logits[0])
    
    # Get the answer from the context
    response_ids = inputs['input_ids'][0, start_position : end_position + 1]
    response = tokenizer.decode(response_ids)

    # Return the answer
    return jsonify({'answer': response})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port="8000")