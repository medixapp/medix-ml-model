from flask import Flask, request, jsonify
import numpy as np
import os
from transformers import AutoTokenizer, TFAutoModelForQuestionAnswering
from inference_helper import *

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
    
    try :
        # Get the request
        question = request.form.get('question')
        if not question :
            return jsonify({'Error':'Tidak dapat membaca pertanyaan'}), 400
        
        question = question.lower()
        if 'gerd' in question :
            with open('GERD.txt', 'r') as file :
                context = file.read()
        elif 'tukak lambung' in question :
            with open('Tukak Lambung.txt', 'r') as file :
                context = file.read()
        elif 'gastroparesis' in question :
            with open('Gastroparesis.txt', 'r') as file :
                context = file.read()
        elif 'maag' in question :
            with open('Maag.txt', 'r') as file :
                context = file.read()
        elif 'dispepsia' in question :
            with open('Dispepsia.txt', 'r') as file :
                context = file.read()
        elif 'gastritis' in question :
            with open('Gastritis.txt', 'r') as file :
                context = file.read()
        elif 'flu perut' in question or 'muntaber' in question or 'gastroenteritis' in question :
            with open('Gastroenteritis.txt', 'r') as file :
                context = file.read()
        else :
            response = "Maaf, saya tidak bisa menjawab pertanyaan yang diberikan. Coba sebutkan penyakit pada pertanyaannya."
    
        # Inference
        inputs = tokenizer(question, context, return_tensors="np")
        if len(inputs['input_ids'][0]) > 512 :
            response = predict_long_context(question, context, model, tokenizer)
            
        else :
            outputs = model(inputs)
            
            # Get the maximum probabilities for start and end positions.
            start_position = np.argmax(outputs.start_logits[0])
            end_position = np.argmax(outputs.end_logits[0])
            
            # Get the answer from the context
            response_ids = inputs['input_ids'][0, start_position : end_position + 1]
            response = tokenizer.decode(response_ids)
        
        response = response[0].upper() + response[1:] + "."

        # Return the answer
        return jsonify({'answer': response})
    except Exception as e :
        return jsonify({'answer': str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port="8000")