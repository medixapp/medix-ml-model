from flask import Flask, request, jsonify
import numpy as np
import os
from transformers import AutoTokenizer, TFAutoModelForQuestionAnswering

# Predefined Context
context = """GERD adalah penyakit yang terjadi ketika asam lambung naik ke kerongkongan. 
Gejala GERD bisa berupa sensasi perih di dada dan perut, rasa pahit di mulut, nyeri ulu 
hati, mual, dan sulit menelan. Otot bagian bawah kerongkongan atau lower esophageal sphincter 
(LES) normalnya akan terbuka saat menelan, kemudian menutup setelah makanan turun ke lambung. 
Namun, otot ini dapat melemah, misalnya akibat penyakit atau pola hidup yang kurang sehat. 
Otot LES yang tidak bisa menutup sepenuhnya dapat menyebabkan asam lambung naik ke kerongkongan. 
Kondisi inilah yang menimbulkan gejala GERD (gastroesophageal reflux disease) atau penyakit asam 
lambung. Penyebab GERD adalah melemahnya otot LES sehingga tidak mampu menahan isi lambung atau 
asam lambung agar tidak naik ke kerongkongan. Jika terjadi terus-menerus, kondisi ini dapat 
mengiritasi lapisan kerongkongan hingga menyebabkan peradangan. Ada beberapa hal yang bisa menjadi 
penyebab GERD, yaitu obesitas, kehamilan, usia lanjut, kebiasaan sering berbaring atau tidur setelah 
makan, gastroparesis, yaitu melemahnya otot dinding lambung sehingga pengosongan lambung melambat, 
gangguan jaringan ikat, misalnya skleroderma atau lupus, penyakit bawaan lahir, seperti hernia hiatus 
dan atresia esofagus, pernah menjalani operasi di area dada atau perut bagian atas sehingga melukai 
kerongkongan, dan efek samping obat-obatan tertentu, misalnya aspirin, ibuprofen, benzodiazepin, 
antidepresan, atau obat terapi hormon untuk menopause. Ada beberapa faktor yang dapat memperparah 
gejala GERD, seperti kebiasaan merokok atau sering terpapar asap rokok (perokok pasif), diet ekstrim 
atau telat makan saat puasa, sering makan dalam porsi besar atau makan pada tengah malam, mengonsumsi 
makanan yang asam, berlemak, atau berbumbu pedas, mengonsumsi minuman berkafein, beralkohol, atau 
bersoda, mengalami gangguan kecemasan atau stres yang tidak terkelola dengan baik (GERD anxiety)."""

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
    response = inputs['input_ids'][0, start_position : end_position + 1]

    # Return the answer
    return jsonify({'answer': response})

if __name__ == "__main__":
    app.run(debug=True)