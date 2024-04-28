import json
from flask import Flask, request, jsonify
from tensorflow import keras 
from keras.models import load_model 
from keras.preprocessing.text import Tokenizer 
from keras.preprocessing.sequence import pad_sequences 

app = Flask(__name__)

TF_ENABLE_ONEDNN_OPTS=0

# Load pre-trained model and tokenizer
model = load_model('Conversation.csv')
tokenizer = Tokenizer()
tokenizer_json = 'tokenizer.json'
with open(tokenizer_json, 'r') as f:
    tokenizer_config = json.load(f)
tokenizer = Tokenizer.from_config(tokenizer_config)
# tokenizer.fit_on_texts(['How are you?', 'What are you doing?', 'What social situations do you find challenging or want to work on?']) 

@app.route('/predict', methods=['POST'])
def predict():
    input_text = request.json['input_text']
    input_sequence = tokenizer.texts_to_sequences([input_text])
    input_sequence = pad_sequences(input_sequence, maxlen=5, padding='post')  
    predicted_response = tokenizer.sequences_to_texts([response_sequence.argmax(axis=1)])[0]
    return jsonify({'response': predicted_response})

if __name__ == '__main__':
    app.run(debug=True)
