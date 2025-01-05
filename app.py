from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from chatbot import greeting, response, ADDITIONAL_RESPONSES
import nltk

app = Flask(__name__)
CORS(app)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/chat')
def chat():
    return render_template('chat.html')

@app.route('/get_response', methods=['POST'])
def get_bot_response():
    try:
        data = request.get_json()
        user_message = data.get('message', '').lower()
        
        # Check if it's a goodbye message
        if user_message == 'selamat tinggal':
            return jsonify({'response': "selamat tinggal! kembali lagi jika kamu memiliki pertanyaan tentang hipertensi."})
        
        # Check if it's a thank you message
        if user_message in ('terima kasih'):
            return jsonify({'response': "sama-sama"})
        
        # Check for greetings
        greeting_response = greeting(user_message)
        if greeting_response is not None:
            return jsonify({'response': greeting_response})
            
        # Get chatbot response
        bot_response = response(user_message)
        return jsonify({'response': bot_response})
        
    except Exception as e:
        print(f"Error: {str(e)}")  # For debugging
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    nltk.download('popular', quiet=True)
    nltk.download('punkt')
    nltk.download('wordnet')
    app.run(debug=True, port=5000)