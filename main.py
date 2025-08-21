from flask import Flask, request, jsonify
from flask_cors import CORS
import json
import pickle
import random
from tensorflow.keras.models import load_model
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
import sys
import os

app = Flask(__name__)
CORS(app)  # Enable CORS for React frontend

lemmatizer = WordNetLemmatizer()

# Global variables for model components
model = None
intents = None
words = None
classes = None


def initialize_chatbot():
    """Initialize the chatbot by loading all required files"""
    global model, intents, words, classes

    try:
        print("🤖 Initializing AI Chatbot...")

        # Download required NLTK data
        nltk.download('omw-1.4', quiet=True)
        nltk.download('punkt', quiet=True)

        # Load the trained model
        if os.path.exists('./trained_model.keras'):
            model = load_model('./trained_model.keras')
            print("Model loaded successfully!")
        else:
            print("Can't find: trained_model.keras")
            return False

        # Load intents JSON
        if os.path.exists('./intents.json'):
            with open('./intents.json', 'r') as file:
                intents = json.load(file)
            print("Intents loaded successfully!")
        else:
            print("Can't find: intents.json")
            return False

        # Load preprocessed words and classes
        if os.path.exists('./words.pkl') and os.path.exists('./classes.pkl'):
            with open('./words.pkl', 'rb') as file:
                words = pickle.load(file)
            with open('./classes.pkl', 'rb') as file:
                classes = pickle.load(file)
            print("Words and classes loaded successfully!")
        else:
            print("Can't find: words.pkl or classes.pkl")
            return False

        print("Chatbot initialized successfully!")
        return True

    except Exception as e:
        print(f"Error initializing chatbot: {e}")
        return False


def clean_up_sentence(sentence):
    """Tokenize and lemmatize the input sentence"""
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words


def bow(sentence, words):
    """Create bag of words array from the input sentence"""
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
    return np.array(bag)


def predict_class(sentence):
    """Predict the intent class for the input sentence"""
    p = bow(sentence, words)
    res = model.predict(np.array([p]), verbose=0)[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list


def get_response(ints):
    """Get a random response for the predicted intent"""
    if len(ints) == 0:
        return "I don't understand that. Could you try rephrasing?"

    tag = ints[0]['intent']
    list_of_intents = intents['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            return result
    return "I'm not sure how to respond to that."


def chatbot_response(msg):
    """Generate chatbot response for user message"""
    try:
        ints = predict_class(msg)
        res = get_response(ints)
        return res
    except Exception as e:
        return f"Sorry, I encountered an error: {str(e)}"


# API Routes
@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'intents_loaded': intents is not None,
        'words_loaded': words is not None,
        'classes_loaded': classes is not None
    })


@app.route('/api/chat', methods=['POST'])
def chat():
    """Handle chat API endpoint"""
    try:
        data = request.get_json()
        if not data or 'message' not in data:
            return jsonify({'error': 'No message provided'}), 400

        user_message = data['message'].strip()
        if not user_message:
            return jsonify({'response': 'Please enter a message!'})

        bot_response = chatbot_response(user_message)
        return jsonify({'response': bot_response})

    except Exception as e:
        return jsonify({'error': f'Server error: {str(e)}'}), 500


@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404


@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500


if __name__ == '__main__':
    if not initialize_chatbot():
        print("Failed to initialize chatbot. Please ensure all required files are present:")
        print("   - trained_model.keras")
        print("   - intents.json")
        print("   - words.pkl")
        print("   - classes.pkl")
        sys.exit(1)

    print("💬 Starting Flask API server...")
    print("🌐 API running on http://localhost:5000")
    print("🔗 React frontend should connect to http://localhost:5000/api")
    print("⏹️  Press Ctrl+C to stop the server")

    try:
        app.run(debug=True, host='0.0.0.0', port=5000)
    except KeyboardInterrupt:
        print("\nChatbot server stopped. Goodbye!")
    except Exception as e:
        print(f"Server error: {e}")
