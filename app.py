import re
import joblib
import nltk
from flask import Flask, request, jsonify
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Make sure Flask knows it's an app
app = Flask(__name__)

# ==========================================
# 1. LOAD THE TRAINED ARTIFACTS
# ==========================================
print("Loading Model and Vectorizer into memory...")
model = joblib.load("sentiment_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")
print("API is ready to serve predictions!")

# ==========================================
# 2. MATCH THE CLEANING PIPELINE EXACTLY
# ==========================================
def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    
    words = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    stop_words.discard('not')
    stop_words.discard('no')
    
    filtered_words = [w for w in words if w not in stop_words]
    return " ".join(filtered_words)

# ==========================================
# 3. DEFINE THE API ENDPOINT
# ==========================================
@app.route('/predict', methods=['POST'])
def predict():
    # Grab incoming JSON data from the request
    data = request.get_json()
    
    # Error checking: did they actually send text?
    if not data or 'text' not in data:
        return jsonify({
            'error': 'Bad Request',
            'message': 'Please provide a JSON payload with a "text" key.'
        }), 400
        
    raw_text = data['text']
    
    # 1. Run the exact same preprocessing
    cleaned = clean_text(raw_text)
    
    # 2. Vectorize using our loaded translator
    vectorized = vectorizer.transform([cleaned])
    
    # 3. Get prediction and confidence scores
    prediction = int(model.predict(vectorized)[0]) # Convert numpy int to standard Python int
    probabilities = model.predict_proba(vectorized)[0]
    confidence = float(probabilities.max()) # Convert float64 to normal float
    
    # Map back to human-readable string
    label_map = {0: "Negative", 1: "Positive"}
    
    # 4. Respond with structured JSON
    return jsonify({
        'status': 'success',
        'raw_text': raw_text,
        'prediction': label_map[prediction],
        'confidence': round(confidence, 4)
    })

# ==========================================
# 4. RUN THE LOCAL SERVER
# ==========================================
if __name__ == '__main__':
    # Runs the local development server at http://127.0.0.1:5000
    app.run(debug=True)