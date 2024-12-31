from flask import Flask, request, render_template
import joblib
from transformers import pipeline
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from collections import Counter
import matplotlib
matplotlib.use('Agg')  # For servers without a display
import matplotlib.pyplot as plt
import io
import base64
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from wordcloud import WordCloud

# Initialize Flask app
app = Flask(__name__)

# Load the Random Forest model, vectorizer, and scaler
rf_model = joblib.load('models/random_forest_model.joblib')
vectorizer = joblib.load('models/tfidf_vectorizer.joblib')
scaler = joblib.load('models/scaler.joblib')

# Load BERT model (DistilBERT) for sentiment analysis
bert_classifier = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

# Store the inputs from users
user_inputs = []  # raw text storage
sentiment_counts = {'positive': 0, 'negative': 0}

# For advanced visualization (trend)
sentiment_history = []

# For actionable insights: store each submission’s tokens + final sentiment
user_data = []

# ------------------------------------------------------------------------------
# Utility functions
# ------------------------------------------------------------------------------

def tokenize_text(text):
    """
    Returns a list of normalized tokens (lowercased, lemmatized, stopwords removed).
    """
    text = re.sub(r'http\S+|www\S+', '', text)  # Remove URLs
    text = re.sub(r'@\w+|#\w+', '', text)       # Remove mentions and hashtags
    text = re.sub(r'\W+|\d+', ' ', text)        # Remove special chars & digits
    text = text.strip()
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    tokens = [
        lemmatizer.lemmatize(word.lower())
        for word in word_tokenize(text)
        if word.lower() not in stop_words
    ]
    return tokens

def preprocess_text_rf(text, vectorizer, scaler):
    """
    Preprocess text for Random Forest:
      1) tokenize & clean,
      2) transform with vectorizer,
      3) scale the vector.
    Returns both scaled_vector and the token list.
    """
    tokens = tokenize_text(text)
    clean_text = ' '.join(tokens)
    tfidf_vector = vectorizer.transform([clean_text])
    scaled_vector = scaler.transform(tfidf_vector)
    return scaled_vector, tokens

def get_bert_prediction(text):
    """ Get label + score from BERT sentiment analysis pipeline. """
    result = bert_classifier(text)
    # Format: [{'label': 'POSITIVE'/'NEGATIVE', 'score': 0.999...}]
    return result[0]['label'], result[0]['score']

def get_rf_prediction(text):
    """ Get label from Random Forest. (Optionally can get probabilities if needed.) """
    rf_input, _ = preprocess_text_rf(text, vectorizer, scaler)
    return rf_model.predict(rf_input)[0]

def combine_predictions(bert_label, rf_label):
    """
    Example strategy: prefer BERT’s label. 
    Replace with your own ensemble logic if desired.
    """
    return bert_label

# ------------------------------------------------------------------------------
# Trending Topics
# ------------------------------------------------------------------------------

def extract_trending_topics(text_data):
    """
    Aggregates tokens from all user inputs to find the top 10 frequent words,
    excluding trivial words like "good" or "bad".
    """
    exclude_words = {"good", "bad"}  # Example trivial words
    all_tokens = []

    for text in text_data:
        tokens = tokenize_text(text)
        # Filter out trivial words
        filtered_tokens = [t for t in tokens if t not in exclude_words]
        all_tokens.extend(filtered_tokens)

    # Count frequencies and return top 10
    word_counts = Counter(all_tokens)
    return word_counts.most_common(10)

# ------------------------------------------------------------------------------
# Actionable Insights
# ------------------------------------------------------------------------------

def get_actionable_insights(user_data, threshold=3, ratio_threshold=0.7):
    """
    Identify topics that are mentioned frequently with negative sentiment.
    Example rule:
    - A word is "actionable" if it appears >= `threshold` times in negative submissions
    - AND at least `ratio_threshold` fraction of the time, it's negative.
    """
    negative_counts = Counter()
    positive_counts = Counter()

    # Tally how often each token appears in negative vs. positive text
    for entry in user_data:
        tokens = entry["tokens"]
        if entry["final_prediction"].upper() == "NEGATIVE":
            negative_counts.update(tokens)
        else:
            positive_counts.update(tokens)

    insights = []
    for word, neg_count in negative_counts.items():
        pos_count = positive_counts[word]
        total_count = neg_count + pos_count
        neg_ratio = neg_count / total_count if total_count > 0 else 0

        if neg_count >= threshold and neg_ratio >= ratio_threshold:
            message = (
                f"'{word}' is frequently mentioned negatively "
                f"({neg_count} times, {int(neg_ratio*100)}% negative). "
                "Investigate possible issues or improvements."
            )
            insights.append(message)

    return insights

# ------------------------------------------------------------------------------
# Graph generation
# ------------------------------------------------------------------------------

def generate_sentiment_graph():
    """Generate a bar chart showing positive vs. negative sentiment counts."""
    fig, ax = plt.subplots()
    ax.bar(sentiment_counts.keys(), sentiment_counts.values(), color=['green', 'red'])
    ax.set_xlabel('Sentiment')
    ax.set_ylabel('Count')
    ax.set_title('Sentiment Distribution')

    # Convert plot to PNG image
    img = io.BytesIO()
    FigureCanvas(fig).print_png(img)
    img.seek(0)
    graph_url = base64.b64encode(img.getvalue()).decode('utf8')
    plt.close(fig)
    return graph_url

def generate_sentiment_trend_graph():
    """
    Example line chart showing "positive probability" over time, 
    using BERT’s confidence score stored in `sentiment_history`.
    """
    times = [i+1 for i in range(len(sentiment_history))]
    scores = []
    for data in sentiment_history:
        # If label is POSITIVE, use the direct score; if NEGATIVE, invert it
        if data["bert_label"].upper() == "POSITIVE":
            scores.append(data["bert_score"])
        else:
            scores.append(1 - data["bert_score"])

    fig, ax = plt.subplots()
    ax.plot(times, scores, marker='o', linestyle='-', color='blue')
    ax.set_xlabel('Submission #')
    ax.set_ylabel('Positive Probability')
    ax.set_title('Sentiment Trend Over Time (BERT)')

    # Convert plot to PNG
    img = io.BytesIO()
    FigureCanvas(fig).print_png(img)
    img.seek(0)
    graph_url = base64.b64encode(img.getvalue()).decode('utf8')
    plt.close(fig)
    return graph_url

# ------------------------------------------------------------------------------
# Word Cloud Generation
# ------------------------------------------------------------------------------

def generate_word_cloud(user_inputs):
    """
    Generates a word cloud image (base64-encoded) from the tokens of all user inputs.
    We reuse the same tokenization approach, but you can also directly use the
    raw text if you prefer.
    """
    all_tokens = []

    # Use the same tokenization for consistency
    for text in user_inputs:
        # Tokenize and remove "good"/"bad" as done in trending topics
        tokens = tokenize_text(text)
        tokens = [t for t in tokens if t not in {"good", "bad"}]
        all_tokens.extend(tokens)

    if not all_tokens:
        return None  # If no tokens, return None to handle gracefully

    # Create a single string for the word cloud
    text_for_cloud = ' '.join(all_tokens)

    # Generate the word cloud
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text_for_cloud)

    # Convert to PNG image in memory
    fig = plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')

    img = io.BytesIO()
    FigureCanvas(fig).print_png(img)
    img.seek(0)
    wordcloud_url = base64.b64encode(img.getvalue()).decode('utf8')
    plt.close(fig)
    return wordcloud_url

# ------------------------------------------------------------------------------
# Routes
# ------------------------------------------------------------------------------

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    text = request.form['text']

    # Get predictions
    bert_label, bert_score = get_bert_prediction(text)
    rf_label = get_rf_prediction(text)

    final_prediction = combine_predictions(bert_label, rf_label)

    # Update sentiment counts
    if final_prediction.upper() == 'POSITIVE':
        sentiment_counts['positive'] += 1
    else:
        sentiment_counts['negative'] += 1

    # For advanced visualization (trend)
    sentiment_history.append({
        "bert_label": bert_label,
        "bert_score": bert_score,
        "rf_label": rf_label
    })

    # Store the user input (raw text)
    user_inputs.append(text)

    # Also store tokens + final sentiment for actionable insights
    _, tokens = preprocess_text_rf(text, vectorizer, scaler)
    user_data.append({
        "text": text,
        "tokens": tokens,
        "final_prediction": final_prediction
    })

    # Generate outputs
    trending_topics = extract_trending_topics(user_inputs)
    sentiment_graph = generate_sentiment_graph()
    sentiment_trend_graph = generate_sentiment_trend_graph()
    actionable_insights = get_actionable_insights(user_data)

    # Generate word cloud
    word_cloud_image = generate_word_cloud(user_inputs)

    return render_template(
        'index.html',
        prediction=final_prediction,
        text=text,
        trending_topics=trending_topics,
        sentiment_graph=sentiment_graph,
        sentiment_trend_graph=sentiment_trend_graph,
        actionable_insights=actionable_insights,
        word_cloud_image=word_cloud_image
    )

@app.route('/about')
def about():
    """
    Render the About Us page.
    """
    return render_template('about.html')

@app.route('/contact')
def contact():
    """
    Render the Contact page.
    """
    return render_template('contact.html')

if __name__ == '__main__':
    app.run(debug=True)
