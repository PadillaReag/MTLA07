from flask import Flask, render_template, request
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import joblib

app = Flask(__name__)

# Load dataset and train models
filename = r'C:\Users\JOEDEM VIGMAR\Desktop\MachineLearning-AT7\text_emotion_prediction.csv'
df = pd.read_csv(filename, encoding='ISO-8859-1')
df.dropna(subset=['Comment', 'Emotion'], inplace=True)

# Vectorization
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df['Comment'])
y = df['Emotion'].values

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

def train_models():
    classifiers = {
        "Naive Bayes": MultinomialNB(),
        "KNN": KNeighborsClassifier(),
        "Decision Tree": DecisionTreeClassifier(),
        "SVM": SVC(probability=True)
    }
    
    accuracy_results = {}
    for name, clf in classifiers.items():
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred) * 100  # Convert to percentage
        accuracy_results[name] = f"{accuracy:.1f}%"  # Format as a string with one decimal place
    
    # Save classifiers
    for name, clf in classifiers.items():
        joblib.dump(clf, f'{name}_model.pkl')
    
    joblib.dump(vectorizer, 'vectorizer.pkl')
    return accuracy_results, classifiers

accuracy_results, classifiers = train_models()

# Load model function
def load_model(model_name):
    model = joblib.load(f'{model_name}_model.pkl')
    vectorizer = joblib.load('vectorizer.pkl')
    return model, vectorizer

# Prediction function
def predict_emotion(text, model_name):
    model, vectorizer = load_model(model_name)
    text_vectorized = vectorizer.transform([text])
    prediction = model.predict(text_vectorized)
    return prediction[0]

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    input_text = None
    model_name = None
    if request.method == 'POST':
        input_text = request.form['text_input']
        model_name = request.form['classification_method']
        prediction = predict_emotion(input_text, model_name)

    return render_template('index.html', prediction=prediction, input_text=input_text, model_name=model_name, accuracy_results=accuracy_results)

if __name__ == '__main__':
    app.run(debug=True)