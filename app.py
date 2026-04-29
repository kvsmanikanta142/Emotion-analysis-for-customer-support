from flask import Flask, render_template, request
import speech_recognition as sr
from transformers import pipeline

app = Flask(__name__)

recognizer = sr.Recognizer()
sentiment_model = pipeline("sentiment-analysis")

def speech_to_text():
    with sr.Microphone() as source:
        print("Listening...")
        audio = recognizer.listen(source)
        try:
            text = recognizer.recognize_google(audio)
            return text
        except:
            return "Could not understand audio"

@app.route("/", methods=["GET", "POST"])
def index():
    text = ""
    sentiment = ""
    score = ""

    if request.method == "POST":
        text = speech_to_text()
        result = sentiment_model(text)[0]
        sentiment = result["label"]
        score = round(result["score"], 3)

    return render_template("index.html",
                           text=text,
                           sentiment=sentiment,
                           score=score)

if __name__ == "__main__":
    app.run(debug=True,use_reloader=False)
