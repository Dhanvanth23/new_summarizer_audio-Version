from flask import Flask, render_template, request, send_from_directory
from transformers import BartForConditionalGeneration, BartTokenizer
import requests
from bs4 import BeautifulSoup
from mtranslate import translate
from gtts import gTTS
import os

app = Flask(__name__)

# Directory to store audio files
AUDIO_DIR = "static/audio"
if not os.path.exists(AUDIO_DIR):
    os.makedirs(AUDIO_DIR)

# Load BART model and tokenizer
model_name = "facebook/bart-large-cnn"
model = BartForConditionalGeneration.from_pretrained(model_name)
tokenizer = BartTokenizer.from_pretrained(model_name)

def fetch_article_text(url):
    try:
        # Add headers to mimic a browser request
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        response = requests.get(url, headers=headers)
        response.raise_for_status()  # Raise an error for bad status codes
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Extract text from paragraphs
        paragraphs = soup.find_all('p')
        if not paragraphs:
            return None  # No paragraphs found
        
        article_text = ' '.join([p.get_text() for p in paragraphs])
        return article_text
    except Exception as e:
        print(f"Error fetching article: {e}")
        return None

def translate_text(text, src_lang, target_lang):
    try:
        if not text or not text.strip():
            print("Translation error: Input text is empty.")
            return None
        
        print(f"Translating text from {src_lang} to {target_lang}: {text}")
        translated = translate(text, to_language=target_lang, from_language=src_lang)
        
        if not translated:
            print("Translation error: No translation returned.")
            return None
        
        print(f"Translated text: {translated}")
        return translated
    except Exception as e:
        print(f"Translation error: {e}")
        return None

def summarize_text(text):
    inputs = tokenizer([text], max_length=1024, return_tensors="pt", truncation=True)
    summary_ids = model.generate(inputs["input_ids"], num_beams=4, max_length=150, early_stopping=True)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

def text_to_speech(text, language='ta'):
    try:
        # Generate audio file using gTTS
        tts = gTTS(text=text, lang=language, slow=False)
        audio_file = os.path.join(AUDIO_DIR, "summary.mp3")
        tts.save(audio_file)
        return audio_file
    except Exception as e:
        print(f"Text-to-speech error: {e}")
        return None

@app.route('/', methods=['GET', 'POST'])
def index():
    summary = None
    error = None
    audio_file = None
    if request.method == 'POST':
        input_type = request.form['input_type']
        language = request.form['language']
        if input_type == 'url':
            url = request.form['url']
            article_text = fetch_article_text(url)
            if article_text:
                if language == 'ta':
                    # Translate Tamil text to English
                    translated_text = translate_text(article_text, src_lang='ta', target_lang='en')
                    if translated_text:
                        # Summarize the English text
                        english_summary = summarize_text(translated_text)
                        # Translate the summary back to Tamil
                        summary = translate_text(english_summary, src_lang='en', target_lang='ta')
                        # Convert summary to speech
                        audio_file = text_to_speech(summary, language='ta')
                    else:
                        error = "Failed to translate Tamil text to English."
                else:
                    summary = summarize_text(article_text)
                    # Convert summary to speech
                    audio_file = text_to_speech(summary, language='en')
            else:
                error = "Failed to fetch article content from the URL. Please check the URL or try a different article."
        elif input_type == 'text':
            text = request.form['text']
            if text.strip():
                if language == 'ta':
                    # Translate Tamil text to English
                    translated_text = translate_text(text, src_lang='ta', target_lang='en')
                    if translated_text:
                        # Summarize the English text
                        english_summary = summarize_text(translated_text)
                        # Translate the summary back to Tamil
                        summary = translate_text(english_summary, src_lang='en', target_lang='ta')
                        # Convert summary to speech
                        audio_file = text_to_speech(summary, language='ta')
                    else:
                        error = "Failed to translate Tamil text to English."
                else:
                    summary = summarize_text(text)
                    # Convert summary to speech
                    audio_file = text_to_speech(summary, language='en')
            else:
                error = "Please provide some text to summarize."
    return render_template('index.html', summary=summary, error=error, audio_file=audio_file)

@app.route('/audio/<filename>')
def serve_audio(filename):
    return send_from_directory(AUDIO_DIR, filename)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)