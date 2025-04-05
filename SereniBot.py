import streamlit as st
import ollama
import base64
import os
import spacy
import torch
import speech_recognition as sr
from gtts import gTTS
from io import BytesIO
from transformers import pipeline
from flask import Flask, request, jsonify
from threading import Thread

# --- Backend API Setup (Flask) ---
flask_app = Flask(__name__)

@flask_app.route('/api/chatbot/message', methods=['POST'])
def chatbot_api_endpoint():
    """
    API endpoint to receive user messages and return chatbot responses.
    Designed to be called by external applications.
    """
    try:
        data = request.get_json()
        if not data or 'message' not in data:
            return jsonify({'error': 'Missing "message" in request body'}), 400

        user_message = data['message']

        # Generate response using the same logic as the Streamlit app
        conversation_history = [{"role": "user", "content": user_message}]
        response = ollama.chat(model="llama3.2", messages=conversation_history)
        ai_response = response["message"]["content"]

        return jsonify({'response': ai_response}), 200

    except Exception as e:
        return jsonify({'error': f'An error occurred in API: {str(e)}'}), 500

def run_flask_app():
    flask_app.run(debug=False, host='0.0.0.0', port=5001)

# Start the Flask API in a separate thread
api_thread = Thread(target=run_flask_app)
api_thread.daemon = True  # Allows the main script to exit even if the thread is running
api_thread.start()

# Load NLP model for mental health analysis
nlp = spacy.load("en_core_web_sm")
sentiment_analyzer = pipeline("sentiment-analysis")

# Initialize session states
if "conversation_history" not in st.session_state:
    st.session_state["conversation_history"] = [
        {"role": "assistant", "content": "Hello"} # Initial AI message
    ]
if "long_term_memory" not in st.session_state:
    st.session_state["long_term_memory"] = ""

# Function to analyze sentiment using deep NLP
def analyze_sentiment(text):
    result = sentiment_analyzer(text)[0]
    return result["label"], result["score"]

# Function for mental health analysis
def detect_mental_health_signs(text):
    label, score = analyze_sentiment(text)
    if label == "NEGATIVE" and score > 0.85:
        return "⚠️ Your response suggests you may be feeling distressed. Consider seeking professional support."
    return "✅ Your response appears stable. Keep focusing on your well-being!"

# Function to generate AI response
def generate_response(user_input):
    st.session_state["conversation_history"].append({"role": "user", "content": user_input})
    st.session_state["long_term_memory"] += f"User: {user_input}\n"
    response = ollama.chat(model="llama3.2", messages=st.session_state["conversation_history"])
    ai_response = response["message"]["content"]
    st.session_state["conversation_history"].append({"role": "assistant", "content": ai_response})
    st.session_state["long_term_memory"] += f"AI: {ai_response}\n"
    return ai_response

# Function for voice recognition (remains the same)
def recognize_speech():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        st.write("🎤 Listening...")
        audio = recognizer.listen(source)
    try:
        text = recognizer.recognize_google(audio)
        return text
    except sr.UnknownValueError:
        return "Sorry, I could not understand the audio."
    except sr.RequestError:
        return "Speech Recognition service is not available."

# Function for text-to-speech (remains the same)
def speak_text(text):
    tts = gTTS(text)
    audio_bytes = BytesIO()
    tts.write_to_fp(audio_bytes)
    st.audio(audio_bytes, format="audio/mp3")

# UI Layout using HTML structure
st.markdown('<div class="wrapper">', unsafe_allow_html=True)
st.markdown('<div class="title">AI Counselor SereniBot</div>', unsafe_allow_html=True)
st.markdown('<div class="box" id="chat-box">', unsafe_allow_html=True)

for msg in st.session_state["conversation_history"]:
    role_class = "right" if msg["role"] == "user" else ""
    icon = '<i class="fa fa-user"></i>' if msg["role"] == "user" else '<i class="fa fa-android"></i>'
    st.markdown(f"""
        <div class="item {role_class}">
            <div class="icon">
                {icon}
            </div>
            <div class="msg">
                <p>{msg['content']}</p>
            </div>
        </div>
        <br clear="both">
    """, unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True) # Close box div

with st.markdown('<div class="typing-area">', unsafe_allow_html=True):
    with st.markdown('<div class="input-field">', unsafe_allow_html=True):
        user_message = st.text_input("", placeholder="Type your message", key="user_input")
        if st.button("Send", key="send_button"):
            if user_message:
                with st.spinner("Thinking..."):
                    ai_response = generate_response(user_message)
                    mental_health_insight = detect_mental_health_signs(user_message)

                st.markdown(f"""
                    <div class="item right">
                        <div class="msg">
                            <p>{user_message}</p>
                        </div>
                    </div>
                    <br clear="both">
                """, unsafe_allow_html=True)
                st.markdown(f"""
                    <div class="item">
                        <div class="icon">
                            <i class="fa fa-android"></i>
                        </div>
                        <div class="msg">
                            <p>{ai_response}</p>
                        </div>
                    </div>
                    <br clear="both">
                """, unsafe_allow_html=True)
                st.markdown(f"<p style='color: green;'>**🧠 Mental Health Insight:** {mental_health_insight}</p>", unsafe_allow_html=True)
                speak_text(ai_response)
                st.session_state["user_input"] = ""
                st.rerun()
    st.markdown('</div>', unsafe_allow_html=True) # Close input-field div
st.markdown('</div>', unsafe_allow_html=True) # Close typing-area div
st.markdown('</div>', unsafe_allow_html=True) # Close wrapper div

# Sidebar Tools (remains the same)
st.sidebar.title("🔧 Tools")
if st.sidebar.button("Give me a positive affirmation"):
    affirmation = generate_response("Provide a positive affirmation.")
    st.sidebar.markdown(f"🌟 **Affirmation:** {affirmation}")

if st.sidebar.button("Give me a guided meditation"):
    meditation = generate_response("Provide a 5-minute guided meditation script.")
    st.sidebar.markdown(f"🧘 **Guided Meditation:** {meditation}")

if st.sidebar.button("Give me self-care advice"):
    self_care_tips = generate_response("Provide self-care tips.")
    st.sidebar.markdown(f"💖 **Self-Care Tips:** {self_care_tips}")

if st.sidebar.button("Give me a CBT technique"):
    cbt_exercise = generate_response("Provide a CBT exercise to manage negative thoughts.")
    st.sidebar.markdown(f"🧠 **CBT Exercise:** {cbt_exercise}")
#Speech to text
def recognize_speech():
    recognizer = sr.Recognizer()
    # Optional: List microphone names for debugging
    # print(sr.Microphone.list_microphone_names())

    # Optional: Specify a specific microphone
    # microphone = sr.Microphone(device_name="your_microphone_name")
    microphone = sr.Microphone() # Use default microphone if not specified

    with microphone as source:
        st.write("🎤 Listening...")
        audio = recognizer.listen(source)
    try:
        text = recognizer.recognize_google(audio)
        return text
    except sr.UnknownValueError:
        return "Sorry, I could not understand the audio."
    except sr.RequestError:
        return "Speech Recognition service is not available."
