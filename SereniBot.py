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

# Load NLP model for mental health analysis
nlp = spacy.load("en_core_web_sm")
sentiment_analyzer = pipeline("sentiment-analysis")

def get_base64(background):
    with open(background, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

# Background Image Setup
background_path = "C:\\Users\\rishi\\OneDrive\\Desktop\\Coding\\Python\\images.jpg"
if os.path.exists(background_path):
    bin_str = get_base64(background_path)
    st.markdown(
        f"""
        <style>
            .stApp {{ 
                background-image: url("data:image/png;base64,{bin_str}");
                background-size: cover;
                background-position: center;
                background-repeat: no-repeat;
            }}
        </style>
        """,
        unsafe_allow_html=True,
    )

# Initialize session states
if "conversation_history" not in st.session_state:
    st.session_state["conversation_history"] = []
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

# Function for voice recognition
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

# Function for text-to-speech
def speak_text(text):
    tts = gTTS(text)
    audio_bytes = BytesIO()
    tts.write_to_fp(audio_bytes)
    st.audio(audio_bytes, format="audio/mp3")

# UI Layout
st.markdown("# 🤖 SereniBot - Your AI Companion for Well-Being")
st.markdown("### How can I assist you today?")

with st.container():
    for msg in st.session_state["conversation_history"]:
        role = "🧑 You" if msg["role"] == "user" else "🤖 AI"
        st.markdown(f"**{role}:** {msg['content']}", unsafe_allow_html=True)

# Voice input option
if st.button("🎤 Speak instead of typing"):
    user_message = recognize_speech()
else:
    user_message = st.text_input("Type your message here...")

if user_message:
    with st.spinner("Thinking..."):
        ai_response = generate_response(user_message)
        mental_health_insight = detect_mental_health_signs(user_message)
    
    st.markdown(f"**🧠 Mental Health Insight:** {mental_health_insight}", unsafe_allow_html=True)
    st.markdown(f"**🤖 AI Response:** {ai_response}", unsafe_allow_html=True)
    speak_text(ai_response)

# Sidebar Tools
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
