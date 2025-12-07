import streamlit as st
import cv2
from ultralytics import YOLO
import numpy as np
import tempfile
import os
import pandas as pd
import plotly.express as px

# Gemini + 
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import initialize_agent, Tool, AgentType
from collections import Counter

# ================================
# LOAD MODELS
# ================================
FACE_MODEL_PATH = "../../MSAAI_501_Project_FacialEmotionDetection/yolov8n-face.pt"
EMOTION_MODEL_PATH = "../../MSAAI_501_Project_FacialEmotionDetection/train50/content/runs/detect/train/weights/best.pt"

face_model = YOLO(FACE_MODEL_PATH)
emotion_model = YOLO(EMOTION_MODEL_PATH)

# ================================
# STREAMLIT PAGE CONFIG
# ================================
st.set_page_config(page_title="Emotion Detection Dashboard", page_icon="😊", layout="wide")

st.title("📊 Facial Emotion Detection Dashboard (Gemini Powered)")
st.write("Upload face images → Detect emotions → View AI-generated summaries using Google's Gemini model.")

# Load the API key from an environment variable or use a default value

GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')

print("Using Gemini API Key:", GOOGLE_API_KEY)

# Configure the Gemini API
genai.configure(api_key=GOOGLE_API_KEY)


# ================================
# GEMINI AGENT SUMMARY FUNCTION
# ================================
def generate_emotion_summary_agent(df_results, emotion_counts):

    # Convert emotion data to text for the LLM
    table_text = df_results.to_string(index=False)
    count_text = ", ".join([f"{k}: {v}" for k, v in emotion_counts.items()])
    total_people = len(df_results)

    # Tool must accept a single parameter (even if unused)
    def emotion_data(_input: str = None):
        return (
            f"Total People: {total_people}\n\n"
            f"Emotion Table:\n{table_text}\n\n"
            f"Counts: {count_text}\n\n"
        )

    tools = [
        Tool(
            name="EmotionData",
            func=emotion_data,
            description="Returns detailed emotion detection statistics based on the uploaded image."
        )
    ]

    # Gemini LLM
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",      # or gemini-1.5-pro
        temperature=0.4,
        google_api_key=GOOGLE_API_KEY
    )

    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=False
    )

    prompt = """
Use the EmotionData tool to read the detection results.

Write a concise, professional summary including:
- How many people were detected
- Which emotion is most common and how often it appears
- Overall emotion distribution
- Key emotional patterns or insights
"""

    summary = agent.run(prompt)
    return summary



# ================================
# FILE UPLOAD
# ================================
uploaded_file = st.file_uploader("📤 Upload an Image", type=["jpg", "jpeg", "png"])

if uploaded_file:

    temp_dir = tempfile.mkdtemp()
    img_path = os.path.join(temp_dir, uploaded_file.name)

    with open(img_path, "wb") as f:
        f.write(uploaded_file.read())

    # Load image
    image = cv2.imread(img_path)
    orig = image.copy()


   

    # ---------------------------------------------
    # FACE DETECTION
    # ---------------------------------------------
    face_results = face_model(image, conf=0.3)
    emotions = []
    person_results = []

    for i, r in enumerate(face_results):
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # Crop face
            face_crop = orig[y1:y2, x1:x2]

            # ---------------------------------------------
            # EMOTION PREDICTION FOR THIS FACE
            # ---------------------------------------------
            emo_result = emotion_model.predict(face_crop, conf=0.1, verbose=False)

            if len(emo_result[0].boxes):
                cls = int(emo_result[0].boxes.cls[0])
                emotion = emotion_model.names[cls]

                # Save emotion
                emotions.append(emotion)

                # Save for table
                person_results.append({
                    "Person": len(person_results) + 1,
                    "Emotion": emotion
                })

                # Draw rectangle
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(image, emotion, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)


    # ---------------------------------------------
    # SUMMARY STATISTICS
    # ---------------------------------------------
    emotion_counts = Counter(emotions)
    total = sum(emotion_counts.values())
    emotion_percent = {k: round((v / total) * 100, 1) for k, v in emotion_counts.items()}

    # Convert to DataFrame (for Streamlit table)
    df_results = pd.DataFrame(person_results)


    # ================================
    # UI LAYOUT
    # ================================
    col1, col2 = st.columns([1.5, 1])

    with col1:
        st.subheader("📋 Detected Emotions (Per Person)")
        st.dataframe(df_results, use_container_width=True)

        # PIE CHART
        st.subheader("📊 Emotion Distribution")
        total_faces = sum(emotion_counts.values())

        emotion_percent = {
            emo: round((count / total_faces) * 100, 1)
            for emo, count in emotion_counts.items()
        }

        fig = px.pie(
            names=list(emotion_percent.keys()),
            values=list(emotion_percent.values()),
            title="Emotion Percentage Distribution",
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        st.plotly_chart(fig, use_container_width=True)

        # GEMINI SUMMARY
        st.subheader("🤖 Gemini AI Summary")
        ai_summary = generate_emotion_summary_agent(df_results, emotion_counts)
        st.write(ai_summary)

    with col2:
        st.subheader("🖼️ Annotated Output Image")
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        st.image(image_rgb, use_container_width=True)
