# main_app.py — Skin Disease Predictor

import streamlit as st
import numpy as np
import json
import plotly.graph_objects as go
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from deep_translator import GoogleTranslator

# Page config
st.set_page_config(
    page_title="Skin Disease Predictor",
    page_icon="skin",
    layout="centered"
)

# Load model and labels
@st.cache_resource
def load_everything():
    model = load_model('model/skin_disease_model.h5')
    with open('model/class_labels.json', 'r') as f:
        class_indices = json.load(f)
    labels = {v: k for k, v in class_indices.items()}
    return model, labels

model, labels = load_everything()

# Disease info
disease_info = {
    'nv'    : {
        'name' : 'Melanocytic Nevi',
        'desc' : 'Common benign moles that appear as brown or black spots. Usually harmless but monitor for changes.'
    },
    'mel'   : {
        'name' : 'Melanoma',
        'desc' : 'A serious form of skin cancer. Early detection is critical for successful treatment.'
    },
    'bkl'   : {
        'name' : 'Benign Keratosis',
        'desc' : 'Non-cancerous skin growths that appear waxy. Very common in older adults.'
    },
    'bcc'   : {
        'name' : 'Basal Cell Carcinoma',
        'desc' : 'Most common skin cancer. Grows slowly. Highly treatable when caught early.'
    },
    'akiec' : {
        'name' : 'Actinic Keratosis',
        'desc' : 'Rough scaly patches caused by sun exposure. Can develop into skin cancer if untreated.'
    },
    'vasc'  : {
        'name' : 'Vascular Lesion',
        'desc' : 'Abnormalities of blood vessels in skin. Usually benign.'
    },
    'df'    : {
        'name' : 'Dermatofibroma',
        'desc' : 'Benign skin growths that feel like hard lumps. Common on legs and usually harmless.'
    }
}

# Language options
language_map = {
    'English' : 'en',
    'Telugu'  : 'te',
    'Hindi'   : 'hi'
}

# Translation
def translate_text(text, target_lang):
    if target_lang == 'en':
        return text
    try:
        return GoogleTranslator(
            source='en',
            target=target_lang
        ).translate(text)
    except:
        return text

# Preprocess image
def preprocess_image(image):
    img = image.convert('RGB')
    img = img.resize((224, 224))
    img = np.array(img)
    img = preprocess_input(img)
    img = np.expand_dims(img, axis=0)
    return img

# ── UI ──────────────────────────────────────────────────────

st.title("Skin Disease Predictor")
st.markdown("Upload a skin image to detect the disease with confidence scores.")
st.markdown("---")

# Language + Upload
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Step 1 - Select Language")
    selected_lang    = st.selectbox(
        "Choose your language",
        list(language_map.keys())
    )
    target_lang_code = language_map[selected_lang]

with col2:
    st.subheader("Step 2 - Upload Image")
    uploaded_file = st.file_uploader(
        "Choose a skin image",
        type=['jpg', 'jpeg', 'png']
    )

st.markdown("---")

if uploaded_file is not None:

    image = Image.open(uploaded_file)

    col_img, col_btn = st.columns([1, 1])

    with col_img:
        st.image(image, caption="Uploaded Image",
                 use_column_width=True)

    with col_btn:
        st.subheader("Step 3 - Predict")
        predict_btn = st.button(
            "Predict Disease",
            use_container_width=True
        )

    if predict_btn:
        with st.spinner("Analyzing image..."):

            # Predict
            processed   = preprocess_image(image)
            predictions = model.predict(processed)[0]

            # Top result
            top_idx     = np.argmax(predictions)
            top_disease = labels[top_idx]
            confidence  = predictions[top_idx] * 100

            # Disease info
            info       = disease_info[top_disease]
            eng_name   = info['name']
            eng_desc   = info['desc']

            # Translate
            trans_name = translate_text(eng_name, target_lang_code)
            trans_desc = translate_text(eng_desc, target_lang_code)

        st.markdown("---")
        st.subheader("Prediction Result")

        st.success(f"Detected Disease : {trans_name}")
        st.info(f"Confidence : {confidence:.2f}%")
        st.markdown(f"**Description :** {trans_desc}")

        st.markdown("---")

        # Confidence Bar Chart
        st.subheader("Confidence Scores - All Diseases")

        disease_names_display = [
            disease_info[labels[i]]['name']
            for i in range(7)
        ]
        confidence_scores = [
            predictions[i] * 100
            for i in range(7)
        ]
        colors = [
            '#4CAF50' if i == top_idx else '#2196F3'
            for i in range(7)
        ]

        bar_fig = go.Figure(go.Bar(
            x=confidence_scores,
            y=disease_names_display,
            orientation='h',
            marker_color=colors,
            text=[f"{s:.2f}%" for s in confidence_scores],
            textposition='outside'
        ))

        bar_fig.update_layout(
            title="Confidence % for each disease",
            xaxis_title="Confidence %",
            yaxis_title="Disease",
            height=400,
            xaxis=dict(range=[0, 115]),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )

        st.plotly_chart(bar_fig, use_container_width=True)

        # Gauge Meter
        st.subheader("Prediction Confidence Meter")

        gauge_fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=confidence,
            title={'text': "Confidence %"},
            delta={'reference': 50},
            gauge={
                'axis'  : {'range': [0, 100]},
                'bar'   : {'color': "#4CAF50"},
                'steps' : [
                    {'range': [0,  50],  'color': "#ffcccc"},
                    {'range': [50, 75],  'color': "#fff3cc"},
                    {'range': [75, 100], 'color': "#ccffcc"}
                ],
                'threshold': {
                    'line'     : {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value'    : 85
                }
            }
        ))

        gauge_fig.update_layout(
            height=350,
            paper_bgcolor='rgba(0,0,0,0)'
        )

        st.plotly_chart(gauge_fig, use_container_width=True)

        st.markdown("---")
        st.warning(
            "This prediction is for educational purposes only. "
            "Please consult a qualified dermatologist for medical advice."
        )

else:
    st.info("Please upload a skin image to get started.")