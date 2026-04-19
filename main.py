import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model

# ==============================
# Load Model
# ==============================
model = load_model("signal_classifier.h5")

# ==============================
# Class Labels
# ==============================
classes = [
    'OOK','4ASK','8ASK','BPSK','QPSK','8PSK',
    '16PSK','32PSK','16APSK','32APSK','64APSK','128APSK',
    '16QAM','32QAM','64QAM','128QAM','256QAM',
    'AM-SSB-WC','AM-SSB-SC','AM-DSB-WC','AM-DSB-SC',
    'FM','GMSK','OQPSK'
]

# ==============================
# Prediction Function
# ==============================
def predict_signal(model, signal, classes):
    signal = signal / np.max(np.abs(signal))
    signal = np.expand_dims(signal, axis=-1)
    signal = np.expand_dims(signal, axis=0)

    pred = model.predict(signal)
    pred_class = np.argmax(pred)
    confidence = np.max(pred)

    return classes[pred_class], confidence


# ==============================
# UI
# ==============================
st.title("📡 Intelligent Signal Classifier")

st.write("Upload a .npy file with shape (2,1024)")

uploaded_file = st.file_uploader("Choose a signal file", type=["npy"])

if uploaded_file is not None:
    
    # Load signal
    signal = np.load(uploaded_file)
    # FIX SHAPE
    if signal.shape == (1024, 2):
        signal = signal.T   # transpose
    
    if signal.shape != (2, 1024):
        st.error("Invalid shape. Expected (2,1024)")
        st.stop()


    st.write("📊 Signal Shape:", signal.shape)

    # Validate shape
    if signal.shape != (2, 1024):
        st.error(" Invalid shape. Expected (2,1024)")
    else:
        # Predict
        result, confidence = predict_signal(model, signal, classes)

        # Output
        st.success(f"Detected Signal: {result}")
        st.info(f"Confidence: {round(confidence*100,2)} %")

        # Optional: show probabilities
        pred = model.predict(np.expand_dims(np.expand_dims(signal, -1), 0))
        st.subheader("Prediction Probabilities")
        st.bar_chart(pred[0])