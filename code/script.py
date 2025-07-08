import os
import librosa
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import streamlit as st
import tempfile
import matplotlib.pyplot as plt
import seaborn as sns

# ----- Your dataset path -----
DATA_PATH = r"E:\CodeAplha\Data"

# Emotion mapping
emotion_map = {
    'angry': 'Angry',
    'disgust': 'Disgust',
    'fear': 'Fear',
    'happy': 'Happy',
    'ps': 'Pleasant Surprise',
    'sad': 'Sad',
    'neutral': 'Neutral'
}

# Descriptions for UI info
emotion_descriptions = {
    'Angry': 'Feeling or showing strong annoyance, displeasure, or hostility.',
    'Disgust': 'A feeling of revulsion or profound disapproval.',
    'Fear': 'An unpleasant emotion caused by the threat of danger or harm.',
    'Happy': 'Feeling or showing pleasure or contentment.',
    'Pleasant Surprise': 'A feeling of unexpected happiness or delight.',
    'Sad': 'Feeling sorrow or unhappiness.',
    'Neutral': 'Neither positive nor negative emotion; calm or indifferent.'
}

def get_emotion(filename):
    for key in emotion_map:
        if key in filename.lower():
            return emotion_map[key]
    return None

def extract_features(file_path):
    try:
        audio, sr = librosa.load(file_path, res_type='kaiser_fast')
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
        return np.mean(mfcc.T, axis=0)
    except Exception as e:
        print(f"Error: {file_path} - {e}")
        return None

# Dataset class
class EmotionDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X)
        self.y = torch.tensor(y, dtype=torch.long)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# Model definition
class EmotionModel(nn.Module):
    def __init__(self, num_classes):
        super(EmotionModel, self).__init__()
        self.fc1 = nn.Linear(40, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.out = nn.Linear(64, num_classes)
        self.dropout = nn.Dropout(0.3)
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        x = self.dropout(x)
        return self.out(x)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@st.cache_resource
def load_data_and_train(epochs):
    features, labels = [], []

    for root, _, files in os.walk(DATA_PATH):
        for filename in files:
            if filename.endswith(".wav"):
                emotion = get_emotion(filename)
                if emotion:
                    path = os.path.join(root, filename)
                    feat = extract_features(path)
                    if feat is not None:
                        features.append(feat)
                        labels.append(emotion)

    X = np.array(features, dtype=np.float32)
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(labels)

    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)
    train_dataset = EmotionDataset(X_train, y_train)
    test_dataset = EmotionDataset(X_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32)

    model = EmotionModel(num_classes=len(label_encoder.classes_)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    progress_bar = st.progress(0)
    loss_text = st.empty()

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            output = model(X_batch)
            loss = criterion(output, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)
        loss_text.text(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}")
        progress_bar.progress((epoch + 1) / epochs)

    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            outputs = model(X_batch)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            y_true.extend(y_batch.numpy())
            y_pred.extend(preds)

    acc = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)

    return model, label_encoder, acc, cm

def predict_emotion_from_audio(model, label_encoder, audio_path):
    model.eval()
    feat = extract_features(audio_path)
    if feat is not None:
        feat_tensor = torch.tensor(feat, dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(feat_tensor)
            predicted = torch.argmax(output, dim=1).item()
            return label_encoder.inverse_transform([predicted])[0]
    return "Unknown"


# ----------- NEW STREAMLIT UI -----------

# Page config
st.set_page_config(
    page_title="Voice Emotion Detector",
    page_icon="🗣️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS to style the app differently
st.markdown(
    """
    <style>
    /* Background gradient */
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }

    /* Title style */
    .title {
        font-size: 3.5rem;
        font-weight: 700;
        text-align: center;
        margin-bottom: 0;
        letter-spacing: 2px;
        text-shadow: 2px 2px 6px #3a136a;
    }

    /* Subtitle style */
    .subtitle {
        font-size: 1.25rem;
        font-weight: 400;
        text-align: center;
        margin-top: 0.1rem;
        margin-bottom: 2.5rem;
        color: #dcdcdccc;
    }

    /* Upload container style */
    .upload-container {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 20px;
        padding: 25px;
        max-width: 600px;
        margin: 0 auto 2rem;
        box-shadow: 0 8px 20px rgba(0,0,0,0.2);
    }

    /* Prediction box */
    .prediction-box {
        background: rgba(255, 255, 255, 0.2);
        border-radius: 18px;
        padding: 20px;
        max-width: 500px;
        margin: 1rem auto 2rem;
        text-align: center;
        box-shadow: 0 6px 18px rgba(0,0,0,0.3);
        font-size: 1.5rem;
        font-weight: 600;
        letter-spacing: 1.2px;
    }

    /* Button styles */
    div.stButton > button {
        background-color: #a892ee;
        color: #1b0e4f;
        font-weight: 600;
        padding: 12px 30px;
        border-radius: 12px;
        border: none;
        box-shadow: 0 5px 15px rgba(168, 146, 238, 0.6);
        transition: background-color 0.3s ease;
        width: 100%;
        max-width: 280px;
        margin: 0 auto;
        display: block;
    }
    div.stButton > button:hover {
        background-color: #816ad1;
        color: white;
        cursor: pointer;
    }

    /* Slider style */
    .stSlider > div > div > div > div {
        color: #dcdcdccc;
    }

    /* Sidebar - hidden */
    .css-1v3fvcr {
        display: none;
    }
    </style>
    """, unsafe_allow_html=True
)

# Main title & subtitle
st.markdown('<h1 class="title">Voice Emotion Detector</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Upload your speech audio and let the AI sense your emotions!</p>', unsafe_allow_html=True)

# Upload audio widget inside styled container
with st.container():
    st.markdown('<div class="upload-container">', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("🎧 Upload a WAV file to analyze", type=["wav"])
    st.markdown('</div>', unsafe_allow_html=True)

# Training epochs slider in a centered container
epochs = st.slider("Select Training Epochs", min_value=10, max_value=100, value=30, step=5, key="epochs", help="More epochs = better accuracy but longer training time.")

# Train model button centered
train_button = st.button("🔄 Train / Retrain Model")

# Status placeholder
status_placeholder = st.empty()

if 'model' not in st.session_state or train_button:
    if uploaded_file is None and not train_button:
        status_placeholder.info("Upload a WAV file and/or retrain the model to start.")
    else:
        with st.spinner("Training the model... This may take a few minutes ⏳"):
            model, label_encoder, test_acc, cm = load_data_and_train(epochs)
            st.session_state['model'] = model
            st.session_state['label_encoder'] = label_encoder
            st.session_state['test_acc'] = test_acc
            st.session_state['conf_matrix'] = cm
        status_placeholder.success(f"Model trained! Test Accuracy: {test_acc*100:.2f}%")

else:
    model = st.session_state['model']
    label_encoder = st.session_state['label_encoder']

# Safely fetch test accuracy and confusion matrix from session state
test_acc = st.session_state.get('test_acc', None)
cm = st.session_state.get('conf_matrix', None)

# If model trained, show confusion matrix and accuracy in a sleek card
if test_acc is not None and cm is not None:
    with st.container():
        st.markdown(f"""
        <div style="background: rgba(255, 255, 255, 0.15); border-radius: 15px; padding: 15px; max-width: 550px; margin: 1rem auto; box-shadow: 0 4px 12px rgba(0,0,0,0.3);">
            <h3 style="text-align:center; color:#fff; margin-bottom:10px;">Model Performance</h3>
            <p style="text-align:center; font-weight:600; font-size:1.2rem;">Test Accuracy: <span style="color:#ffd700;">{test_acc*100:.2f}%</span></p>
        </div>
        """, unsafe_allow_html=True)

        fig, ax = plt.subplots(figsize=(7,5))
        sns.heatmap(cm, annot=True, fmt='d', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_, cmap="coolwarm", ax=ax)
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix')
        st.pyplot(fig)

# Predict emotion from uploaded audio
if uploaded_file is not None and 'model' in st.session_state:
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
        tmp_file.write(uploaded_file.getbuffer())
        temp_path = tmp_file.name

    st.audio(temp_path, format="audio/wav")

    prediction = predict_emotion_from_audio(model, label_encoder, temp_path)

    # Display prediction with emoji & description box
    emoji_map = {
        'Angry': "😠",
        'Disgust': "🤢",
        'Fear': "😨",
        'Happy': "😄",
        'Pleasant Surprise': "😲",
        'Sad': "😢",
        'Neutral': "😐"
    }
    emoji = emoji_map.get(prediction, "❓")
    description = emotion_descriptions.get(prediction, "")

    st.markdown(
        f"""
        <div style="max-width: 600px; margin: 2rem auto; background: rgba(255, 255, 255, 0.2); border-radius: 20px; padding: 25px; text-align: center; box-shadow: 0 6px 18px rgba(0,0,0,0.4);">
            <h2 style="font-weight:700; font-size: 2.8rem; margin-bottom: 0;">{emoji} {prediction}</h2>
            <p style="font-size: 1.3rem; font-style: italic; margin-top: 10px; color: #eee;">{description}</p>
        </div>
        """, unsafe_allow_html=True
    )
