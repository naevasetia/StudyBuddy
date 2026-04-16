# FINAL attentive/distracted detection (App-friendly version)
# attentiveORdistracted_copy.py
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
import cv2
import numpy as np
from pathlib import Path
import time
from datetime import datetime
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# ==================== CONFIGURATION ====================
BASE_DIR = Path(__file__).resolve().parent

BEHAVIORAL_MODEL_PATH = BASE_DIR / "models" / "attention_model_best.pth"
EMOTION_MODEL_PATH = BASE_DIR / "models" / "face_model.h5"

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
CAPTURE_INTERVAL = 120
SEQUENCE_DURATION = 10
FPS = 1


class AttentionDetectionModel(nn.Module):
    # Model Architecture
    def __init__(self):
        super().__init__()

        resnet = models.resnet50(weights=None)
        self.spatial_features = nn.Sequential(*list(resnet.children())[:-1])

        self.temporal_lstm = nn.LSTM(
            input_size=2048,
            hidden_size=512,
            num_layers=2,
            batch_first=True,
            dropout=0.3,
            bidirectional=True
        )

        self.attention = nn.Sequential(
            nn.Linear(1024, 256),
            nn.Tanh(),
            nn.Linear(256, 1)
        )

        self.classifier = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 4)
        )

    def forward(self, x):
        batch_size, seq_len, C, H, W = x.shape

        x = x.view(batch_size * seq_len, C, H, W)
        spatial_feat = self.spatial_features(x)
        spatial_feat = spatial_feat.view(batch_size, seq_len, -1)

        temporal_feat, _ = self.temporal_lstm(spatial_feat)

        attention_weights = self.attention(temporal_feat)
        attention_weights = torch.softmax(attention_weights, dim=1)
        weighted_feat = torch.sum(temporal_feat * attention_weights, dim=1)

        output = self.classifier(weighted_feat)

        if not self.training:
            output = torch.clamp(output, 0, 3)

        return output


def load_behavioral_model(model_path, device):
    print(f"Loading behavioral model from {model_path}...")

    if not Path(model_path).exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    model = AttentionDetectionModel().to(device)
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    # Load model weights
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Behavioral model loaded successfully")
    else:
        model.load_state_dict(checkpoint)
        print("Behavioral model loaded successfully")

    model.eval()
    return model


def load_emotion_model(model_path):
    print(f"Loading emotion model from {model_path}...")

    if not Path(model_path).exists():
        raise FileNotFoundError(f"Emotion model file not found: {model_path}")

    try:
        emotion_model = load_model(model_path)
        print("Emotion model loaded successfully")
        return emotion_model
    except Exception as e:
        raise RuntimeError(f"Error loading emotion model: {e}")


def detect_emotion_from_frame(frame, emotion_model, face_cascade):
    # Convert RGB to BGR for OpenCV
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))

    if len(faces) == 0:
        # No face detected, return neutral as default
        return 'neutral', 0.0

    # Use the largest detected face
    (x, y, w, h) = faces[0]
    face_roi = frame_bgr[y:y + h, x:x + w]

    # Preprocess face for emotion model
    face_image = cv2.resize(face_roi, (48, 48))
    face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
    face_image = image.img_to_array(face_image)
    face_image = np.expand_dims(face_image, axis=0)

    # Predict emotion
    predictions = emotion_model.predict(face_image, verbose=0)
    emotion_idx = np.argmax(predictions)
    confidence = np.max(predictions)

    emotion_map = {
        0: 'anger',
        1: 'neutral',
        2: 'neutral',
        3: 'happy',
        4: 'sad',
        5: 'neutral',
        6: 'neutral'
    }

    emotion = emotion_map.get(emotion_idx, 'neutral')

    if emotion not in ['happy', 'sad', 'neutral', 'anger']:
        emotion = 'neutral'

    return emotion, confidence


def detect_emotion_from_frames(frames, emotion_model, face_cascade):
    emotions = []
    confidences = []

    print("Detecting emotions from frames...")

    for i, frame in enumerate(frames):
        emotion, confidence = detect_emotion_from_frame(frame, emotion_model, face_cascade)
        emotions.append(emotion)
        confidences.append(confidence)
        print(f"Frame {i+1}/10: {emotion} (confidence: {confidence:.2f})", end='\r')

    print()

    # Find most common emotion
    from collections import Counter
    emotion_counts = Counter(emotions)
    dominant_emotion = emotion_counts.most_common(1)[0][0]
    avg_confidence = np.mean(confidences)

    print(f"Dominant emotion: {dominant_emotion.upper()} (avg confidence: {avg_confidence:.2f})")
    print(f"Emotion distribution: {dict(emotion_counts)}")

    return dominant_emotion, avg_confidence


def apply_emotion_modifiers(engagement, boredom, confusion, frustration, emotion):
    """
    Apply emotion-based adjustments to behavioral scores

    RESTRICTED to only 4 emotions: happy, sad, neutral, anger
    """

    if emotion not in ['happy', 'sad', 'neutral', 'anger']:
        emotion = 'neutral'

    if emotion == 'happy':
        engagement_adj = engagement + 1
        boredom_adj = boredom - 1
        confusion_adj = confusion + 0
        frustration_adj = frustration - 0.5
    elif emotion == 'sad':
        engagement_adj = engagement - 1
        boredom_adj = boredom + 1
        confusion_adj = confusion + 0.5
        frustration_adj = frustration + 0.5
    elif emotion == 'anger':
        engagement_adj = engagement - 1
        boredom_adj = boredom + 1
        confusion_adj = confusion + 0.5
        frustration_adj = frustration + 0.5
    else:
        engagement_adj = engagement
        boredom_adj = boredom
        confusion_adj = confusion
        frustration_adj = frustration

    return engagement_adj, boredom_adj, confusion_adj, frustration_adj


def calculate_attentiveness_score(engagement, boredom, confusion, frustration, emotion):
    """
    Calculate attentiveness score using weighted formula

    Returns:
        score: Normalized attentiveness score (0-10)
        classification: 'Attentive' or 'Distracted'
        details: Dictionary with intermediate values
    """

    # Step 1: Apply emotion modifiers
    eng_adj, bor_adj, conf_adj, frust_adj = apply_emotion_modifiers(
        engagement, boredom, confusion, frustration, emotion
    )

    # Step 2: Calculate weighted raw score
    # Weights: engagement=5, boredom=3, confusion=1, frustration=1
    raw_score = (
        5 * eng_adj +
        3 * (3 - bor_adj) +
        1 * (3 - conf_adj) +
        1 * (3 - frust_adj)
    )

    # Step 3: Center raw score, apply sigmoid, then scale to 0-10
    centered_raw = raw_score - 19.25
    sigmoid_score = 1 / (1 + np.exp(-centered_raw))
    normalized_score = sigmoid_score * 10

    # Ensure score is within bounds
    normalized_score = np.clip(normalized_score, 0, 10)

    # Step 4: Classify
    classification = 'Attentive' if normalized_score >= 7.0 else 'Distracted'

    details = {
        'base_scores': {
            'engagement': engagement,
            'boredom': boredom,
            'confusion': confusion,
            'frustration': frustration
        },
        'adjusted_scores': {
            'engagement': eng_adj,
            'boredom': bor_adj,
            'confusion': conf_adj,
            'frustration': frust_adj
        },
        'raw_score': raw_score,
        'normalized_score': normalized_score,
        'classification': classification,
        'emotion': emotion
    }

    return normalized_score, classification, details


def capture_frames_from_webcam(duration=10, fps=1):
    """
    Capture frames from webcam at specified FPS for given duration

    Args:
        duration: Duration in seconds (default: 10)
        fps: Frames per second (default: 1)

    Returns:
        frames: List of captured frames (numpy arrays)

    Raises:
        RuntimeError if webcam cannot be opened or no frames are captured
    """
    # Try DirectShow backend on Windows – avoids some MSMF issues
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    if not cap.isOpened():
        cap.release()
        raise RuntimeError(
            "Could not open webcam. It may be in use by another app "
            "(for example, your browser preview)."
        )

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    frames = []
    frame_interval = 1.0 / fps
    num_frames = duration * fps
    max_failures = 20
    fail_count = 0

    print(f"Capturing {num_frames} frames at {fps} FPS for {duration} seconds...")

    start_time = time.time()
    next_capture_time = start_time

    while len(frames) < num_frames:
        ret, frame = cap.read()

        if not ret or frame is None:
            fail_count += 1
            print("Warning: Failed to capture frame")
            if fail_count >= max_failures:
                print("Too many capture failures, aborting capture.")
                break
            # small sleep so we don't busy-loop
            time.sleep(0.1)
            continue

        fail_count = 0
        current_time = time.time()

        # Capture frame at specified intervals
        if current_time >= next_capture_time:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)
            print(f"Frame {len(frames)}/{num_frames} captured", end="\r")
            next_capture_time += frame_interval

    cap.release()
    print(f"\nCaptured {len(frames)} frames successfully")

    if len(frames) == 0:
        raise RuntimeError(
            "Failed to capture frames from webcam. "
            "Close other apps that use the camera and try again."
        )

    return frames



def preprocess_frames(frames, transform):
    """
    Preprocess captured frames for model input

    Returns:
        tensor: Preprocessed frames tensor [1, num_frames, 3, 224, 224]
    """
    processed_frames = []

    for frame in frames:
        # Apply transforms
        frame_tensor = transform(frame)
        processed_frames.append(frame_tensor)

    # Stack frames and add batch dimension
    frames_tensor = torch.stack(processed_frames).unsqueeze(0)

    return frames_tensor


def predict_attentiveness(behavioral_model, emotion_model, face_cascade, frames, device):
    """
    Make attentiveness prediction from frames

    Returns:
        score: Attentiveness score (0-10)
        classification: 'Attentive' or 'Distracted'
        details: Dictionary with all details
    """

    # Step 1: Detect emotion from frames
    emotion, emotion_confidence = detect_emotion_from_frames(frames, emotion_model, face_cascade)

    # Step 2: Define preprocessing transform
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Step 3: Preprocess frames
    frames_tensor = preprocess_frames(frames, transform).to(device)

    # Step 4: Get behavioral predictions
    with torch.no_grad():
        predictions = behavioral_model(frames_tensor)

    # Extract predictions (batch_size=1)
    preds = predictions.cpu().numpy()[0]

    boredom = float(preds[0])
    engagement = float(preds[1])
    confusion = float(preds[2])
    frustration = float(preds[3])

    # Step 5: Calculate attentiveness score using detected emotion
    score, classification, details = calculate_attentiveness_score(
        engagement, boredom, confusion, frustration, emotion
    )

    # Add emotion confidence to details
    details['emotion_confidence'] = emotion_confidence

    return score, classification, details


# ============================================================
#  NEW: Cached models + single-shot API for your app
# ============================================================

_behavioral_model = None
_emotion_model = None
_face_cascade = None


def init_models():
    """
    Load models once and cache them in module-level globals.
    Safe to call multiple times.
    """
    global _behavioral_model, _emotion_model, _face_cascade

    if _behavioral_model is None:
        _behavioral_model = load_behavioral_model(BEHAVIORAL_MODEL_PATH, DEVICE)

    if _emotion_model is None:
        _emotion_model = load_emotion_model(EMOTION_MODEL_PATH)

    if _face_cascade is None:
        _face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )


def run_attentiveness_check(duration: int = SEQUENCE_DURATION,
                            fps: int = FPS) -> dict:
    """
    Single 10-second webcam check for integration.

    Captures frames, runs both models and returns a JSON-friendly dict:
      {
        "score": float,
        "classification": "Attentive" | "Distracted",
        "emotion": "happy" | "sad" | "neutral" | "anger",
        "raw_score": float,
        "base_scores": {...},
        "adjusted_scores": {...},
        "emotion_confidence": float
      }
    """
    init_models()

    # 1) capture frames
    frames = capture_frames_from_webcam(duration=duration, fps=fps)

    # 2) predict
    score, classification, details = predict_attentiveness(
        _behavioral_model, _emotion_model, _face_cascade, frames, DEVICE
    )

    # 3) build result dict with plain Python types
    result = {
        "score": float(details.get("normalized_score", score)),
        "classification": str(classification),
        "emotion": str(details.get("emotion", "neutral")),
        "raw_score": float(details.get("raw_score", 0.0)),
        "base_scores": {
            k: float(v) for k, v in details.get("base_scores", {}).items()
        },
        "adjusted_scores": {
            k: float(v) for k, v in details.get("adjusted_scores", {}).items()
        },
        "emotion_confidence": float(details.get("emotion_confidence", 0.0)),
    }
    return result


# ============================================================

def display_results(score, classification, details):
    """Display prediction results"""

    print("\n" + "=" * 70)
    print("ATTENTIVENESS PREDICTION RESULTS")
    print("=" * 70)

    print(f"Base Behavioral Scores (Model Output):")
    base = details['base_scores']
    print(f"  • Engagement:  {base['engagement']:.2f} / 3.00")
    print(f"  • Boredom:     {base['boredom']:.2f} / 3.00")
    print(f"  • Confusion:   {base['confusion']:.2f} / 3.00")
    print(f"  • Frustration: {base['frustration']:.2f} / 3.00")

    print(f"Emotion Detected: {details['emotion'].upper()}")

    print(f"Adjusted Scores (After Emotion Modifier):")
    adj = details['adjusted_scores']
    print(f"  • Engagement:  {adj['engagement']:.2f}")
    print(f"  • Boredom:     {adj['boredom']:.2f}")
    print(f"  • Confusion:   {adj['confusion']:.2f}")
    print(f"  • Frustration: {adj['frustration']:.2f}")

    print(f"Calculation:")
    print(f"  • Raw Score: {details['raw_score']:.2f}")
    print(f"  • Normalized Score: {score:.2f} / 10.00")

    print(f"\n{'=' * 70}")
    print(f"RESULT: {classification.upper()} (Score: {score:.2f}/10)")
    print("=" * 70 + "\n")


def main():
    """Main function to run continuous attentiveness monitoring (CLI use)"""

    print("\n" + "=" * 70)
    print("LIVE WEBCAM ATTENTIVENESS MONITORING SYSTEM")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  • Behavioral Model: {BEHAVIORAL_MODEL_PATH}")
    print(f"  • Emotion Model: {EMOTION_MODEL_PATH}")
    print(f"  • Device: {DEVICE}")
    print(f"  • Capture Interval: Every {CAPTURE_INTERVAL} seconds")
    print(f"  • Sequence Duration: {SEQUENCE_DURATION} seconds @ {FPS} FPS")
    print(f"  • Total Frames per Capture: {SEQUENCE_DURATION * FPS}")

    # Load behavioral model
    try:
        behavioral_model = load_behavioral_model(BEHAVIORAL_MODEL_PATH, DEVICE)
    except Exception as e:
        print(f"Error loading behavioral model: {e}")
        return

    # Load emotion model
    try:
        emotion_model = load_emotion_model(EMOTION_MODEL_PATH)
    except Exception as e:
        print(f"Error loading emotion model: {e}")
        return

    # Load face cascade for emotion detection
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    print("System ready. Press Ctrl+C to stop.\n")

    capture_count = 0

    try:
        while True:
            capture_count += 1
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            print(f"\n{'=' * 70}")
            print(f"Capture #{capture_count} - {timestamp}")
            print(f"{'=' * 70}")

            try:
                # Capture frames from webcam
                frames = capture_frames_from_webcam(
                    duration=SEQUENCE_DURATION,
                    fps=FPS
                )

                # Make prediction
                print("Making prediction...")
                score, classification, details = predict_attentiveness(
                    behavioral_model, emotion_model, face_cascade, frames, DEVICE
                )

                # Display results
                display_results(score, classification, details)

            except Exception as e:
                print(f"Error during capture/prediction: {e}")
                print("Retrying in next cycle...")

            # Wait for next capture interval
            if capture_count == 1:
                print(f"Waiting {CAPTURE_INTERVAL} seconds until next capture...")
            else:
                print(f"Next capture in {CAPTURE_INTERVAL} seconds...")

            time.sleep(CAPTURE_INTERVAL)

    except KeyboardInterrupt:
        print("Monitoring stopped by user")
        print(f"Total captures: {capture_count}")
        print("System shutdown complete")


if __name__ == "__main__":
    main()
