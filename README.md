# 🧠 StudyBuddy — AI-Powered Intelligent Study Assistant

> An intelligent desktop application that monitors student engagement in real-time through webcam-based behavioural and emotion analysis, and adapts the learning experience accordingly.

---

<img width="1600" height="934" alt="2" src="https://github.com/user-attachments/assets/46461184-e4ac-4266-8b1d-65ec348e209d" />
<img width="1600" height="940" alt="1" src="https://github.com/user-attachments/assets/05dc5dab-0803-4d2f-9cbd-99bd290acece" />
<img width="1291" height="889" alt="3" src="https://github.com/user-attachments/assets/5086a724-2023-4e6b-8a89-f12633ba0b7c" />
<img width="1342" height="924" alt="4" src="https://github.com/user-attachments/assets/acd234b3-07d2-4d97-8826-a3bb6f07d029" />
<img width="583" height="774" alt="5" src="https://github.com/user-attachments/assets/ef68e5c7-db6a-4c58-be63-583d8fc9241b" />
<img width="560" height="805" alt="6" src="https://github.com/user-attachments/assets/52e9ca5a-c66a-4b72-853b-c910583ac026" />


## 📌 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [AI Backend — Emotion & Behavioural Detection](#ai-backend--emotion--behavioural-detection)
  - [Dataset](#dataset-daisee)
  - [Emotion Detection Module](#emotion-detection-module)
  - [Behavioural Analysis Module](#behavioural-analysis-module)
  - [Integration & Scoring](#integration--attentiveness-scoring)
  - [Attentiveness Score Pipeline](#attentiveness-score-pipeline)
  - [Model Evolution & Training Challenges](#model-evolution--training-challenges)
- [Dashboard Analytics](#dashboard-analytics)
- [RAG Integration](#rag-integration)
- [Deployment](#deployment)
- [Limitations](#limitations)

---

## Overview

Traditional e-learning platforms track study time but cannot distinguish genuine focus from passive screen presence. StudyBuddy addresses this by combining real-time attention monitoring with an adaptive AI learning assistant.

The system uses a webcam to continuously analyse a student's facial expressions and behavioural cues, produce a live attentiveness score, and feed that score into a RAG-based chatbot that adjusts quiz difficulty, summarization depth, and feedback tone accordingly. Everything runs locally, no data leaves the device.

---

## Features

- **Real-time attentiveness detection** using dual-model pipeline (emotion + behaviour)
- **Adaptive quiz generation** — difficulty auto-adjusts based on attention score
- **RAG chatbot** — answers doubts and generates quizzes from uploaded PDFs
- **Pomodoro timer** with session logging and study streak tracking
- **Analytics dashboard** — focus trends, distraction ratios, session durations
- **Print report** — exportable productivity insight reports
- **Fully offline** — all inference runs on-device, no server required

---

## Tech Stack

| Layer | Technology |
|---|---|
| Frontend | Electron.js, React |
| AI Backend | Python, FastAPI |
| CV & Emotion | OpenCV, TensorFlow/Keras |
| Behavioural Model | PyTorch, ResNet-50, BiLSTM |
| RAG | ChromaDB, LangChain |
| Packaging | PyInstaller, Electron Builder |

---

## Project Structure

```
StudyBuddy/
├── ai-backend/
│   ├── Emotion_Behavior/
│   │   ├── models/               # Trained model weights (.h5 / .pth)
│   │   ├── __init__.py
│   │   ├── attentiveORdistracted.py   # Core inference pipeline
│   │   └── attentiveORdistracted_c.py
│   ├── ai_core.py                # RAG chatbot core
│   ├── embeddings.py             # ChromaDB vector store
│   ├── model.ipynb               # Training notebook
│   ├── requirements.txt
│   ├── scriptv4.py               # Attentiveness score computation
│   ├── server.py                 # FastAPI server entry point
│   └── start.sh
├── buddy-ui/                     # React UI components
├── dashboard-ui/                 # Analytics dashboard
├── main/
│   ├── main.js                   # Electron main process
│   └── preload.js
├── assets/
├── package.json
└── runtime.txt
```

---

## Getting Started

### Prerequisites

- Python 3.9+
- Node.js 18+
- Webcam

### Installation

```bash
# Clone the repository
git clone https://github.com/naevasetia/StudyBuddy.git
cd StudyBuddy

# Install Python dependencies
cd ai-backend
pip install -r requirements.txt
cd ..

# Install Node dependencies
npm install
```

### Running the App

You need two terminals running simultaneously.

**Terminal 1 — Start the AI backend:**
```bash
cd ai-backend
python server.py
```

**Terminal 2 — Start the Electron frontend:**
```bash
npm run dev
```

The desktop app window will open automatically once both processes are running.

---

## AI Backend — Emotion & Behavioural Detection

> This section covers the entire dual-model attention detection system — the core of StudyBuddy's intelligence. This module was built from scratch, including dataset preprocessing, model architecture design, training, and integration with the app.

The system analyses a short webcam clip (10 frames, ~10 seconds) every time attention needs to be assessed. Two models run in sequence on this clip, and their outputs are fused into a single normalised attentiveness score between 0 and 10.

---

### Dataset: DAiSEE

The behavioural model was trained on the **DAiSEE (Dataset for Affective States in E-Environments)** dataset — one of the few datasets specifically designed for student engagement in naturalistic e-learning settings.

| Property | Detail |
|---|---|
| Total clips | 8,924 video clips |
| Participants | 112 students (ages 18–30) |
| Clip duration | 10 seconds each |
| Recording setup | Uncontrolled, personal webcams |
| Annotation | Human expert evaluation |

Each clip is annotated across four affective dimensions on a 0–3 intensity scale:

| Dimension | 0 | 1 | 2 | 3 |
|---|---|---|---|---|
| Boredom | Alert, interested | Mildly bored | Moderately bored | Very bored |
| Engagement | Not engaged | Slightly | Moderately | Highly engaged |
| Confusion | Clear understanding | Minor | Moderate | Severe |
| Frustration | None | Slight | Moderate | High |

**Frame extraction pipeline:**  
Each video clip was processed with FFmpeg at 1 FPS → 10 frames per clip → 89,240 total training frames.

**Dataset split:**

| Split | Videos | Frames | Usage |
|---|---|---|---|
| Train | 5,481 | 54,810 | Model training |
| Validation | 1,720 | 17,200 | Hyperparameter tuning |
| Test | 1,723 | 17,230 | Final evaluation |

---

### Emotion Detection Module

The first model in the pipeline detects the student's dominant facial emotion across the 10-frame clip.

**Architecture:**
- Base: CNN pre-trained on FER-2013 (7-class facial expression dataset)
- Input: 48×48 grayscale images
- Face detection: OpenCV Haar Cascade (`haarcascade_frontalface_default.xml`)
  - Scale factor: 1.3, Min neighbours: 5, Min face size: 30×30 px

**Why 4 emotions instead of 7:**  
FER-2013 covers 7 emotions (angry, disgust, fear, happy, sad, surprise, neutral). Disgust, fear, and surprise rarely appear in real study sessions and don't have a meaningful correlation with attention quality. Mapping them to "neutral" reduced noise in the downstream scoring without losing meaningful signal.

**Temporal aggregation:**  
Rather than classifying a single frame (which is noisy), emotion detection runs on all 10 frames. The dominant emotion is determined by majority vote, and confidence is averaged across frames. This significantly reduces single-frame misclassification errors.

**Emotion modifier table** (applied to behavioural scores):

| Emotion | Engagement Δ | Boredom Δ | Confusion Δ | Frustration Δ |
|---|---|---|---|---|
| Happy | +1.0 | −1.0 | 0.0 | −0.5 |
| Sad | −1.0 | +1.0 | +0.5 | +0.5 |
| Neutral | 0.0 | 0.0 | 0.0 | 0.0 |
| Anger | −1.0 | +1.0 | +0.5 | +0.5 |

---

### Behavioural Analysis Module

The second model analyses temporal behaviour patterns across the 10-frame sequence to predict affective state intensities.

#### Architecture: ResNet-50 + BiLSTM + Attention

**Stage 1 — Spatial feature extraction (ResNet-50):**
- Pre-trained on ImageNet1K
- Input: 224×224×3 RGB frames
- Output: 2048-dimensional feature vector per frame
- Transfer learning: First 80% of layers frozen (generic features), last 20% fine-tuned (task-specific)

**Stage 2 — Temporal modelling (Bidirectional LSTM):**
- Input: Sequence of 10 × 2048-dim feature vectors
- Hidden units: 512 per direction (1024 total)
- 2 stacked BiLSTM layers with 0.3 dropout between them
- Bidirectional processing is critical here — attention lapses often have temporal signatures (gradual disengagement or sudden drops) that require both past and future context to identify accurately

**Stage 3 — Attention mechanism:**
- Linear(1024 → 256) + Tanh → Linear(256 → 1) → Softmax across time
- Learns which frames are most discriminative (e.g., student looking away, yawning)
- Downweights uninformative frames (steady reading, normal posture)
- This also makes predictions interpretable — attention weights can show *why* a session was classified as distracted

**Stage 4 — Classification head:**
- Linear(1024 → 512) → ReLU → Dropout(0.4)
- Linear(512 → 256) → ReLU → Dropout(0.3)
- Linear(256 → 4) — output clamped to [0, 3] at inference

**Loss function — Weighted MSE:**

```
weights = [Boredom: 1.5, Engagement: 2.0, Confusion: 0.9, Frustration: 0.9]
```

Engagement gets the highest weight because it's the most reliable predictor of attention. Boredom is second-highest. Confusion and frustration are contextual indicators and are weighted lower to avoid over-penalising ambiguous states.

**Training configuration:**
- Optimizer: Adam (lr=1e-4, weight decay=1e-5)
- Epochs: 100 with early stopping (patience=10)
- Batch size: 16 sequences
- Training time: ~15 hours

---

### Integration & Attentiveness Scoring

After both models run, their outputs are fused:

**Step 1 — Apply emotion modifiers:**
```
Eng_adj  = Engagement  + Δ_emotion(Engagement)
Bor_adj  = Boredom     + Δ_emotion(Boredom)
Conf_adj = Confusion   + Δ_emotion(Confusion)
Frust_adj = Frustration + Δ_emotion(Frustration)
```

**Step 2 — Weighted raw score:**
```
Raw_Score = 5·Eng_adj + 3·(3 - Bor_adj) + 1·(3 - Conf_adj) + 1·(3 - Frust_adj)
```

Engagement is weighted 5× because it is the most direct indicator of productive attention. Boredom is inverse-weighted (3 - Bor_adj) because higher boredom means less attention.

**Step 3 — Sigmoid normalisation to 0–10:**
```
Centered  = Raw_Score - 19.25
Sigmoid   = 1 / (1 + e^(-Centered))
Score     = clip(Sigmoid × 10, 0, 10)
```

The sigmoid is applied instead of linear normalisation because it naturally compresses extreme values and amplifies mid-range differences — which is exactly where most students sit during real study sessions.

**Step 4 — Binary classification:**
```
Score ≥ 7.0 → Attentive
Score  < 7.0 → Distracted
```

The threshold was raised from 6.0 to 7.0 after observing that the sigmoid transformation pushed score distributions higher, and 6.0 was producing too many false "Attentive" labels.

---

### Attentiveness Score Pipeline

```
Webcam clip (10 frames)
        │
        ├──► Emotion CNN ──► Dominant emotion + modifier table
        │
        └──► ResNet-50 ──► BiLSTM ──► Attention ──► [Boredom, Engagement, Confusion, Frustration]
                                                              │
                                            Apply emotion modifiers
                                                              │
                                            Weighted raw score formula
                                                              │
                                            Sigmoid → normalised score (0–10)
                                                              │
                                     Threshold → Attentive / Distracted
```

---

### Model Evolution & Training Challenges

Building this system involved several iterations, each with distinct problems that had to be diagnosed and solved.

#### Challenge 1: Score collapse in early models

The first two models (ResNet50+TCN and ResNet50+BiLSTM) predicted all four affective dimensions independently. After training, the models would consistently output scores hovering around ~6/10 regardless of the actual input. The dashboard was essentially useless — everyone looked equally "semi-attentive".

**Root cause:** The DAiSEE dataset is heavily imbalanced. Most clips feature students who are mildly engaged (mid-range on all dimensions). The model learned to predict the mean of the distribution and was rewarded for it by the loss function.

**What was tried:**
- Adjusting loss weights (higher weight on engagement and boredom)
- Increasing training epochs — no improvement
- Trying two different architectures (TCN vs BiLSTM) — both suffered from the same collapse

#### Challenge 2: Constant-score predictions

The third model iteration moved to a single attentiveness score output (0–10) instead of four separate dimensions. This was architecturally inspired by a 2024 paper on student engagement detection using EfficientNetV2-L with RNN models. The intent was to give the model a clearer objective.

However, this model also collapsed — it would predict ~6.5 for virtually every input.

**What fixed it:**
1. **Ordinal-aware Huber loss** — standard MSE treated all prediction errors equally. Huber loss penalises wrong-direction predictions more aggressively, pushing the model to at least get the relative ordering of scores correct.
2. **Binary cross-entropy head** — a second output head was added to predict Attentive vs Distracted directly. The dual-task loss forced the model to learn discriminative features rather than just predicting the mean.
3. **Weighted sampling / oversampling** — distracted samples were upsampled in the training batches so the model saw enough examples of low-attention states to learn them properly.

These three changes together eliminated the constant-score problem.

#### Challenge 3: Linear normalisation was too coarse

Even after fixing score collapse, the distribution of scores was clustered in the 5–7 range. Small changes in the student's behaviour produced barely noticeable changes in the score.

**Fix:** Replaced linear normalisation with sigmoid-based normalisation. The sigmoid function is steep in the mid-range and flat at the extremes, so small changes in raw scores around the decision boundary produce large changes in the normalised output — exactly the sensitivity needed to reliably distinguish attentive from distracted.

#### Model comparison:

| Model | Architecture | Test Accuracy | MSE | MAE | Training Time |
|---|---|---|---|---|---|
| Model 1 | ResNet50 + TCN | 69.12% | 0.572 | 0.550 | ~15 hours |
| Model 2 | ResNet50 + BiLSTM + Attention | 71.04% | 0.651 | 0.593 | ~15 hours |
| New Model | EfficientNet-B0 + Temporal Self-Attention | 61.0% | 0.282 | 0.430 | ~8.4 hours |

Model 2 (ResNet50 + BiLSTM + Attention) was selected for production. While the newer EfficientNet model had lower MSE and MAE, its overall accuracy was lower — and for a binary Attentive/Distracted classification task, accuracy is the metric that matters most to the end user experience.

---

## Dashboard Analytics

The analytics module visualises productivity and attentiveness data computed locally from session history.

**Metrics computed:**

| Metric | Formula |
|---|---|
| Total Focus Time | Σ (session_seconds / 60) |
| Completion Rate | (Completed Focus Sessions / Total Sessions) × 100 |
| Average Attention | (1/n) × Σ score_i |
| Focus Ratio | Focus Sessions / (Focus + Break Sessions) |
| Study Streak | Consecutive calendar days with at least one completed focus session |

The study streak is computed by normalising each session timestamp to midnight, sorting unique study days, and detecting consecutive runs. Visualisations include focus trend line charts, session duration bar charts, focus-vs-break doughnut charts, and an attention trend time-series for the last 10 snapshots. A print report feature allows exporting productivity insights as a formatted report.

---

## RAG Integration

Before a quiz starts, the system captures a 10-second webcam clip and runs the full attention detection pipeline. The resulting score determines quiz difficulty:

| Score | Difficulty |
|---|---|
| Low (< 4) | Easy |
| Medium (4–7) | Medium |
| High (> 7) | Hard |

Users can also bypass this and select difficulty manually. The RAG chatbot uses ChromaDB for semantic retrieval over uploaded PDFs, ensuring quiz questions and answers are grounded in the student's actual study material.

---

## Deployment

The app is packaged as a single installable desktop executable:

- Python backend → PyInstaller standalone executable (no Python required on target machine)
- Backend bundled inside Electron app → Electron Builder
- All ML models, libraries, and the local database are included in the bundle

**Design rationale for on-device execution:**  
Cloud-based inference (e.g., Render, AWS Lambda) would introduce latency that makes real-time attentiveness feedback impossible. Continuous video stream processing over a network is also bandwidth-intensive and raises serious privacy concerns. Local execution keeps all student data on the device and delivers sub-second feedback.

---

## Limitations

- **Storage footprint:** Bundling ML models and the Python runtime locally results in a large install size.
- **First-launch latency:** The backend executable takes several seconds to initialise on first run.
- **Webcam dependency:** The entire attention detection pipeline requires a functioning webcam. The app degrades gracefully (manual quiz mode) but the core monitoring feature is unavailable without one.
- **Dataset bias:** DAiSEE was recorded with students aged 18–30 in Western university settings. Performance may vary across different demographics or lighting conditions.

---

## References

- DAiSEE Dataset: Gupta et al., "DAiSEE: Towards User Engagement Recognition in the Wild", arXiv 2016
- Behavioural model architecture inspired by: "Detection of Student Engagement in E-Learning Environments Using EfficientNetV2-L Together with RNN-Based Models", Journal of Artificial Intelligence, April 2024
- FER-2013 Dataset: Goodfellow et al., ICML 2013

---
