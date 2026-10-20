# Breast Cancer Histopathology Classification

A Deep Learning project for binary classification of breast histopathology images into **benign** and **malignant** classes, with a Streamlit web application for interactive predictions.

The project explores different Convolutional Neural Network (CNN) architectures and transfer learning strategies to study their performance on histopathological image classification.

## 🎥 Demo

Watch the application demonstration on LinkedIn:

**[▶️ Watch the project demo](https://fr.linkedin.com/posts/rami-yacoub3_deeplearning-computervision-healthtech-activity-7508618276332027905-kLCT)**

## 📌 Project Overview

Histopathological examination of breast tissue is an important part of breast cancer assessment. This project explores how Deep Learning can be used to classify microscopic tissue images and how different neural network architectures compare on a binary classification task.

Developed as part of a Deep Learning course, the project covers the machine learning workflow, from data preparation and model training to evaluation and deployment through a web interface.

### Objectives

- Build a complete image classification pipeline.
- Compare a CNN trained from scratch with transfer learning approaches.
- Evaluate VGG16 feature extraction, VGG16 fine-tuning, and ResNet50.
- Develop an interactive Streamlit application for image classification.
- Explore the importance of evaluation metrics beyond accuracy, particularly recall and F1-score.

## 🧠 Models Compared

Four Deep Learning approaches are included:

| Model | Approach |
|---|---|
| CNN from Scratch | A custom convolutional neural network trained from scratch |
| VGG16 — Feature Extraction | A pretrained VGG16 backbone with frozen feature layers |
| VGG16 — Fine-Tuning | A pretrained VGG16 model with selected layers unfrozen for further training |
| ResNet50 | A pretrained residual network used for feature extraction |

The comparison explores the trade-offs between learning representations from scratch and leveraging pretrained networks.

## 🔬 Workflow

The project follows five main stages:

1. **Data preparation** — Load and organize the histopathological images and prepare the data for training.
2. **Preprocessing** — Resize images and apply the preprocessing required by each architecture.
3. **Model development** — Train a custom CNN and experiment with pretrained VGG16 and ResNet50.
4. **Evaluation** — Compare model performance using classification metrics and evaluation outputs.
5. **Application development** — Integrate the trained models into a Streamlit interface for interactive predictions.

## 🖥️ Web Application

The Streamlit application provides an interface to:

- Upload a histopathological image.
- Select one of the four available models.
- Run a prediction.
- View the predicted class: benign or malignant.
- Inspect the displayed class probabilities.

The application resizes uploaded images to 96 × 96 pixels before model inference. Preprocessing depends on the selected architecture.

## 📊 Evaluation

The project evaluates the models using classification metrics such as:

- **Accuracy** — Overall proportion of correct predictions.
- **Precision** — Proportion of positive predictions that are correct.
- **Recall** — Proportion of actual positive cases correctly identified.
- **F1-score** — Harmonic mean of precision and recall.
- **ROC-AUC** — Ability to distinguish between the two classes across classification thresholds.

The final comparison should be based on the evaluation results recorded in the training notebooks. No performance figures are reported here until those results have been verified.

## 🛠️ Tech Stack

- **Language:** Python
- **Deep Learning:** TensorFlow / Keras
- **Web application:** Streamlit
- **Image processing:** Pillow, NumPy
- **Development environment:** Google Colab and a local Python environment

## 📁 Repository Structure

```text
breast-cancer-classification/
├── notebooks/              # Training and evaluation notebooks
├── models/                 # Trained model files required by the app
│   ├── first_model.keras
│   ├── VGG_model.keras
│   ├── VGG_model_2.keras
│   └── ResNet50_model.keras
├── app-final.py            # Streamlit application
├── requirements.txt        # Python dependencies
├── .gitignore
└── README.md
```

The `models/` directory must contain the trained model files expected by the application. Large model files may need to be distributed separately if they exceed GitHub's file-size limits.

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/Rami-Yacoub/breast-cancer-classification.git
cd breast-cancer-classification
```

### 2. Create a virtual environment

**Windows:**

```bash
python -m venv .venv
.venv\Scripts\activate
```

**Linux / macOS:**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## ▶️ Run the Application

Ensure that the trained model files are available in the `models/` directory, then run:

```bash
streamlit run app-final.py
```

Streamlit will open the application in your browser. Upload an image, select a model, and click the analysis button to view the prediction.

## ⚠️ Limitations and Responsible Use

This project is intended for educational and experimental purposes only.

It is **not a medical diagnostic tool** and must not be used to make clinical decisions. Predictions may be affected by dataset limitations, image quality, preprocessing choices, and model generalization.

Any potential medical application would require rigorous independent validation, representative clinical data, and assessment by qualified healthcare professionals.

## 👤 Author

**Rami Yacoub**

- GitHub: [@Rami-Yacoub](https://github.com/Rami-Yacoub)
- LinkedIn: [Rami Yacoub](https://fr.linkedin.com/posts/rami-yacoub3_deeplearning-computervision-healthtech-activity-7508618276332027905-kLCT)

---

*Developed as part of a Deep Learning course project.*
