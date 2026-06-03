# DeepFake Detection 

A deep learning image classifier that detects AI-generated or manipulated (deepfake) faces using transfer learning with MobileNetV2.

---

## Overview

This project builds a binary image classifier to distinguish **real** images from **deepfakes**. It uses a pretrained MobileNetV2 backbone (trained on ImageNet) as a frozen feature extractor, then fine-tunes a small classification head on top — making it fast enough to run on CPU.

> **Config:** `IMAGE_SIZE = (64, 64)` · `BATCH_SIZE = 32`

---

## Project Structure

```
DeepFake Detection/
├── train/
│   ├── real/
│   └── fake/
├── test/
│   ├── real/
│   └── fake/
├── deepfake-detection.ipynb      ← main notebook
├── deepfake_detector_model.h5    ← saved trained model
├── app.py                        ← inference / demo script
├── op.jpg                        ← sample output image
├── .gitignore
├── LICENSE
└── README.md
```

---

## How It Works

1. **Data Loading** — Images are loaded from a directory structure at **64×64** resolution with a batch size of **32**, using Keras's `image_dataset_from_directory`.

2. **Preprocessing** — Pixel values are rescaled from `[0, 255]` to `[0, 1]` using a `Rescaling` layer baked into the model graph.

3. **Transfer Learning** — A MobileNetV2 backbone (pretrained on ImageNet, top layers excluded) is used as a frozen feature extractor. This lets the model leverage rich visual features without requiring GPU training.

4. **Classification Head** — On top of the backbone:
   - `GlobalAveragePooling2D` — collapses spatial dimensions
   - `Dense(32, relu)` — lightweight learned transformation
   - `Dense(1, sigmoid)` — binary output (real vs. fake)

5. **Training** — Adam optimizer with `lr=0.001`, binary cross-entropy loss, trained for 5 epochs.

6. **Prediction** — A utility function loads any local image, runs inference, and plots the result with a verdict and confidence score.

---

## Requirements

```
tensorflow>=2.10
numpy
matplotlib
streamlit
pillow
```

Install all dependencies at once with:

```bash
pip install tensorflow numpy matplotlib streamlit pillow
```

---

## Dataset Setup

Organize your dataset into the following folder structure before running:

```
train/
  real/   ← real face images
  fake/   ← deepfake/AI-generated images

test/
  real/
  fake/
```

Update the paths in the script to match your local directory:

```python
train_data = tf.keras.utils.image_dataset_from_directory(
    r"C:\path\to\your\train",
    ...
)
```

---

## Usage

### 1. Train the Model & Explore Data

Open the main notebook to train the pipeline or test single-image logic:

```bash
jupyter notebook deepfake-detection.ipynb
```

At the bottom of the notebook, you can run `predict_local_image("your_image.jpg", model)` for a quick plot-based prediction.

The function will:
- Check if the file exists (with a helpful error if not)
- Resize the image to 64×64
- Run inference
- Display the image with a **REAL** (green) or **FAKE** (red) verdict and confidence score

> **Note:** Windows often hides file extensions. If `"OIP"` fails, try `"OIP.jpg"` or `"OIP.png"`.

### 2. Run the Streamlit Web App

For an interactive browser-based interface, launch the app:

```bash
streamlit run app.py
```

This opens a local web UI where you can upload any image and instantly get a real/fake verdict with a confidence score — no code required.

---

## Model Architecture

```
Input (64, 64, 3)
  └── Rescaling (÷255)
  └── MobileNetV2 [frozen]  ← pretrained ImageNet weights
  └── GlobalAveragePooling2D
  └── Dense(32, relu)
  └── Dense(1, sigmoid)      ← output: P(real)
```

- Score ≥ 0.5 → **REAL** with confidence `score × 100%`
- Score < 0.5 → **FAKE** with confidence `(1 - score) × 100%`

---

## Performance Notes

- MobileNetV2 is frozen during training — only the classification head learns. This keeps training fast on CPU.
- For higher accuracy, consider unfreezing the top layers of MobileNetV2 for fine-tuning after initial training.
- Training for more epochs or using a larger image size (e.g., 224×224) may improve results.

---

## Potential Improvements

| Idea | Description |
|------|-------------|
| Fine-tuning | Unfreeze top MobileNetV2 layers after initial training |
| Data augmentation | Add flips, rotations, brightness shifts to reduce overfitting |
| Larger backbone | Try EfficientNetB0 or ResNet50 for better feature extraction |
| Load saved model | Use `tf.keras.models.load_model('deepfake_detector_model.h5')` to skip retraining |
| Larger input | 224×224 matches MobileNetV2's native input size |

---

## License

MIT License. Free to use and modify for personal and educational projects.
