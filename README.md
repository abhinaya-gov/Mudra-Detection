# Mudra Detection Pipeline

An end-to-end computer vision system for **hand mudra detection**, built using a **two-stage pipeline**:
hand localization using YOLO, followed by mudra classification using a custom CNN trained from scratch.

This project focuses on **pipeline design, dataset handling, and model integration**, rather than treating models as black boxes.

---

## Problem Overview

Mudra recognition is a fine-grained vision task:

* Hands occupy a small region of the image
* Mudras differ by subtle finger configurations
* Lighting, background, and scale vary significantly

A single end-to-end classifier struggles with these constraints.
To address this, the problem is decomposed into **detection → classification**, improving robustness and interpretability.

---

## Pipeline Design

The system is structured as a modular pipeline:

Input (Image / Webcam)
↓
YOLO Hand Detector
↓
Hand Cropping
↓
Mudra Classifier (CNN)
↓
Mudra Label + Confidence

### Why a pipeline?

* Separates localization from classification
* Allows independent training and testing of each stage
* Mirrors real-world production vision systems
* Makes debugging and experimentation easier

---

## Project Structure

mudra-detection/ <br>
├── hand_detection/ <br>
│   ├── model.ipynb              – YOLO hand detection logic <br>
│   ├── infer_image.ipynb        – Hand detection on images <br>
│   └── infer_webcam.ipynb       – Real-time hand detection <br>
│<br>
├── mudra_classification/<br>
│   ├── model.py              – Custom CNN architecture <br>
│   ├── infer_image.py        – Mudra classification on cropped hands <br>
│   └── infer_webcam.py       – Standalone live classification <br>
│<br>
|── run_pipeline.py       – End-to-end detection + classification <br>
│<br>
└── README.md

---

## Dataset

### Hand Detection

* Trained using **multiple hand datasets** merged into a single YOLO training setup
* Required resolving:

  * Different class definitions
  * Different directory layouts
  * Dataset imbalance
* A unified YOLO configuration was generated programmatically

### Mudra Classification

* Custom mudra image dataset (had to collect some on my own)
* Took only 10 of the most common mudras for simplicity and clarity
* Each class corresponds to a specific mudra
* Key challenges:

  * High inter-class similarity
  * Sensitivity to orientation and lighting
  * Limited variation per class

Datasets are **not included** due to size and licensing constraints.

Expected dataset structure:

datasets/ <br>
└── mudras/ <br>
    ├── train/<br>
    │   ├── Asamyukta_Hastas/ <br>
    │   ├── Samyukta_Hastas/ <br>
    │   └── ... <br>

    (For now I have only used 10 mudras from Asamyukta Hastas(single-hand gestures))

---

## Training Details

### Hand Detection

* Model: YOLO
* Training:

  * Trained yolo8 **from scratch** (no pretrained weights)
  * Multi-dataset setup
  * Data augmentation for scale, color, and spatial variance
* Output:

  * Bounding boxes used for downstream classification

### Mudra Classification

* Model: Custom CNN (implemented from scratch)
* Architecture:

  * Stacked convolutional blocks with BatchNorm and ReLU
  * Progressive spatial downsampling
  * Fully connected classification head
* Training:

  * Random rotations, flips, and color jitter
  * Fixed train/validation split for reproducibility
* Output:

  * Mudra class probabilities via softmax

---

## Running the Project

This project cannot be run directly on GitHub.
Clone the repository and run locally.

### Clone

git clone [https://github.com/abhinaya-gov/mudra-detection.git](https://github.com/abhinaya-gov/mudra-detection.git)
cd mudra-detection

### Install dependencies

torch
torchvision
ultralytics
opencv-python
numpy
pillow
matplotlib
seaborn
pandas
os
yaml

### Run full pipeline (recommended)

python full_pipeline.ipynb

This launches a real-time demo that:

* Detects hands via YOLO
* Crops detected regions
* Classifies mudras
* Displays labels and confidence scores live

---

## Running Components Independently

### Hand detection only

python hand_detection/infer_webcam.ipynb

### Mudra classification only

python mudra_classification/infer_image.ipynb --image path/to/cropped_hand.jpg

Each component is independently testable for easier debugging.

---

## Key Challenges & Trade-offs

* Fine-grained visual differences between mudras
* Error propagation from detection to classification
* Dataset imbalance and limited diversity
* Latency vs accuracy trade-offs in real-time inference
* Similarities between the mudras
* No consistency in the dataset

These constraints directly influenced model and pipeline design choices.

---

## Future Improvements

* Temporal smoothing across video frames
* Multi-hand and simultaneous mudra support
* Improved dataset diversity
* Model optimization for faster inference
* Better detection of hands when smaller

---

## Author

**Abhinaya**

Built as a learning-focused project to understand machine learning systems end-to-end, from data preparation and training to real-time inference.

---

## License

This project is intended for educational and research purposes.

---
