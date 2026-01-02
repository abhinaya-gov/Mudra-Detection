
<p align="center">
  <a href="#why-this-model-architecture">
    <img src="https://img.shields.io/badge/Architecture-View-blue?style=for-the-badge" />
  </a>
  <a href="#dataset">
    <img src="https://img.shields.io/badge/How%20It%20Works-Explore-green?style=for-the-badge" />
  </a>
  <a href="#future-enhancements">
    <img src="https://img.shields.io/badge/Future%20Plans-Roadmap-orange?style=for-the-badge" />
  </a>
</p>



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

## Why This Project?

This project was built to explore a more complex, real-world computer vision pipeline that goes beyond simple image classification.

Unlike the hand/no-hand classifier, this system combines **object detection + classification + explainability** into a single end-to-end pipeline for recognizing Bharatanatyam mudras in real time.

The main goals of this project were:

- To design and implement a **multi-stage vision system** (detection → classification → interpretation).
- To gain practical experience with **YOLO-style object detection** and region-based inference.
- To work with a culturally meaningful dataset (Indian classical dance mudras) rather than generic benchmarks.
- To understand how detection and classification models interact in a production-like pipeline.
- To explore model explainability using tools like **Grad-CAM**.

This project also serves as a foundation for:
- Sign and gesture recognition systems
- Human-computer interaction
- Assistive technologies
- Cultural heritage digitization

Overall, this was built as both a **technical learning project** and a **meaningful application of deep learning to real-world visual understanding**.

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
  
This project uses a combination of real and synthetic hand image datasets to train and evaluate a **binary hand detection model (hand vs no-hand)**.

### Hand-Bo3ot (Roboflow Universe)
A general-purpose hand detection dataset with bounding box annotations across different poses, lighting conditions, and backgrounds. Used as the primary dataset for training the detector.  
https://universe.roboflow.com/yolov4tiny-wzb2k/hand-bo3ot

### Bharatanatyam Mudras (Roboflow Universe)
Contains annotated images of classical Indian hand gestures (mudras). Used to add diversity in hand shapes, orientations, and fine-grained poses.  
https://universe.roboflow.com/mudras-avdrb/bharatanatyam-mudras-fg9qo-gcruc

### Hand Gesture Dataset (Roboflow Universe)
A collection of hand gesture images used for additional data exploration and optional augmentation.  
https://universe.roboflow.com/horyzn-qhfq4/hand-gesture-gizg2

### Hand Detection Dataset — VOC/YOLO Format (Kaggle)
A ready-to-use dataset in VOC/YOLO format for standard object detection training and benchmarking.  
https://www.kaggle.com/datasets/nomihsa965/hand-detection-dataset-vocyolo-format/data

### Synthetic Hand Detection Dataset (Kaggle)
A synthetic dataset used to improve generalization and robustness, especially for rare poses and edge cases.  
https://www.kaggle.com/datasets/zeyadkhalid/hand-detection

These datasets together provide a mix of real-world variability and synthetic augmentation, helping the model generalize better across different environments, poses, and lighting conditions.


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

## Why This Model Architecture?

This project uses a two-stage architecture:

1. A **YOLO-based object detector** to locate hands in the image.
2. A **custom CNN classifier** to recognize the specific mudra from the detected hand region.

This separation was chosen intentionally:

- Detection and classification solve different problems and benefit from specialized architectures.
- YOLO is optimized for fast and accurate localization.
- A custom CNN allows fine-grained control over gesture recognition features.

### Architectural Design Choices

- **YOLO for detection**  
  Provides real-time performance and robust bounding box predictions across varying backgrounds.

- **Custom CNN for classification**  
  Enables targeted learning of subtle finger and hand shape variations between mudras.

- **Two-stage pipeline**  
  Improves robustness by isolating the gesture classification task from background noise.

- **Grad-CAM integration**  
  Allows visualization of which regions influence the model’s predictions, improving interpretability and debugging.

### Why Not an End-to-End Single Model?

An end-to-end model could jointly learn detection and classification, but:

- It requires significantly more data.
- It is harder to debug and interpret.
- It is more complex to train and tune.

The two-stage approach provides better modularity, interpretability, and flexibility for experimentation and improvement.

Overall, this architecture balances **performance, modularity, interpretability, and learning value**, making it well-suited for gesture recognition tasks.

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

## Future Enhancements

Planned or possible improvements for this project include:

- Expand the mudra dataset to include more dancers, lighting conditions, and camera angles.
- Add **temporal modeling** (LSTM / 3D CNN / Transformers) to capture motion dynamics in gestures.
- Add multi-hand interaction support.
- Improve detection accuracy for occluded or partially visible hands.
- Deploy the pipeline as a web app or mobile app.
- Add multilingual explanations of mudras and their meanings.
- Add continuous learning from user feedback.

These enhancements aim to evolve the project from a research and learning system into a more robust, scalable, and user-facing application.


---

## Author

**Abhinaya**

Built as a learning-focused project to understand machine learning systems end-to-end, from data preparation and training to real-time inference.

---

## License

This project is intended for educational and research purposes.

---
