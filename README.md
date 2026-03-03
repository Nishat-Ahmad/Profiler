# Face Profiler (Multi-Task Learning)

A Deep Learning project for analyzing facial attributes from images. This model simultaneously predicts **Age**, **Gender**, and **Race** using a single convolutional neural network based on the VGG16 architecture.

## Features

*   **Multi-Task Learning (MTL):** Solves three distinct tasks with a shared feature extractor.
    *   **Age:** Regression (Continuous value prediction).
    *   **Gender:** Binary Classification (Male/Female).
    *   **Race:** Multi-Class Classification (7 distinct categories).
*   **Backbone:** Utilizes a pre-trained **VGG16** (frozen layers) for robust feature extraction.
*   **Web Interface:** Includes a FastAPI-based web application for easy interaction and demonstration.
*   **Training Pipeline:** comprehensive scripts for training, validation, and visualizing loss metrics.

## Project Structure

```
Profiler/
├── app/                  # Web Application (FastAPI)
│   ├── main.py           # Main application entry point
│   ├── static/           # CSS and static assets
│   └── templates/        # HTML templates
├── FairFace/             # Dataset directory (Make sure to place data here)
├── scripts/              # Core logic and utilities
│   ├── model.py          # Model architecture definition (FairFaceVGG16)
│   ├── train.py          # Training loop with validation and logging
│   ├── dataset.py        # Custom Dataset class for loading images/labels
│   ├── inference.py      # CLI script for running predictions on single images
│   ├── check_kernels.py  # Utility to inspect model layers
│   └── compare_summary.py # Utility to compare model parameters
├── weights/              # Directory to store trained model weights (.pth)
└── requirements.txt      # Python dependencies
```

## Installation

1.  **Clone the repository:**
    ```bash
    git clone <repository_url>
    cd Profiler
    ```

2.  **Install Dependencies:**
    It is recommended to use a virtual environment.
    ```bash
    pip install -r requirements.txt
    ```

## Dataset

This project is designed to work with the **FairFace** dataset.
1.  Download the dataset.
2.  Place the images in `FairFace/train/` and `FairFace/val/`.
3.  Ensure the label CSV files (`fairface_label_train.csv`, `fairface_label_val.csv`) are in the `FairFace/` root folder.

## Usage

### 1. Training the Model
To train the model from scratch (or continue training):

```bash
python scripts/train.py --epochs 10 --batch_size 32
```
*   **Outputs:** Saves the best model weights to `fairface_multi_task_model.pth` and generates a loss plot `training_loss_plot.png`.

### 2. Running Inference (CLI)
To analyze a single image from the command line:

```bash
python scripts/inference.py --image path/to/your_image.jpg
```

### 3. Launching the Web App
To start the interactive web interface:

```bash
uvicorn app.main:app --reload
```
*   Open your browser and navigate to `http://127.0.0.1:8000`.
*   Upload an image to see the predicted Age, Gender, and Race.

## Model Architecture

The model uses a **Shared Trunk -> Multi-Head** design:

1.  **Feature Extractor:** VGG16 (ImageNet weights, frozen).
2.  **Bottleneck:** Adaptive Average Pooling reduces features to a compact vector.
3.  **Shared Layers:** Dense layers to learn common high-level facial features.
4.  **Task Heads:**
    *   **Gender:** Sigmoid activation (Binary Cross Entropy Loss).
    *   **Age:** Linear activation (L1 Loss / MAE).
    *   **Race:** Linear activation (Cross Entropy Loss).

## License

This project is for educational and research purposes.