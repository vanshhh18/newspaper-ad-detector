Demo: https://drive.google.com/file/d/1mE3uxHD0wBOjXQOicjscgbKSB3zCgoEa/view?usp=sharing

---

# Newspaper Advertisement Detector & Extractor

An AI-powered application that detects advertisements in newspaper pages, identifies their boundaries, crops individual advertisements, and allows users to download the extracted images.

Built using **YOLO11s, Python, OpenCV, Pillow, and Streamlit**.

## Features

- Upload high-resolution newspaper pages in JPG, JPEG, or PNG format.
- Detect multiple advertisements on a single page.
- Identify advertisement boundaries using object detection.
- Display detected advertisements with bounding boxes and confidence scores.
- Automatically crop each detected advertisement separately.
- Preview individual advertisement crops.
- Download individual crops or all extracted advertisements as a ZIP file.
- Adjust the detection confidence threshold.

## Technology Stack

| Component               | Technology                   |
| ----------------------- | ---------------------------- |
| Programming language    | Python                       |
| Object detection        | YOLO11s                      |
| Deep learning framework | PyTorch                      |
| Model training          | Ultralytics                  |
| Image processing        | Pillow, OpenCV               |
| Data handling           | Pandas, NumPy                |
| User interface          | Streamlit                    |
| Training environment    | Kaggle GPU (NVIDIA Tesla T4) |

## Project Structure

```text
newspaper_ad_detector/
│
├── app.py
├── best.pt
├── requirements.txt
├── README.md
├── .gitignore
│
└── sample_outputs/
    ├── Paper1/
        ├── Full_newspaper
        ├── Cropped_ad
    ├── Paper2/
```

`best.pt` contains the trained YOLO11s model weights. It is kept separate from the source repository because of its binary size.

## Model Selection

The project uses **YOLO11s (You Only Look Once)** for advertisement detection.

YOLO was selected because:

- It performs object detection and predicts bounding-box coordinates in a single inference process.
- It supports multiple object instances in the same image.
- It provides confidence scores for individual detections.
- It offers a practical balance between detection performance and inference speed.

The model was fine-tuned from pretrained YOLO11s weights for a single detection class:

`advertisement`

## Dataset Preparation

The training dataset was obtained from the Hugging Face dataset:

[Detect-AD Dataset](https://huggingface.co/datasets/t-Shr/Detect-AD)

The original dataset contained:

- 88 newspaper pages.
- 396 annotated advertisement bounding boxes.
- Training, validation, and test splits.
- Pages without advertisements, retained as background examples.

### Small Advertisement Handling

Many newspaper advertisements, especially classified advertisements, occupy very small regions of a full newspaper page.

To improve their detection, additional training patches were generated from the original training pages.

Two patch-generation strategies were used:

1. Grid-based patch generation.
2. Advertisement-centred patch generation.

The resulting augmented training dataset contained:

| Data                    |  Images | Annotations |
| ----------------------- | ------: | ----------: |
| Original training pages |      42 |         208 |
| Grid patches            |      62 |          99 |
| Centred patches         |      56 |          81 |
| **Total**               | **160** |     **388** |

The validation and test sets remained unchanged.

This approach increased the number of training examples containing small advertisements while retaining full-page context.

## Model Training

The model was trained using Kaggle with an NVIDIA Tesla T4 GPU.

Key training configuration:

| Parameter               | Value      |
| ----------------------- | ---------- |
| Model                   | YOLO11s    |
| Pretrained weights      | yolo11s.pt |
| Epochs                  | 120        |
| Image size              | 1280       |
| Batch size              | 2          |
| Optimizer               | AdamW      |
| Initial learning rate   | 0.0005     |
| Weight decay            | 0.0005     |
| Early stopping patience | 30         |
| Random seed             | 42         |

Training used controlled augmentation, including small rotations, translations, scaling, colour augmentation, and limited mosaic augmentation.

The best-performing checkpoint was saved as `best.pt`.

## Evaluation Results

The augmented model was evaluated on the original test split.

| Metric     | Result |
| ---------- | -----: |
| Precision  | 81.69% |
| Recall     | 76.27% |
| mAP\@50    | 85.08% |
| mAP\@50–95 | 75.71% |

These results indicate that patch-based training improved advertisement detection, particularly recall, compared with the earlier model trained only on full-page images.

**Evaluation note:** The test split was used during iterative model development and should be treated as a diagnostic evaluation rather than a completely untouched benchmark.

## Installation

### 1. Clone the repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd newspaper_ad_detector
```

### 2. Create a virtual environment

Windows PowerShell:

```powershell
python -m venv venv
venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Add model weights

Download the trained `best.pt` file and place it in the project root, alongside `app.py`.

The application expects:

```text
newspaper_ad_detector/
├── app.py
└── best.pt
```

### 5. Run the application

```bash
streamlit run app.py
```

Open the local URL displayed in the terminal, usually:

```text
http://localhost:8501
```

## How to Use

1. Open the Streamlit application.
2. Upload a newspaper page in JPG, JPEG, or PNG format.
3. Adjust the detection confidence threshold if necessary.
4. Click **Detect Advertisements**.
5. View the original newspaper page with detected bounding boxes.
6. Inspect the individual cropped advertisements.
7. Download individual crops or download all extracted advertisements as a ZIP file.

### Detection Confidence

The confidence slider controls the minimum confidence required for a detection to be retained.

- Lower thresholds may identify more advertisements but can introduce false positives.
- Higher thresholds may reduce false positives but can miss genuine advertisements.

The default threshold is 0.25.

## Output

For every detected advertisement, the application generates an individual image crop.

The application also supports downloading all extracted advertisements together in a ZIP archive.

For the assignment's BMW reference task, the expected output is a separate high-resolution PNG containing the complete BMW advertisement without surrounding newspaper content.

## Limitations

- Detection performance depends on the quality, resolution, layout, and appearance of the newspaper page.
- Very small advertisements and densely packed classified sections can be challenging.
- The model may occasionally miss advertisements or detect unrelated content.
- Detection confidence does not guarantee that a bounding box captures every advertisement perfectly.
- Performance may vary across newspaper publications and layouts not represented in the training dataset.

Further improvements could include expanding the training dataset, adding more newspaper layouts, and improving high-resolution inference for densely packed advertisements.

## Future Improvements

- Expand training data across different newspaper publications and languages.
- Improve detection of very small classified advertisements.
- Explore tiled inference for extremely high-resolution pages.
- Improve boundary refinement to avoid cutting off advertisement content.
- Evaluate on additional unseen newspaper pages.

## Author

Developed as part of the **AI/ML Engineer Intern Assignment**.

**Project:** Automated Newspaper Advertisement Detection & Cropping

## Copyright

Copyright (c) 2026 Vansh Mittal.

All rights reserved. This project and its original
source code may not be copied, redistributed,
or commercially used without written permission
from the author.
