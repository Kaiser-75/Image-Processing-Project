## Project 4 
## Title: Automated Playing Card Recognition using Template-Based Computer Vision

## Algorithm Overview:
The algorithm detects a playing card by locating its contour against a plain background and warping it to a fixed upright view. It then crops the top-left corner, where the rank and suit symbols appear. The cropped region is binarized using an adaptive threshold and split into two parts, one for rank and one for suit. Each part is cleaned, resized to a standard size, and compared to pre-made templates using normalized cross-correlation. The template with the highest similarity is selected as the card’s rank or suit. This process makes the system invariant to rotation, scale, and lighting.

## Assumptions:
1) Only one main card to process
2) Good lighting, low motion blur
3) Strong contrast between card and background
4) Camera not extremely tilted
5) The card covers about 60% of the frame.

## Directory Overview
```
Project 4/
│
├── main.py
│   ├─ Launches the GUI with Tkinter and OpenCV.
│   ├─ Provides two modes:
│   │   • Webcam mode – real-time detection.
│   │   • Image mode – detect from a selected file.
│   └─ Handles visualization, keyboard input, and user interaction.
│
├── vision/
│   ├── __init__.py
│   │   • Initializes the vision module and exposes main classes.
│   │
│   ├── detector.py
│   │   • Handles card detection, warping, and orientation correction.
│   │   • Steps:
│   │       1. Preprocess image (grayscale + Otsu threshold).
│   │       2. Find card contour.
│   │       3. Warp to 200×300 canonical view.
│   │       4. Normalize upright orientation.
│   │       5. Extract and binarize rank & suit patches.
│   │   • Outputs a structured object `CardCrops` with all extracted regions.
│   │
│   ├── matcher.py
│       • Loads pre-saved binary templates for ranks and suits.
│       • Compares extracted patches with templates using:
│           – Normalized Cross-Correlation (appearance similarity)
│           – Distance-Field Similarity (shape alignment)
│       • Produces a `MatchResult` with predicted rank, suit, and confidence scores.
│
├── assets/
│   ├── templates/
│   │   ├── ranks/       → Binary rank templates (A–K)
│   │   └── suits/       → Binary suit templates (Hearts, Spades, Clubs, Diamonds)
│   └── (optional demo images or sample frames)
│
└── README.md
    • Project overview, installation, and usage instructions.

```

## Document link: [PDF File](https://drive.google.com/file/d/17xpe8O7JdY2evqKMsyXgTrJEmiOorkPA/view?usp=drive_link)
## N.B: Better version of this will be uploaded soon.


