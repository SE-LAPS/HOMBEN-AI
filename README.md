# 🐄 Cow Identity Recognition by Nose-Prints

![Cow Identity Recognition](./_docs/ui.gif)

This project aims to develop a reliable system for identifying cows based on their unique nose-prints. This method can be highly beneficial for livestock management, ensuring accurate identification without the need for invasive or stressful methods.

## ✨ Key Features

- **Image Preprocessing**: Scripts for resizing, grayscale conversion, adaptive thresholding, and morphological operations to highlight nose patterns.
- **Annotation Tools**: Tools for manually annotating images to mark nose areas, creating a dataset for training.
- **Rescaling**: Automated scripts to rescale images and annotations to multiple sizes for robust training.
- **Negative Sample Preparation**: Guides for preparing and resizing negative samples to improve the model's ability to distinguish noses from other parts of the cow or background.
- **Cascade Classifier Training**: Steps to train a Haar cascade classifier using positive and negative samples, tailored to detect cow noses.
- **Nose Extraction**: Tools to use the trained classifier to extract nose regions from images.
- **Nose-Print Processing**: Scripts to process extracted noses into high-contrast prints suitable for recognition.
- **Identifier Training**: Training an identifier model using KNN or CNN algorithms to recognize individual cows from their nose prints.
- **GUI Application**: A user-friendly graphical interface for testing the system by dragging and dropping test images to see the identification results.

## 🚀 Current Status

This project is currently in the research and prototype stage. It is not yet intended for industrial use. The current focus is on developing and refining the algorithms and methods to ensure accurate and reliable cow identification. Contributions and feedback are welcome to help improve the system.

## 🧑‍🏫 Cascade Classifier Training for Cow Noses

We trained a Haar cascade classifier to identify the nose print from a clear cow face, following stages 1-8. The classifier was trained using 50 cow face images as positive samples and 1500 non-nose images as negative samples. The trained classifier, saved as `cascade.xml`, can be used to detect the nose area in cow face images accurately.

![Cascade Classifier Training](./_docs/cownose_cascade_training.jpg)
![Cascade Classifier Test](./_docs/cow_node_detection_cascade_test.jpg)

[Download cascade.xml](https://your-download-link-here)

## 🐮 Nose Extraction

Using the trained cascade classifier (`cascade.xml`), we extract nose regions from cow face images. This involves loading the classifier, detecting the nose region, and cropping the detected nose area. This is implemented in the `script_extract/nose_extract.py` script.

![Extracted Cow Noses](./_docs/extracted_cow_noses.png)

## 🧠 Cow Nose Identifier Training

We employ machine learning techniques, specifically KNN and CNN, to recognize individual cows based on their unique nose prints.

![Identifier Training](./_docs/cownose_identifier_training.jpg)

## 🖼️ Nose-Print Processing

After extracting the nose regions, we process these images to create high-contrast nose prints suitable for recognition. This involves grayscale conversion, adaptive thresholding, and morphological operations, implemented in `script_extract/node_to_print.py`.

![Nose-Print Processing](./_docs/nose_print_processing.png)

## 🛠️ Algorithms

### 🥇 K-Nearest Neighbors (KNN)

KNN is a simple and effective algorithm for classification, especially suited for small datasets.

- **Advantages**:
  - Simplicity
  - Effective for small datasets
  - No training phase needed

### 🥈 Convolutional Neural Networks (CNN)

CNNs are powerful for large datasets and complex image recognition tasks.

- **Advantages**:
  - High accuracy
  - Scalability
  - Robustness

## 📝 Installation

1. **Install Python**:
    - Download and install the latest version from [python.org](https://www.python.org/downloads/).
    - Add Python and the Scripts folder to your environment variable PATH.

    Example for Windows:

    ```plaintext
    C:\Python39;
    C:\Python39\Scripts;
    ```

2. **Install OpenCV**:
    - Download OpenCV [opencv-3.4.11-vc14_vc15.exe](https://sourceforge.net/projects/opencvlibrary/files/3.4.11/opencv-3.4.11-vc14_vc15.exe/download).
    - Run the installer and extract the files.
    - Add the bin directory to your environment variable PATH.

    Example for Windows:

    ```plaintext
    C:\opencv\build\x64\vc15\bin;
    ```

3. **Install Python Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

4. **Run the Project**:
    ```bash
    python script_name.py
    ```

## 📅 Stages

### Stage 1: Positive Image Samples Preparation

Convert all positive images to a resolution of 1024x1024.

```bash
python download_image_resize.py
