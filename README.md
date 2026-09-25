# 🍓 Freshify — AI-Based Multi-Fruit Freshness Classification

An AI-based web application that uses a **Multilayer Perceptron (MLP) neural network** to classify fruit images as **Fresh or Rotten**. The application is developed using Python, TensorFlow, and Streamlit, allowing users to upload an image and receive a freshness prediction.

## 📌 Project Overview

Freshify is an image-based classification application developed as part of the **M.Tech Artificial Intelligence and Machine Learning LCA1 project**.

The project aims to explore how neural networks can be used for image classification. The trained MLP model processes fruit images and predicts whether the fruit is fresh or rotten.

The application supports multiple fruit categories, including:

* Apple
* Banana
* Grape
* Guava
* Jujube
* Orange
* Pomegranate
* Strawberry

> **Note:** This application is an educational AI project. Its predictions are experimental and should not be treated as a reliable food-safety assessment.

---

## ✨ Features

* 🍎 Multi-fruit freshness classification
* 🧠 Neural network-based image classification
* 🖼️ Image upload functionality
* 🔄 Image preprocessing and normalization
* 📊 Freshness prediction with confidence scores
* 🌸 User-friendly Streamlit interface
* ☁️ Cloud deployment support
* 📱 Interactive web application

---

## 🛠️ Technologies Used

| Technology         | Purpose                             |
| ------------------ | ----------------------------------- |
| Python             | Programming language                |
| TensorFlow / Keras | Neural network development          |
| NumPy              | Numerical computations              |
| Pillow             | Image loading and preprocessing     |
| Streamlit          | Web application development         |
| GitHub             | Version control and project hosting |

---

## 🧠 Machine Learning Model

The project uses a **Multilayer Perceptron (MLP)** for binary image classification.

### Model Architecture

```text
Input Image
    ↓
Resize to 32 × 32 pixels
    ↓
RGB Image
    ↓
Pixel Normalization
    ↓
Flatten Image into 3072 Features
    ↓
Dense Layer — 128 Neurons, ReLU
    ↓
Dense Layer — 64 Neurons, ReLU
    ↓
Output Layer — 1 Neuron, Sigmoid
    ↓
Fresh / Rotten Prediction
```

### Model Configuration

| Parameter            | Value                 |
| -------------------- | --------------------- |
| Input image size     | 32 × 32               |
| Image channels       | RGB                   |
| Input features       | 3072                  |
| Hidden layers        | 2                     |
| Hidden layer neurons | 128 and 64            |
| Hidden activation    | ReLU                  |
| Output activation    | Sigmoid               |
| Loss function        | Binary Crossentropy   |
| Optimizer            | Adam                  |
| Classification type  | Binary classification |

---

## 📂 Dataset

The dataset contains **3,200 images** distributed across eight fruit types.

| Dataset characteristic  | Details                  |
| ----------------------- | ------------------------ |
| Total images            | 3,200                    |
| Fruit types             | 8                        |
| Fresh images per fruit  | 200                      |
| Rotten images per fruit | 200                      |
| Image format            | RGB                      |
| Image preprocessing     | Resize and normalization |

### Dataset Split

| Dataset        | Number of images |
| -------------- | ---------------: |
| Training set   |            2,560 |
| Validation set |              640 |
| Total          |            3,200 |

The dataset was divided using a stratified approach to preserve the distribution of fruit categories and freshness labels.

---

## ⚙️ Image Preprocessing

Before being passed to the neural network, each image undergoes the following preprocessing steps:

1. Convert the image to RGB format.

2. Resize the image to `32 × 32` pixels.

3. Convert the image into a NumPy array.

4. Normalize pixel values using:

   ```python
   normalized_image = image / 255.0
   ```

5. Flatten the image into a vector of 3,072 features.

The model uses the following input shape:

```text
32 × 32 × 3 = 3072 features
```

---

## 📊 Model Performance

The baseline MLP model achieved the following validation results during experimentation:

| Metric                            |               Result |
| --------------------------------- | -------------------: |
| Validation accuracy               | Approximately 86.41% |
| Best observed validation accuracy | Approximately 88.91% |
| Validation images                 |                  640 |

Performance may vary depending on the model version, training configuration, preprocessing, and evaluation process.

---

## 🖥️ Application Workflow

```text
User uploads a fruit image
          ↓
Image is converted to RGB
          ↓
Image is resized to 32 × 32
          ↓
Pixel values are normalized
          ↓
Image is flattened
          ↓
Trained MLP model processes the image
          ↓
Freshness score is generated
          ↓
Application displays Fresh or Rotten
```

---

## 📁 Project Structure

```text
freshify-fruit-freshness/
│
├── app.py
├── multi_fruit_mlp_baseline.keras
├── requirements.txt
└── README.md
```

### File Description

| File                             | Description               |
| -------------------------------- | ------------------------- |
| `app.py`                         | Streamlit web application |
| `multi_fruit_mlp_baseline.keras` | Trained MLP model         |
| `requirements.txt`               | Required Python libraries |
| `README.md`                      | Project documentation     |

---

## 🚀 Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/ruchabpatil/freshify_fruit_freshness.git
```

Navigate to the project directory:

```bash
cd freshify-fruit-freshness
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

Activate the virtual environment on Windows:

```bash
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit application

```bash
streamlit run app.py
```

The application will open in your browser.

---

## 📦 Requirements

The project requires the following libraries:

```txt
streamlit
tensorflow
numpy
pillow
```

---

## ☁️ Deployment

The Streamlit application can be deployed using a cloud hosting platform that supports Streamlit applications.

Deployment workflow:

1. Upload the project files to GitHub.
2. Connect the repository to the selected hosting platform.
3. Configure the application entry point as `app.py`.
4. Install the dependencies from `requirements.txt`.
5. Deploy the application.
6. Access the application using the generated public URL.

---

## 🔮 Future Enhancements

The project can be extended with the following features:

* [ ] Add more fruit categories.
* [ ] Improve image classification accuracy.
* [ ] Experiment with Convolutional Neural Networks (CNNs).
* [ ] Add fruit-type classification alongside freshness classification.
* [ ] Develop a REST API using FastAPI.
* [ ] Create a separate frontend using React or Next.js.
* [ ] Add model performance monitoring.
* [ ] Improve image quality and preprocessing.
* [ ] Explore transfer learning using pretrained models.
* [ ] Add database support for prediction history.
* [ ] Develop a mobile application.
* [ ] Explore explainable AI techniques for image predictions.

---
