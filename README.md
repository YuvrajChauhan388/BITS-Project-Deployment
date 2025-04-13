# 🔬 Sybilytics-AI: Signal Analysis Dashboard

A powerful Streamlit-based dashboard for uploading, analyzing, visualizing, denoising, and extracting features from time-series sensor data.

---

## 🔍 Features

### 📁 File Upload
- Upload `.txt` or `.lvm` sensor data files with time-series signals.

### 📈 Signal Visualization
- Time-domain plots (raw & denoised)
- Wavelet decomposition visualizations (approximation & detail coefficients)
- FFT (Fast Fourier Transform) plots
- Spectrograms using STFT (Short-Time Fourier Transform)

### 🌊 Wavelet Processing
- Support for **Biorthogonal (bior)** wavelet family
- Selectable decomposition levels (1–20)
- Soft thresholding for denoising

### 📊 Statistical Feature Extraction
- 20+ statistical features including:
  - Mean, RMS, Skewness, Kurtosis, Entropy, Crest Factor, and more
- Separate analysis for **raw** and **denoised** signals

### 📤 Downloadable Outputs
- Export visualizations as **PNG**
- Export statistical features as **CSV**

---

## 🚀 How It Works

### 📌 Upload Data
- Upload a tab-delimited `.txt` or `.lvm` file
- The app automatically parses the columns

### 📌 Select Columns
- Choose which columns represent **Time** and **Signal**

### 📌 Visualize & Denoise
- Compare raw vs denoised signals
- Customize wavelet type and decomposition level
- Explore wavelet coefficients and correlation plots

### 📌 Explore Frequency Content
- Analyze FFT of selected components
- Visualize time-frequency behavior via spectrogram

### 📌 Extract Features
- View and download a full suite of statistical parameters for both raw and denoised signals

---

## 📦 Installation

To run this app locally:

```bash
git clone https://github.com/YuvrajChauhan388/sybilytics-ai.git
cd sybilytics-ai
pip install -r requirements.txt
streamlit run app.py
