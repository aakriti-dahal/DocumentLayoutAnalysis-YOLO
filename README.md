# DocumentLayoutAnalysis-YOLO
# 📄 Document Layout Detector

A lightweight Streamlit app that detects and extracts **tables**, **figures**, and **text regions** from uploaded PDFs or images using a YOLOv8 model.

---

## ✨ Features

- 📥 Upload PDFs or images (PNG, JPG, JPEG)
- 🧠 Run layout detection using a YOLO model (`doclayout v1.pt`)
- ✂️ Automatically crop out detected regions:
  - `tables/table_1.png`, `table_2.png`, ...
  - `figures/figure_1.png`, `figure_2.png`, ...
  - `text/text_1.png`, `text_2.png`, ...
- 📁 All cropped outputs saved into a `cropped/` directory (auto-created if missing)

---

## 🚀 How It Works

1. Upload a document (PDF or image)
2. PDFs are converted into images
3. YOLOv8 model detects layout components
4. Each detected region is cropped and saved based on its class name

---

## 🛠️ Installation

```bash
https://github.com/aakriti-dahal/DocumentLayoutAnalysis-YOLO.git
cd DocumentLayoutAnalysis-YOLO
pip install -r requirements.txt
streamlit run UI.py
