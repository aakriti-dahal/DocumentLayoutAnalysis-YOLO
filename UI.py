import streamlit as st
import tempfile
import os
from ultralytics import YOLO
from pdf2image import convert_from_path
from PIL import Image
import cv2
import numpy as np

# Load your YOLO model
model = YOLO("doclayout v1.pt")  # replace with correct path if needed

st.title("Document Layout Detector")
st.write("Upload a PDF or image. The model will detect tables, figures, and text, then save cropped results.")

# File uploader
uploaded_file = st.file_uploader("Upload PDF or Image", type=["pdf", "png", "jpg", "jpeg"])

# Ensure output dir exists
output_crop_dir = "cropped"
os.makedirs(output_crop_dir, exist_ok=True)

def draw_and_save_crops(results, image_np, output_dir):
    counters = {"tables": 1, "figures": 1, "text": 1}
    
    for r in results:
        boxes = r.boxes
        names = r.names

        for box in boxes:
            cls_id = int(box.cls[0])
            cls_name = names[cls_id].lower()

            # Normalize class names
            if "table" in cls_name:
                key = "tables"
            elif "figure" in cls_name or "image" in cls_name:
                key = "figures"
            elif "text" in cls_name or "paragraph" in cls_name or "title" in cls_name:
                key = "text"
            else:
                continue

            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            crop_img = image_np[y1:y2, x1:x2]

            filename = f"{key}_{counters[key]}.png"
            filepath = os.path.join(output_dir, filename)
            cv2.imwrite(filepath, crop_img)
            counters[key] += 1

            # Optionally draw box on image
            cv2.rectangle(image_np, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(image_np, f"{key}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
    
    return image_np

if uploaded_file:
    with tempfile.TemporaryDirectory() as tmpdir:
        file_path = os.path.join(tmpdir, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.read())

        images = []

        if uploaded_file.name.endswith(".pdf"):
            pages = convert_from_path(file_path)
            images = [np.array(p.convert("RGB")) for p in pages]
        else:
            img = Image.open(file_path).convert("RGB")
            images = [np.array(img)]

        st.write(f"Processing {len(images)} page(s)...")
        for idx, img_np in enumerate(images):
            results = model(img_np)
            processed_img = draw_and_save_crops(results, img_np.copy(), output_crop_dir)
            st.image(processed_img, caption=f"Detected Elements on Page {idx+1}")
        
        st.success(f"All cropped regions saved to '{output_crop_dir}/' directory.")
