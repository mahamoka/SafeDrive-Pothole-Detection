import streamlit as st
import cv2
import numpy as np
import tempfile
import os
from ultralytics import YOLO
from PIL import Image

# Load YOLO Model
model = YOLO("best.pt")  # Ensure 'best.pt' is in the same folder

# Streamlit UI
st.title("🚗 SafeDrive: Pothole & Speed Breaker Detection")
st.write("Upload an image or video to detect potholes and speed breakers.")

# Upload Image/Video
uploaded_file = st.file_uploader("Choose an image or video...", type=["jpg", "png", "mp4"])

if uploaded_file is not None:
    file_bytes = uploaded_file.read()

    # If Image
    if uploaded_file.type in ["image/jpeg", "image/png"]:
        image = Image.open(uploaded_file)
        image = np.array(image)

        # Run YOLO Detection
        results = model(image)

        # Draw bounding boxes
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                label = f"{model.names[cls]}: {conf:.2f}"
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Show Processed Image
        st.image(image, caption="Detected Objects", use_container_width=True)

    # If Video
    elif uploaded_file.type == "video/mp4":
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        tfile.write(file_bytes)
        tfile_path = tfile.name

        # Open Video
        cap = cv2.VideoCapture(tfile_path)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        output_path = "processed_video.mp4"
        out = cv2.VideoWriter(
            output_path, fourcc, cap.get(cv2.CAP_PROP_FPS),
            (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        )

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Run YOLO on Frame
            results = model(frame)
            for result in results:
                for box in result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = float(box.conf[0])
                    cls = int(box.cls[0])
                    label = f"{model.names[cls]}: {conf:.2f}"
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            out.write(frame)

        cap.release()
        out.release()

        # Provide Download Link for Processed Video
        with open(output_path, "rb") as f:
            st.download_button(label="Download Processed Video", data=f, file_name="detected_potholes.mp4", mime="video/mp4")
