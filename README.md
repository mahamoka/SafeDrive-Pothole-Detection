SAFEDRIVE – ADVANCED DETECTION OF POTHOLES AND SPEED BREAKERS USING DEEP LEARNING:

SafeDrive is a real-time road hazard detection system that identifies potholes and speed breakers from images and videos using deep learning. Built using YOLOv11 (Ultralytics) for object detection, the model was trained on a custom-annotated dataset (created with Roboflow) containing diverse road conditions and lighting environments.

The project includes a Streamlit web application that allows users to upload images or videos and instantly view detected potholes and speed breakers with bounding boxes — making it easy for drivers, road maintenance teams, and civic authorities to identify road hazards.

Key Features:

Real-time detection of potholes and speed breakers using YOLOv11
Custom dataset annotation and preprocessing via Roboflow
Interactive Streamlit interface for image/video uploads
Achieved strong model performance with mAP@0.5: 0.75, Precision: 0.85, Recall: 0.74

Tech Stack: Python, YOLOv11 (Ultralytics), OpenCV, Streamlit, NumPy, Pandas, Matplotlib

Applications: Road safety monitoring, proactive infrastructure maintenance, ADAS (Advanced Driver Assistance Systems), and smart city planning
