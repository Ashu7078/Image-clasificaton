# YOLO Object Detection with Streamlit  

This Streamlit app allows users to perform object detection on both images and videos using YOLO (You Only Look Once). The app also features a dynamic background video, a progress bar for processing, and a user-friendly interface.

---

## Features  

- **Upload Media**: Single uploader for both images and videos.  
- **YOLO Detection**: Object detection with bounding boxes and class labels.  
- **Progress Bar**: Indicates processing progress for videos.  
- **Interactive Interface**: Intuitive UI for seamless media processing.  

---

## Installation  

1. **Clone the Repository**:
    ```bash
    git clone https://github.com/MuskanVerma24/object-detection-using-yolo.git
    cd object-detection-using-yolo
    ```

2. **Install Requirements**:
    ```bash
    pip install -r requirements.txt
    ```

3. **Download YOLO Files**:
    - **`yolov3.weights`**: Pre-trained weights file required for object detection. Download it from the [YOLO website](https://pjreddie.com/darknet/yolo/).
    - **`yolov3.cfg`**: Configuration file defining the YOLOv3 network architecture.  
    - **`coco.names`**: A list of 80 object categories supported by YOLO.

4. **Run the App**:
    ```bash
    streamlit run app.py
    ```

---

## Usage  

1. **Upload Media**:
   - Drag and drop an image or video into the uploader.  

2. **Processing**:
   - The app will detect objects in images/videos and display the results with bounding boxes and labels.  

---

## Contributing  

Contributions are welcome! Feel free to open an issue or submit a pull request.  

---

## Acknowledgments  

- **YOLOv3**: Pre-trained model by Joseph Redmon and Ali Farhadi.  
- **Streamlit**: Framework for building interactive Python apps.  

---
