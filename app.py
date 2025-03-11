import streamlit as st
import cv2
import numpy as np
from PIL import Image

def load_model():
    try:
        net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
        if net.empty():
            raise ValueError("Failed to load YOLO model. Check weight and config files.")
        layer_names = net.getLayerNames()
        output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
        with open("coco.names", "r") as f:
            classes = [line.strip() for line in f.readlines()]
        return net, output_layers, classes
    except Exception as e:
        st.error(f"Error loading YOLO model: {e}")
        return None, None, None

def predict(image, net, output_layers):
    try:
        if image is None or net is None or output_layers is None:
            raise ValueError("Invalid inputs for prediction.")
        height, width, channels = image.shape
        blob = cv2.dnn.blobFromImage(image, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        net.setInput(blob)
        outputs = net.forward(output_layers)
        return outputs, width, height
    except Exception as e:
        st.error(f"Error during prediction: {e}")
        return None, None, None

def draw_boxes(image, outputs, width, height, classes):
    try:
        if outputs is None:
            raise ValueError("No output from YOLO model.")
        boxes, confidences, class_ids = [], [], []
        for output in outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:
                    center_x, center_y, w, h = (detection[0:4] * [width, height, width, height]).astype(int)
                    x, y = center_x - w // 2, center_y - h // 2
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)
        
        indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
        if indices is not None and len(indices) > 0:
            for i in indices.flatten():
                x, y, w, h = boxes[i]
                label = f"{classes[class_ids[i]]} {confidences[i]:.2f}"
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        else:
            st.warning("No objects detected.")
        return image
    except Exception as e:
        st.error(f"Error drawing boxes: {e}")
        return image

st.title("YOLOv3 Image Classification App")
st.write("Upload an image to detect objects using YOLOv3")

uploaded_file = st.file_uploader("Choose an image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file)
    image_np = np.array(image)
    
    net, output_layers, classes = load_model()
    if net is not None and output_layers is not None and classes is not None:
        outputs, width, height = predict(image_np, net, output_layers)
        if outputs is not None:
            annotated_image = draw_boxes(image_np.copy(), outputs, width, height, classes)
            st.image(annotated_image, caption="Detected Objects", use_column_width=True)
        else:
            st.error("Prediction failed. Ensure YOLO model is loaded correctly.")
