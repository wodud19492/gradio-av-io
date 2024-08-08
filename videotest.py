import gradio as gr
import torch
from facenet_pytorch import MTCNN
import cv2


# Load the face detection model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
mtcnn = MTCNN(keep_all=True, device=device)

def detect_faces(im):
    # Detect faces
    im_zip = cv2.resize(im, (426, 240), interpolation=cv2.INTER_AREA)
    boxes, _ = mtcnn.detect(im_zip)
    
    if boxes is not None:
        for box in boxes:
            cv2.rectangle(im_zip, 
                          (int(box[0]), int(box[1])), 
                          (int(box[2]), int(box[3])), 
                          (0, 255, 0), 
                          2)
    return im_zip

# Create a Gradio interface for real-time video
iface = gr.Interface(fn=detect_faces, 
                     inputs=gr.Image(sources='webcam', streaming = True), 
                     outputs=gr.Image(type='numpy'),
                     live=True,
                     title="Real-Time Face Detection",
                     description="Use your webcam to detect faces in real-time using a PyTorch model.")

# Launch the interface
iface.launch()
