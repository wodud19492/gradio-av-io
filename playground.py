import os
import cv2
import time
import torch
import gradio as gr
from argparse import ArgumentParser
from ibug.face_detection import RetinaFacePredictor, S3FDPredictor
from ibug.face_detection.utils import SimpleFaceTracker, HeadPoseEstimator

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Device is', device)

os.environ['GRADIO_TEMP_DIR'] = os.path.expanduser('~/.gradio')

def video_to_img(video):
    video = cv2.VideoCapture(video)
    fps = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    image_out = None
    for i in range(fps):
        ret, frame = video.read()
        image_out = cv2.imwrite('image_output.png',frame)
    return image_out

def init_face_detector(method='retinaface', weights=None, alternative_pth=None, threshold=0.8, device='cuda:0'):
    method = method.lower().strip()
    if method == 'retinaface':
        face_detector_class = (RetinaFacePredictor, 'RetinaFace')
    elif method == 's3fd':
        face_detector_class = (S3FDPredictor, 'S3FD')
    else:
        raise ValueError('Method must be set to either RetinaFace or S3FD')

    if weights is None:
        fd_model = face_detector_class[0].get_model()
    else:
        fd_model = face_detector_class[0].get_model(weights)

    if alternative_pth is not None:
        fd_model.weights = alternative_pth

    face_detector = face_detector_class[0](threshold=threshold, device=device, model=fd_model)
    return face_detector, face_detector_class[1]

#Initialize the models

face_detector, model_name = init_face_detector()
face_tracker = SimpleFaceTracker(iou_threshold=0.4, minimum_face_size=0.0)
head_pose_estimator = HeadPoseEstimator()

def detect_faces(image):
    try:
        faces = face_detector(image, rgb=False)
        tids = face_tracker(faces)
        if faces.shape[1] >= 15:
            head_poses = [head_pose_estimator(face[5:15].reshape((-1, 2)), *image.shape[1::-1],
                                              output_preference=0)
                          for face in faces]
        else:
            head_poses = [None] * faces.shape[0]

    # Rendering

        colours = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255), (255, 0, 255), (255, 255, 0),
               (0, 128, 255), (128, 255, 0), (255, 0, 128), (128, 0, 255), (0, 255, 128), (255, 128, 0)]
    
        for face, tid, head_pose in zip(faces, tids, head_poses):
            bbox = face[:4].astype(int)
            if tid is None:
                colour = (128, 128, 128)
            else:
                colour = colours[(tid - 1) % len(colours)]
            cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color=colour, thickness=2)
            if len(face) > 5:
                for pts in face[5:].reshape((-1, 2)):
                    cv2.circle(image, tuple(pts.astype(int).tolist()), 3, colour, -1)
            if tid is not None:
                cv2.putText(image, f'Face {tid}', (bbox[0], bbox[1] - 10),
                        cv2.FONT_HERSHEY_DUPLEX, 0.6, colour, lineType=cv2.LINE_AA)
            if head_pose is not None:
                pitch, yaw, roll = head_pose
                cv2.putText(image, f'Pitch: {pitch:.1f}', (bbox[2] + 5, bbox[1] + 10),
                            cv2.FONT_HERSHEY_DUPLEX, 0.5, colour, lineType=cv2.LINE_AA)
                cv2.putText(image, f'Yaw: {yaw:.1f}', (bbox[2] + 5, bbox[1] + 30),
                            cv2.FONT_HERSHEY_DUPLEX, 0.5, colour, lineType=cv2.LINE_AA)
                cv2.putText(image, f'Roll: {roll:.1f}', (bbox[2] + 5, bbox[1] + 50),
                            cv2.FONT_HERSHEY_DUPLEX, 0.5, colour, lineType=cv2.LINE_AA)

        return image
    except Exception as e:
        print(f"Error detecting faces: {e}")
        return image

def main(video):
    image_out = video_to_img(video)
    return detect_faces(image_out)

with gr.Blocks() as demo:
    with gr.TabItem("Face Detection"):
        video_input = gr.Video(sources="webcam")
        image_output = gr.Image()

    video_input.play(
        fn=main,
        inputs=video_input,
        outputs=image_output
    )

demo.launch()