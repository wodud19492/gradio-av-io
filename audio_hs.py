import gradio as gr  # 그라디오
import numpy as np  # 행렬연산
import matplotlib.pyplot as plt  # 그래프
import librosa  # 음성처리
import librosa.display  # 스펙트로그램
import io  # 입출력 처리를 위한 라이브러리 - 여기서는 디스크에 저장하지 않고 메모리를 사용하는데 사용됨.
import base64  # 바이너리를 텍스트로 변환해주는 라이브러리
import torch  # 파이토치, 딥러닝에 필요한 라이브러리를 제공
from transformers import pipeline  # 트랜스포머 모델을 사용해서 이미 학습된 모델을 가져오는데 사용
from PIL import Image  # 파이썬 이미지 처리 라이브러리
import os  # 운영체제 관련 라이브러리
import cv2  # 컴퓨터비전 라이브러리

HOME_DIR = os.path.expanduser("~")  # HOME_DIR을 경로로 설정

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

"""
Transcriber 설정
"""
transcriber = pipeline("automatic-speech-recognition", model="openai/whisper-base.en", device=device)

MAX_STREAM_DURATION = 8  # 최대 스트림 길이 (초)
def create_spectrogram(y, sr, n_fft=2048):
    try:
        time_per_chunk = n_fft / sr
        print(f"Creating spectrogram with n_fft={n_fft} (Time per chunk: {time_per_chunk:.4f} seconds)...")
        plt.figure(figsize=(10, 4))
        S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, n_fft=n_fft)
        S_DB = librosa.power_to_db(S=S, ref=np.max)
        librosa.display.specshow(S_DB, sr=sr, x_axis='time', y_axis='mel')
        plt.colorbar(format='%+2.0f dB')
        plt.title('Real-time Mel-frequency spectrogram')
        plt.tight_layout()

    # 메모리에 저장
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        img_str = base64.b64encode(buffer.read()).decode('utf-8')
        buffer.close()
        plt.close()
        print("Spectrogram created and encoded")
        return f'<img src="data:image/png;base64,{img_str}">'
    
    except Exception as e:
        print(f"Error creating spectrogram: {e}")  # 예외가 발생하였을 때의 예외를 출력
    return None

def transcribe_and_create_spectrogram(stream, new_chunk, n_fft=2048):
    try:
        print("Receiving new chunk")
        sr, y = new_chunk
        y = y.astype(np.float32)
        y /= np.max(np.abs(y))  # 정규화

        if stream is not None:
            stream = np.concatenate((stream, y))  # 여기에는 y값들 + stream(이전의 y값들)
            # 스트림 길이 제한
            max_length = int(sr * MAX_STREAM_DURATION)
            if len(stream) > max_length:
                stream = stream[-max_length:]
        else:
            stream = y

        print("Transcribing audio...")
        result = transcriber({"sampling_rate": sr, "raw": stream})
        text = result["text"]
        print(f"Transcription result: {result}")
        spectrogram_image = create_spectrogram(stream, sr, n_fft)
        print(f"Transcription: {text}")
    
        return stream, text, spectrogram_image
    
    except Exception as e:
        print(f"Error in transcribe and spectrogram: {e}")
    return stream, "", None

"""
YuNet 모델 로드
"""

model_path = "yunet.onnx"
try:
    face_detector = cv2.FaceDetectorYN.create(model_path, "", (320, 320))
    model_loaded = "YuNet model loaded successfully."
except Exception as e:
    face_detector = None
    model_loaded = f"Error loading YuNet model: {e}"

def detect_faces(image):
    try:
        if face_detector is None:
            raise Exception("Face detector is not initialized.")

        print("Detecting faces using YuNet...")
        height, width = image.shape[:2]
        face_detector.setInputSize((width, height))

        _, faces = face_detector.detect(image)

        if faces is not None:
            for face in faces:
                box = face[:4].astype(int)
                cv2.rectangle(image, (box[0], box[1]), (box[0] + box[2], box[1] + box[3]), (0, 255, 0), 2)

        return image
    except Exception as e:
        print(f"Error detecting faces: {e}")
        return image

with gr.Blocks() as demo:  # gr.Blocks()는 gradio에서 기본 인터페이스의 컨테이너(demo)를 생성하는 것을 의미
    with gr.TabItem("Audio Transcription"):  # Audio Transcription이라는 이름의 탭을 생성
        state = gr.State()
        audio_input = gr.Audio(sources="microphone", type="numpy", streaming=True)  # 오디오 입력을 받는 UI 요소를 생성
        text_output = gr.Textbox()
        spectrogram_output = gr.HTML()  # HTML 형식의 출력을 위한 UI 요소를 생성

    # 오디오 입력이 변경될 때마다 함수 호출
        audio_input.stream(
            fn=transcribe_and_create_spectrogram,
            inputs=[state, audio_input],
            outputs=[state, text_output, spectrogram_output]
        )

    with gr.TabItem("Camera Feed"):
        model_status = gr.Markdown(value=model_loaded)
        image_input = gr.Image(sources="webcam", streaming=True)
        image_output = gr.Image()

        image_input.stream(
            fn=detect_faces,
            inputs=image_input,
            outputs=image_output
        )
demo.launch()