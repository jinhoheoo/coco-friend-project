#  이것은 cam1.py 파일이야

import cv2
import numpy as np
import collections
import tarfile
import time
from pathlib import Path
import ipywidgets as widgets
from IPython import display
import openvino as ov
from openvino.tools.mo.front import tf as ov_tf_front
from openvino.tools import mo
import os
import urllib.request
import notebook_utils as utils
from keras.models import load_model

def val(VAL2):
    return VAL2

# measure_height 함수 정의
def measure_height(frame):
    # 캘리브레이션 값
    PIXEL_TO_MM = 1

    # 흑백으로 변환 후 이진화
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY)

    # 이진화된 이미지에서 윤곽 찾기
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 가장 큰 윤곽 찾기 (아기라고 가정)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)

        # 픽셀 단위의 높이 계산
        height_in_pixels = h

        # 픽셀을 실제 거리로 변환
        height_in_mm = height_in_pixels * PIXEL_TO_MM

        return height_in_mm
    else:
        return None

# 다운로드 및 파일 관리 함수
def download_file(url, filename, save_dir='.'):
    os.makedirs(save_dir, exist_ok=True)
    filepath = os.path.join(save_dir, filename)
    urllib.request.urlretrieve(url, filepath)
    return filepath

# 모델 다운로드 및 변환
base_model_dir = Path("model")
model_name = "ssdlite_mobilenet_v2"
archive_name = Path(f"{model_name}_coco_2018_05_09.tar.gz")
model_url = f"https://storage.openvinotoolkit.org/repositories/open_model_zoo/public/2022.1/{model_name}/{archive_name}"

if not (base_model_dir / archive_name).exists():
    download_file(model_url, archive_name.name, base_model_dir)

tf_model_path = base_model_dir / archive_name.with_suffix("").stem / "frozen_inference_graph.pb"

precision = "FP16"
converted_model_path = Path("model") / f"{model_name}_{precision.lower()}.xml"

if not converted_model_path.exists():
    trans_config_path = Path(ov_tf_front.__file__).parent / "ssd_v2_support.json"
    ov_model = mo.convert_model(
        tf_model_path, 
        compress_to_fp16=(precision == 'FP16'), 
        transformations_config=trans_config_path,
        tensorflow_object_detection_api_pipeline_config=tf_model_path.parent / "pipeline.config", 
        reverse_input_channels=True
    )
    ov.save_model(ov_model, converted_model_path)
    del ov_model

# 모델 로드
core = ov.Core()
device = "CPU"
model = core.read_model(model=converted_model_path)
compiled_model = core.compile_model(model=model, device_name=device)
input_layer = compiled_model.input(0)
output_layer = compiled_model.output(0)
height, width = list(input_layer.shape)[1:3]

# 클래스 및 색상 정의
classes = [
    "background", "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "street sign", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant",
    "bear", "zebra", "giraffe", "hat", "backpack", "umbrella", "shoe", "eye glasses",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
    "plate", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "mirror", "dining table", "window", "desk", "toilet",
    "door", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven",
    "toaster", "sink", "refrigerator", "blender", "book", "clock", "vase", "scissors",
    "teddy bear", "hair drier", "toothbrush", "hair brush"
]

expression_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# 표정 가중치 모델
emotion_model = load_model('./models/emotion_model.hdf5')

# Colors for the classes above (Rainbow Color Map).
colors = cv2.applyColorMap(
    src=np.arange(0, 255, 255 / len(classes), dtype=np.float32).astype(np.uint8),
    colormap=cv2.COLORMAP_RAINBOW,
).squeeze()

# Process Results 함수
def process_results(frame, results, thresh=0.6):
    h, w = frame.shape[:2]
    results = results.squeeze()
    boxes = []
    labels = []
    scores = []
    for _, label, score, xmin, ymin, xmax, ymax in results:
        boxes.append(
            tuple(map(int, (xmin * w, ymin * h, (xmax - xmin) * w, (ymax - ymin) * h)))
        )
        labels.append(int(label))
        scores.append(float(score))

    indices = cv2.dnn.NMSBoxes(
        bboxes=boxes, scores=scores, score_threshold=thresh, nms_threshold=0.6
    )

    if len(indices) == 0:
        return []

    return [(labels[idx], scores[idx], boxes[idx]) for idx in indices.flatten()]


# Draw Boxes 함수
def draw_boxes(frame, boxes):
    for label, score, box in boxes:
        color = tuple(map(int, colors[label]))
        x2 = box[0] + box[2]
        y2 = box[1] + box[3]
        cv2.rectangle(img=frame, pt1=box[:2], pt2=(x2, y2), color=color, thickness=3)

        cv2.putText(
            img=frame,
            text=f"{classes[label]} {score:.2f}",
            org=(box[0] + 10, box[1] + 30),
            fontFace=cv2.FONT_HERSHEY_COMPLEX,
            fontScale=frame.shape[1] / 1000,
            color=color,
            thickness=1,
            lineType=cv2.LINE_AA,
        )

        if classes[label] == "person":
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            face_roi = gray[box[1]:y2, box[0]:x2]

            face_roi = cv2.resize(face_roi, (64, 64))
            face_roi = np.expand_dims(face_roi, axis=-1)
            face_roi = np.expand_dims(face_roi, axis=0)
            face_roi = face_roi / 255.0

            output = emotion_model.predict(face_roi)[0]
            expression_index = np.argmax(output)
            expression_label = expression_labels[expression_index]
    
            global VAL1
            VAL1= expression_label
            #val(expression_label)
            cv2.putText(frame, expression_label, (box[0], box[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    return frame

# Main Processing Function
# Main Processing Function
def run_object_detection(source=0, flip=False, use_popup=False, skip_first_frames=0):
    player = None
    try:
        player = utils.VideoPlayer(
            source=source, flip=flip, fps=30, skip_first_frames=skip_first_frames
        )
        player.start()
        if use_popup:
            title = "Press ESC to Exit"
            cv2.namedWindow(
                winname=title, flags=cv2.WINDOW_GUI_NORMAL | cv2.WINDOW_AUTOSIZE
            )

        processing_times = collections.deque()
        while True:
            frame = player.next()
            if frame is None:
                print("Source ended")
                break

            scale = 1280 / max(frame.shape)
            if scale < 1:
                frame = cv2.resize(
                    src=frame,
                    dsize=None,
                    fx=scale,
                    fy=scale,
                    interpolation=cv2.INTER_AREA,
                )

            input_img = cv2.resize(
                src=frame, dsize=(width, height), interpolation=cv2.INTER_AREA
            )
            input_img = input_img[np.newaxis, ...]

            start_time = time.time()
            results = compiled_model([input_img])[output_layer]
            stop_time = time.time()
            boxes = process_results(frame=frame, results=results)

            height_mm = None
            height_mm = None
            for label, _, box in boxes:
                if classes[label] == "person":
                    height_mm = measure_height(frame[box[1]:box[1] + box[3], box[0]:box[0] + box[2]])
                    if height_mm is not None:
                        height_cm = height_mm * 0.1  # mm를 cm로 변환
                        cv2.putText(frame, f"Height: {height_cm:.2f}cm", (box[0], box[1] - 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                    break




            frame = draw_boxes(frame=frame, boxes=boxes)

            processing_times.append(stop_time - start_time)
            if len(processing_times) > 200:
                processing_times.popleft()

            _, f_width = frame.shape[:2]
            processing_time = np.mean(processing_times) * 1000
            fps = 1000 / processing_time
            cv2.putText(
                img=frame,
                text=f"Inference time: {processing_time:.1f}ms ({fps:.1f} FPS)",
                org=(20, 40),
                fontFace=cv2.FONT_HERSHEY_COMPLEX,
                fontScale=f_width / 1000,
                color=(0, 0, 255),
                thickness=1,
                lineType=cv2.LINE_AA,
            )

            if use_popup:
                cv2.imshow(winname=title, mat=frame)
                key = cv2.waitKey(50)
                if 3 > time.time() or key == 'q':
                    break
            else:
                _, encoded_img = cv2.imencode(
                    ext=".jpg", img=frame, params=[cv2.IMWRITE_JPEG_QUALITY, 100]
                )
                i = display.Image(data=encoded_img)
                display.clear_output(wait=True)
                display.display(i)
    except KeyboardInterrupt:
        print("Interrupted")
    except RuntimeError as e:
        print(e)
    finally:
        if player is not None:
            player.stop()
        if use_popup:
            cv2.destroyAllWindows()


# 객체 탐지 실행
USE_WEBCAM = True
#video_file = "https://storage.openvinotoolkit.org/repositories/openvino_notebooks/data/data/video/Coco%20Walking%20in%20Berkeley.mp4"
video_file = "/home/ubuntu/workdir2/cam/baby.gif"

cam_id = 0
source = cam_id if USE_WEBCAM else video_file

# 객체 탐지 실행
run_object_detection(source=source, flip=isinstance(source, int), use_popup=True)


