import sys
sys.path.append('../')
import os
import cv2
import numpy as np
from pathlib import Path

from boxmot import DeepOCSORT
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction


tracker = DeepOCSORT(
    model_weights=Path('./tracking/osnet_x0_25_msmt17.pt'), # which ReID model to use
    device='cuda:0',
    fp16=False,
)

detection_model = AutoDetectionModel.from_pretrained(
    model_type='yolov8',
    model_path='./tracking/yolov8l.pt',
    confidence_threshold=0.75,
    device='cuda:0',  # or 'cuda:0'
)

video_file = '/home/zhang/Desktop/Code/openpose-pytorch/examples/media/hands_up1.mp4'
output_dir = './output/'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
output_file = os.path.join(output_dir, 'output_' + os.path.basename(video_file))

videoclip = cv2.VideoCapture(video_file)
width = int(videoclip.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(videoclip.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = videoclip.get(cv2.CAP_PROP_FPS)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # or 'X264', 'avc1'
writer = cv2.VideoWriter(output_file, fourcc, fps, (width, height))

# webcam
# videoclip = cv2.VideoCapture(0)

while videoclip.isOpened():
    ret, im = videoclip.read()
    if not ret:
        break

    # get sliced predictions
    result = get_sliced_prediction(
        im,
        detection_model,
        slice_height=256,
        slice_width=256,
        overlap_height_ratio=0.2,
        overlap_width_ratio=0.2, 
        postprocess_type="NMS",
        postprocess_match_threshold=0.75
    )
    # only person
    filtered_predictions = [
        prediction for prediction in result.object_prediction_list
        if prediction.category.name == 'person'
    ]
    num_predictions = len(filtered_predictions)
    dets = np.zeros([num_predictions, 6], dtype=np.float32)
    #print(f'1xyxy: {dets}')
    for ind, object_prediction in enumerate(filtered_predictions):
        dets[ind, :4] = np.array(object_prediction.bbox.to_xyxy(), dtype=np.float32)
        dets[ind, 4] = object_prediction.score.value
        dets[ind, 5] = object_prediction.category.id
    #print(f'2xyxy: {dets[ind, :4]}')

    tracks = tracker.update(dets, im) # --> (x, y, x, y, id, conf, cls, ind)

    tracker.plot_results(im, show_trajectories=True)
    
    writer.write(im)

    # # break on pressing q or space
    #cv2.imshow('BoxMOT detection', im)     
    #key = cv2.waitKey(1) & 0xFF
    #if key == ord(' ') or key == ord('q'):
    #    break

videoclip.release()
writer.release()
cv2.destroyAllWindows()
print('Finish!')
