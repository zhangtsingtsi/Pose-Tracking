import sys
sys.path.append('../')
import os
import cv2
import numpy as np
from pathlib import Path
import argparse

from boxmot import DeepOCSORT

from openpose.body.estimator import BodyPoseEstimator
from openpose.utils import draw_body_connections, draw_keypoints
from utils.pose2boxes import poses2boxes


def parse_arguments():
    parser = argparse.ArgumentParser(description="Hands Up!")
    parser.add_argument("--video_file", type=str, default='./video/hands_up1.mp4', help="Path to the input video file or '0' for webcam.")
    # parser.add_argument("--output_dir", type=str, default='./output/output_hand_up1.mp4', help="Path to the output directory for processed video.")
    return parser.parse_args()

class PoseTracking():
    def __init__(self):
        self.tracker = DeepOCSORT(
            model_weights=Path('./tracking/osnet_x0_25_msmt17.pt'), # which ReID model to use
            device='cuda:0',
            fp16=False,)
        self.estimator = BodyPoseEstimator(pretrained=True)
    
    def run_local(self, video_file):
        if not Path(video_file).exists():
            print(f"Video file {video_file} not found.")
            return
        # output path
        output_dir = './output/'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        output_file = Path(output_dir) / ('output_' + Path(video_file).name)
        
        videoclip = cv2.VideoCapture(video_file)
        width = int(videoclip.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(videoclip.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = videoclip.get(cv2.CAP_PROP_FPS)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # or 'X264', 'avc1'
        writer = cv2.VideoWriter(str(output_file), fourcc, fps, (width, height))
        
        while videoclip.isOpened():
            ret, frame = videoclip.read()
            if not ret:
                break
            
            keypoints = self.estimator(frame)  # [p, 18, 3]
            boxes = poses2boxes(keypoints, width, height)
            boxes_xyxysc = [[x1, y1, x2, y2, 1.0, 0] for [x1, y1, x2, y2] in boxes]
            tracks = self.tracker.update(np.array(boxes_xyxysc, dtype=np.float32), frame)  # boxes_xyxysc --> (x, y, x, y, id, conf, cls, ind)
            self.tracker.plot_results(frame, show_trajectories=True)
            
            xyxys = tracks[:, 0:4].astype('int')  # top-left and bottom-right and score
            ids = tracks[:, 4].astype('int')  # tracking ID
            confs = tracks[:, 5]  # confidences
            clss = tracks[:, 6].astype('int')  # class ID. person:0
            inds = tracks[:, 7].astype('int')  # indexs  
            # xyxys: [[849, 218, 1017, 359], [381, 291, 537, 393]], ids: [2, 1], confs: [1., 1.], clss: [0, 0], inds: [1, 0]
            # In this frame，the id of the first bbox is 2, the index of the keypoint to the first bbox is 1.
            # xyxys: [[847, 217, 1016, 359], [370, 293, 528, 392]], ids: [2, 1], confs: [1., 1.], clss: [0, 0], inds: [0, 1]
            # In this frame，the id of the first bbox is 2, the index of the keypoint to the first bbox is 1.
            print(f'xyxys: {xyxys}, ids: {ids}, confs: {confs}, clss: {clss}, inds: {inds}')

            # in case you have poses alongside with your detections you can use
            # the ind variable in order to identify which track is associated to each pose by:
            keypoints = keypoints[inds]
            # such that you then can zip(tracks, keypoints)
            
            frame = draw_body_connections(frame, keypoints, thickness=2, alpha=0.7)
            frame = draw_keypoints(frame, keypoints, radius=4, alpha=0.8)
            
            writer.write(frame)
            # break on pressing q or space
            # cv2.imshow('Hands up', frame)     
            # key = cv2.waitKey(1) & 0xFF
            # if key == ord(' ') or key == ord('q'):
            #    break
        videoclip.release()
        writer.release()
        #cv2.destroyAllWindows()
        print(f'Output video saved as {output_file}. Processing finished!')

    def run_webcam():
        # videoclip = cv2.VideoCapture(0) # webcam
        return


if __name__ == '__main__':
    args = parse_arguments()
    poseTracking = PoseTracking()
    if args.video_file == '0':
        poseTracking.run_webcam()
    else:
        poseTracking.run_local(args.video_file)
        
        