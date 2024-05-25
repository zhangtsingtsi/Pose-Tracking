import sys
sys.path.append('../')
import os
import cv2
from openpose.body.estimator import BodyPoseEstimator
from openpose.utils import draw_body_connections, draw_keypoints
from simple_hands_up import hands_up


estimator = BodyPoseEstimator(pretrained=True)

video_file = './media/hands_up_60.mp4'
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

if not writer.isOpened():
    print("Error: VideoWriter not opened.")
    sys.exit()

previous_hands_up = []
hands_up_count = []
while videoclip.isOpened():
    flag, frame = videoclip.read()
    if not flag:
        break
    
    keypoints = estimator(frame)
    frame = draw_body_connections(frame, keypoints, thickness=2, alpha=0.7)
    frame = draw_keypoints(frame, keypoints, radius=4, alpha=0.8)
    
    keypoints = sorted(keypoints, key=lambda x: x[0][0])
    
    if not previous_hands_up:
        previous_hands_up = [False] * len(keypoints)
    if not hands_up_count:
        hands_up_count = [0] * len(keypoints)
    print(f'previous_hands_up: {previous_hands_up}')
    print(f'hands_up_count: {hands_up_count}')
    hands_up_count, previous_hands_up = hands_up(keypoints, previous_hands_up, hands_up_count)
    
    for i, person in enumerate(keypoints):
        head_x, head_y = int(person[0][0]), int(person[0][1])
        cv2.putText(frame, f'Hands up: {hands_up_count[i]}', (head_x, head_y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    writer.write(frame)
    
    # cv2.imshow('Video Demo', frame)
    # if cv2.waitKey(20) & 0xff == 27:
    #     break
    
videoclip.release()
writer.release()
print('Finish!')
# cv2.destroyAllWindows()
