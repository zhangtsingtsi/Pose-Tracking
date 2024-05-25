import sys
sys.path.append('../')
import os
import cv2
from openpose.body.estimator import BodyPoseEstimator
from openpose.utils import draw_body_connections, draw_keypoints
from hands_up import initialize_ids, update_ids, hands_up


estimator = BodyPoseEstimator(pretrained=True)

video_file = './media/hands_up1.mp4'
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
ids = []
while videoclip.isOpened():
    flag, frame = videoclip.read()
    if not flag:
        break
    
    keypoints = estimator(frame)
    selected_indices = [0, 1, 2, 3, 4, 5, 6, 7, 14, 15, 16, 17]
    selected_keypoints = [keypoints[i] for i in selected_indices if i < len(keypoints)]
    
    sorted_keypoints = sorted(selected_keypoints, key=lambda x: x[1][0])
    
    if not ids:
        ids = initialize_ids(sorted_keypoints)
    else:
        ids = update_ids(ids, sorted_keypoints)
    
    hands_up_count, previous_hands_up = hands_up(sorted_keypoints, previous_hands_up, ids)
    
    frame = draw_body_connections(frame, sorted_keypoints, thickness=2, alpha=0.7)
    frame = draw_keypoints(frame, sorted_keypoints, radius=4, alpha=0.8)
    
    for i, person in enumerate(sorted_keypoints):
        head_x, head_y = int(person[0][1]), int(person[0][2])
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
