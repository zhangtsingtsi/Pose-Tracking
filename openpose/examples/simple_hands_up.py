import numpy as np


def hands_up(keypoints, previous_hands_up, hands_up_count):

    current_hands_up = [False] * len(keypoints)
    
    for i, person in enumerate(keypoints):
        r_shoulder_y = person[5][2]
        l_shoulder_y = person[2][2]
        r_wrist_y = person[7][2]
        l_wrist_y = person[4][2]
        
        if r_wrist_y < r_shoulder_y or l_wrist_y < l_shoulder_y:
            current_hands_up[i] = True
        else:
            current_hands_up[i] = False
            
        if current_hands_up[i] and not previous_hands_up[i]:
            hands_up_count[i] += 1

        previous_hands_up[i] = current_hands_up[i]

    return hands_up_count, previous_hands_up