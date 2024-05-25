
def update_hand_states(keypoints, hand_states):
    """
    Update hand states.
    
    Parameters:
    - keypoints: ndarray [num_people, keypoint, 3]
    - hand_states: [{'state': False/True, 'count': num}, ...]
    
    Return: 
    - hand_states
    """
    
    #num_people = keypoints.shape[0]

    for i in range(keypoints.shape[0]):
        #hand_states[i] = {'state': False, 'count': 0}
        right_wrist = keypoints[i, 4, :]
        left_wrist = keypoints[i, 7, :]
        right_shoulder = keypoints[i, 2, :]
        left_shoulder = keypoints[i, 5, :]

        if right_wrist[-1] > 0 and left_wrist[-1] > 0 and right_shoulder[-1] > 0 and left_shoulder[-1] > 0:
            right_wrist_y = right_wrist[1]
            left_wrist_y = left_wrist[1]
            right_shoulder_y = right_shoulder[1]
            left_shoulder_y = left_shoulder[1]

            hand_is_up_now = (right_wrist_y < right_shoulder_y) or (left_wrist_y < left_shoulder_y)
            
            if hand_is_up_now and not hand_states[i]['state']:
                hand_states[i]['count'] += 1
            hand_states[i]['state'] = hand_is_up_now
        else:
            continue
        
    return hand_states
