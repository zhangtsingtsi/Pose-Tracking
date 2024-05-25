import numpy as np


def initialize_ids(keypoints):
    ids = [np.mean(kp[:, :2], axis=0).tolist() for kp in keypoints]  # 使用头部中心的坐标作为简单的ID
    return ids

def update_ids(old_ids, current_keypoints):
    current_ids = []
    threshold = 50  # 50像素
    for kp in current_keypoints:
        kp_center = np.mean(kp[:, :2], axis=0)
        # 找到最近的旧ID
        distances = [np.linalg.norm(np.array(oid) - kp_center) for oid in old_ids]
        min_dist = min(distances)
        if min_dist < threshold:
            current_ids.append(old_ids[distances.index(min_dist)])
        else:
            current_ids.append(np.mean(kp[:, :2], axis=0).tolist())  # 新的人物加入，赋予新ID
    return current_ids


def hands_up(sorted_keypoints, previous_hands_up, ids):
    hands_up_count = dict.fromkeys(ids, 0)
    current_hands_up = dict.fromkeys(ids, False)
    
    for kp, pid in zip(sorted_keypoints, ids):
        r_shoulder_y = kp[5][2]
        l_shoulder_y = kp[2][2]
        r_wrist_y = kp[7][2]
        l_wrist_y = kp[4][2]

        current_hands_up[pid] = (r_wrist_y < r_shoulder_y or l_wrist_y < l_shoulder_y)
        if current_hands_up[pid] and not previous_hands_up.get(pid, False):
            hands_up_count[pid] += 1

        previous_hands_up[pid] = current_hands_up[pid]

    return hands_up_count, previous_hands_up