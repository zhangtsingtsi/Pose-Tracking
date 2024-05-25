import numpy as np


def poses2boxes(poses, image_width, image_height):
    """
    Parameters
    ----------
    poses: ndarray of human 2D poses [People * BodyPart * [x,y, confidence]]
    image_width: int, width of the image to ensure box coordinates are within bounds
    image_height: int, height of the image to ensure box coordinates are within bounds
    
    Returns
    ----------
    boxes: ndarray of containing boxes [People * [x1,y1,x2,y2]]
    """
    # Indices of keypoints to use for bounding box
    indices = [0, 1, 2, 5, 14, 15, 16, 17]
    boxes = []
    for person in poses:
        # Filter out the specific keypoints for each person
        keypoints = person[indices]
        # Filter out keypoints where the confidence is 0 (not detected)
        visible_keypoints = keypoints[keypoints[:, 2] > 0][:, :2]

        if visible_keypoints.size == 0:
            continue  # Skip if no visible keypoints
        
        x1, y1 = np.min(visible_keypoints, axis=0)
        x2, y2 = np.max(visible_keypoints, axis=0)

        # Clamping the coordinates to the image boundaries
        x1 = max(0, min(x1, image_width - 1))
        y1 = max(0, min(y1, image_height - 1))
        x2 = max(0, min(x2, image_width - 1))
        y2 = max(0, min(y2, image_height - 1))

        box = [int(x1), int(y1), int(x2), int(y2)]
        boxes.append(box)
    return np.array(boxes)

