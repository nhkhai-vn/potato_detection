import numpy as np


def to_yolo(segmentation, imwidth, imheight):
    '''
        Convert an instance's contours to one YOLO segment polygon, normalized to [0, 1]
    '''
    polygon = max(segmentation, key=len).reshape(-1, 2).astype(float)
    polygon[:, 0] = np.clip(polygon[:, 0] / imwidth, 0, 1)
    polygon[:, 1] = np.clip(polygon[:, 1] / imheight, 0, 1)
    return [round(x, 6) for x in polygon.flatten().tolist()]
