import numpy as np
from numba import jit
import itertools
import mimetypes

mimetypes.init()


def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = itertools.tee(iterable)
    next(b, None)
    return izip(a, b)


def poses2boxes(poses):
    global seen_bodyparts
    """
    Parameters
    ----------
    poses: ndarray of human 2D poses [People * BodyPart]
    Returns
    ----------
    boxes: ndarray of containing boxes [People * [x1,y1,x2,y2]]
    """
    boxes = []
    for person in poses:
        seen_bodyparts = person[np.where((person[:, 0] != 0) | (person[:, 1] != 0))]
        # box = [ int(min(seen_bodyparts[:,0])),int(min(seen_bodyparts[:,1])),
        #        int(max(seen_bodyparts[:,0])),int(max(seen_bodyparts[:,1]))]
        mean = np.mean(seen_bodyparts, axis=0)
        deviation = np.std(seen_bodyparts, axis=0)
        box = [int(mean[0] - deviation[0]), int(mean[1] - deviation[1]), int(mean[0] + deviation[0]),
               int(mean[1] + deviation[1])]
        boxes.append(box)
    return np.array(boxes)


def distancia_midpoints(mid1, mid2):
    return np.linalg.norm(np.array(mid1) - np.array(mid2))


def pose2midpoint(pose):
    """
    Parameters
    ----------
    poses: ndarray of human 2D pose [BodyPart]
    Returns
    ----------
    boxes: pose midpint [x,y]
    """
    box = poses2boxes([pose])[0]
    midpoint = [np.mean([box[0], box[2]]), np.mean([box[1], box[3]])]
    return np.array(midpoint)


@jit
def iou(bb_test, bb_gt):
    """
    Computes IUO between two bboxes in the form [x1,y1,x2,y2]
    """
    xx1 = np.maximum(bb_test[0], bb_gt[0])
    yy1 = np.maximum(bb_test[1], bb_gt[1])
    xx2 = np.minimum(bb_test[2], bb_gt[2])
    yy2 = np.minimum(bb_test[3], bb_gt[3])
    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    wh = w * h
    o = wh / ((bb_test[2] - bb_test[0]) * (bb_test[3] - bb_test[1])
              + (bb_gt[2] - bb_gt[0]) * (bb_gt[3] - bb_gt[1]) - wh)
    return (o)


def isMediaFile(fileName):
    mimestart = mimetypes.guess_type(fileName)[0]
    # print(mimestart)
    if mimestart != None:
        mimestart = mimestart.split('/')[0]

    # print(mimestart)

    return mimestart


def get_iou(bb1, bb2):
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.

    Parameters
    ----------
    bb1 : dict
        Keys: {'x1', 'x2', 'y1', 'y2'}
        The (x1, y1) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner
    bb2 : dict
        Keys: {'x1', 'x2', 'y1', 'y2'}
        The (x, y) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner

    Returns
    -------
    float
        in [0, 1]
    """
    if bb1['x1'] > bb1['x2']:
        bb1['x1'], bb1['x2'] = bb1['x2'], bb1['x1']
    if bb1['y1'] > bb1['y2']:
        bb1['y1'], bb1['y2'] = bb1['y2'], bb1['y1']
    if bb2['x1'] > bb2['x2']:
        bb2['x1'], bb2['x2'] = bb2['x2'], bb2['x1']
    if bb2['y1'] > bb2['y2']:
        bb2['y1'], bb2['y2'] = bb2['y2'], bb2['y1']

    # determine the coordinates of the intersection rectangle
    x_left = max(bb1['x1'], bb2['x1'])
    y_top = max(bb1['y1'], bb2['y1'])
    x_right = min(bb1['x2'], bb2['x2'])
    y_bottom = min(bb1['y2'], bb2['y2'])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # compute the area of both AABBs
    bb1_area = (bb1['x2'] - bb1['x1']) * (bb1['y2'] - bb1['y1'])
    bb2_area = (bb2['x2'] - bb2['x1']) * (bb2['y2'] - bb2['y1'])

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    assert iou >= 0.0
    assert iou <= 1.0
    return iou
