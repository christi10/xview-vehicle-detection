from pathlib import Path
import cv2


def draw_bbox(img, bbox, color=(0,255,0), thickness=2):
    x, y, w, h = bbox
    pt1 = (int(x), int(y))
    pt2 = (int(x + w), int(y + h))
    cv2.rectangle(img, pt1, pt2, color, thickness)
    return img
