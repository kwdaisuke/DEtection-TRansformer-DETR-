import numpy as np
from PIL import Image
import tensorflow as tf


def frame(boxes):
    cx, cy, w, h = [boxes[..., i] for i in range(4)]
    
    xmin, ymin = cx - w*0.5, cy - h*0.5
    xmax, ymax = cx + w*0.5, cy + h*0.5
    
    boxes = tf.stack([xmin, ymin, xmax, ymax], axis=-1)
    return boxes