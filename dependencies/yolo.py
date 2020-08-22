import os
import numpy as np

# Class to store parameters for YOLO algorithm
class Yolo:
    def __init__(self, labels, colors, targets, weights_path, config_path):
        self.labels = labels
        self.colors = colors
        self.targets = targets
        self.weights_path = weights_path
        self.config_path = config_path

# Function to initialize YOLO algorithm parameters
def initialize(args):
    """
    Returns a object of type Yolo with the parameters initialized
    """

    # load the COCO class labels our YOLO model was trained on
    labels_path = os.path.sep.join([args.yolo, "coco.names"])
    LABELS = open(labels_path).read().strip().split("\n")

    # initialize a list of colors to represent each possible class label
    np.random.seed(42)
    COLORS = np.random.randint(0, 255, size=(len(LABELS), 3), dtype="uint8")

    # derive the paths to the YOLO weights and model configuration
    weights_path = os.path.sep.join([args.yolo, "yolov3.weights"])
    config_path = os.path.sep.join([args.yolo, "yolov3.cfg"])

    # list the target classes
    TARGETS = [0, 1, 2, 3, 5, 7]

    return Yolo(labels=LABELS, colors=COLORS, targets=TARGETS, weights_path=weights_path, config_path=config_path)
