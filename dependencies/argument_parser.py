# Class to store command line arguments
class Arguments:
    def __init__(self, input, output, yolo, confidence, threshold):

        # Initializing default values
        if input == None:
            input = 'videos/test.mp4'
        if output == None:
            output = 'output/test.avi'
        if yolo == None:
            yolo = 'yolo-coco'
        if confidence == None:
            confidence = 0.0
        if threshold == None:
            confidence = 0.0

        self.input = input
        self.output = output
        self.yolo = yolo
        self.confidence = confidence
        self.threshold = 0

# Function to parse command line arguments
def parseArguments(ap):
    """
    Parses command line arguments and returns an object of type Arguments.
    """
    
    ap.add_argument("-i", "--input",
                    help="path to input video")
    ap.add_argument("-o", "--output",
                    help="path to output video")
    ap.add_argument("-y", "--yolo",
                    help="base path to YOLO directory")
    ap.add_argument("-c", "--confidence", type=float, default=0.5,
                    help="minimum probability to filter weak detections")
    ap.add_argument("-t", "--threshold", type=float, default=0.3,
                    help="threshold when applyong non-maxima suppression")
    args = vars(ap.parse_args())

    return Arguments(input=args["input"], output=args["output"], yolo=args["yolo"], confidence=args["confidence"], threshold=args["threshold"])
