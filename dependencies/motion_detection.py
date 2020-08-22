import cv2

# Class to store parameters for motion detection
class MotionDetection:
    def __init__(self, status):
        self.status = status
        self.frame = None
        self.gray = None
        self.delta_frame = None
        self.thresh_frame = None

def motionDetector(frame, first_frame):
    """
    Performs object detection by comparison between the current frame and the previous frame.
    """

    detector = MotionDetection(status=0)

    # convert the color frame to gray frame as an extra layer of color
    # is not required
    detector.gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # convert gray scale image to GaussianBlur
    detector.gray = cv2.GaussianBlur(detector.gray, (21, 21), 0)

    # set first frame as the baseline frame
    if first_frame[0] is None:
        first_frame[0] = detector.gray
        return False, detector.status

    # calculate difference between static background and current frame
    detector.delta_frame = cv2.absdiff(first_frame[0], detector.gray)

    # apply the threshold
    detector.thresh_frame = cv2.threshold(detector.delta_frame, 30, 255, cv2.THRESH_BINARY)[1]

    # dilate the Threshold Frame and find pixel contours in it
    detector.thresh_frame = cv2.dilate(detector.thresh_frame, None, iterations=3)

    # find contours in the frame
    contours, _ = cv2.findContours(
        detector.thresh_frame.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        if cv2.contourArea(contour) < 10000:
            continue
        detector.status = 1

    return True, detector.status

