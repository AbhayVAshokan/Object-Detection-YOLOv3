# Usage: python3 detect.py --input videos/test.mp4 --output output/test.avi --yolo yolo-coco

# import required libraries
import cv2
import time
import pandas
import numpy as np
import argparse
import imutils
import os
from imutils.video import FPS
from datetime import datetime

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=True,
                help="path to input video")
ap.add_argument("-o", "--output", required=True,
                help="path to output video")
ap.add_argument("-y", "--yolo", required=True,
                help="base path to YOLO directory")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
                help="minimum probability to filter weak detections")
ap.add_argument("-t", "--threshold", type=float, default=0.3,
                help="threshold when applyong non-maxima suppression")
args = vars(ap.parse_args())

# create required directories
dir = args["input"].split("/")[1].split(".")[0]
if not os.path.exists('./snapshots/'+dir):
    os.makedirs('./snapshots/'+dir)
if not os.path.exists('./output/'):
    os.makedirs('./output')
if not os.path.exists('./time/'):
    os.makedirs('./time')
if not os.path.exists('./frames/'):
    os.makedirs('./frames')

# load the COCO class labels our YOLO model was trained on
labels_path = os.path.sep.join([args["yolo"], "coco.names"])
LABELS = open(labels_path).read().strip().split("\n")

# initialize a list of colors to represent each possible class label
np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3), dtype="uint8")

# derive the paths to the YOLO weights and model configuration
weights_path = os.path.sep.join([args["yolo"], "yolov3.weights"])
config_path = os.path.sep.join([args["yolo"], "yolov3.cfg"])

# start the FPS timer
timer = FPS().start()

# load our YOLO object detector trained on COCO dataset (80 classes)
# and determine only the output layer names that we need from YOLO
print("[INFO] loading YOLO from disk...")
net = cv2.dnn.readNetFromDarknet(config_path, weights_path)
ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# list the target classes
TARGETS = [0, 1, 2, 3, 5, 7]

# assign our first frame to None
first_frame = None

# list to track movement
status_list = [None, None]

# list to store time stamp and frame number of movement
time_stamp = []
frames = []

# initialize dataframes for storing time and frame numbers indicating
# start and end of movements
time_of_movements = pandas.DataFrame(columns=["Start", "End"])
frames_of_movements = pandas.DataFrame(columns=["Start", "End"])

# initialise other variables
frame_count = 0
flag = False
factor = 2

# initialize the video stream, pointer to output video file and
# frame dimensions
video = cv2.VideoCapture(args["input"])
writer = None
(W, H) = (None, None)

# check if the video capture is successful
if not video.isOpened():
    print("Error opening the video")

# loop over frames from the video stream
while True:
    # read the next frame from the file
    ret, frame = video.read()

    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    # determine the fps of the video
    fps = int(video.get(cv2.CAP_PROP_FPS))
    fps += fps % 2

    # increment the frame count
    frame_count += 1

    # if the frame dimensions are empty, grab them
    if W is None or H is None:
        (H, W) = frame.shape[:2]

    # initialize status = 0 (no movements)
    status = 0

    # convert the color frame to gray frame as an extra layer of color
    # is not required
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # convert gray scale image to GaussianBlur
    gray = cv2.GaussianBlur(gray, (21, 21), 0)

    # set first frame as the baseline frame
    if first_frame is None:
        first_frame = gray
        continue

    # calculate difference between static background and current frame
    delta_frame = cv2.absdiff(first_frame, gray)

    # apply the threshold
    thresh_frame = cv2.threshold(delta_frame, 30, 255, cv2.THRESH_BINARY)[1]

    # dilate the Threshold Frame and find pixel contours in it
    thresh_frame = cv2.dilate(thresh_frame, None, iterations=3)

    # find contours in the frame
    contours, _ = cv2.findContours(
        thresh_frame.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        if cv2.contourArea(contour) < 10000:
            continue
        status = 1

    # append status of movements
    status_list.append(status)

    status_list = status_list[-2:]

    # append time and frame number for the start of movements
    if status_list[-1] == 1 and status_list[-2] == 0:
        time_stamp.append(datetime.now())
        frames.append(frame_count)
        flag = True

        # perform detection if there is movement
    if flag and not ((frame_count - frames[-1]) % factor):
                # Construct a blob from the input frame and then perform
                # a forward pass of the YOLO object detector to get the
                # bounding boxes and associated probabilities
        blob = cv2.dnn.blobFromImage(
            frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
        net.setInput(blob)
        layerOutputs = net.forward(ln)

        # initialize the lists for detected bounding boxes, confidences
        # and class IDs respectively
        boxes = []
        confidences = []
        classIDs = []

        # loop over each of the layer outputs
        for output in layerOutputs:
                        # loop over each of the detections
            for detection in output:
                                # extract the class ID and confidence (i.e., probability)
                                # of the current object detection
                scores = detection[5:]
                classID = np.argmax(scores)
                if classID not in TARGETS:
                    continue
                confidence = scores[classID]

                # filter out weak predictions by ensuring the detected
                # probability is greater than the minimum probability
                if confidence > args["confidence"]:
                                        # scale the bounding box coordinates back relative to
                                        # the size of the image
                    box = detection[0:4] * np.array([W, H, W, H])
                    (centerX, centerY, width, height) = box.astype("int")

                    # use the center (x, y)-coordinates to derive the top
                    # and and left corner of the bounding box
                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))

                    # update the list of bounding box coordinates,
                    # confidences, and class IDs
                    boxes.append([x, y, int(width), int(height)])
                    confidences.append(float(confidence))
                    classIDs.append(classID)

                # apply non-maxima suppression to suppress weak, overlapping
                # bounding boxes
        idxs = cv2.dnn.NMSBoxes(
            boxes, confidences, args["confidence"], args["threshold"])

        # ensure at least one detection exists
        if len(idxs) > 0:
                    # loop over the indexes
            for i in idxs.flatten():
                                # extract the bounding box coordinates
                (x, y) = (boxes[i][0], boxes[i][1])
                (w, h) = (boxes[i][2], boxes[i][3])

                # draw a bounding box rectangle and label on the frame
                color = [int(c) for c in COLORS[classIDs[i]]]
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])
                cv2.putText(frame, text, (x, y - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                # save snapshots
        if (frame_count - frames[-1]) in [0, fps, 2*fps]:
            print("Saving snapshot -> {}".format(frame_count))
            cv2.imwrite(os.path.sep.join(
                ["snapshots", dir, "frame_{}.jpg".format(frame_count)]), frame)

            # check if the video writer is None
        if writer is None:
                    # initialize our video writer
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            writer = cv2.VideoWriter(
                args["output"], fourcc, fps//2, (frame.shape[1], frame.shape[0]), True)

            # write the output frame to disk
        print("Writing to output -> {}".format(frame_count))
        writer.write(frame)

        # append the time and frame number for the end of movements
    if status_list[-1] == 0 and status_list[-2] == 1:
        time_stamp.append(datetime.now())
        frames.append(frame_count)
        flag = False

        # if q entered whole process will stop
    if cv2.waitKey(1) == ord('q'):
                # if something is moving then append time and frame number
                # for the end of movements
        if status == 1:
            time_stamp.append(datetime.now())
            frames.append(frame_count)
            flag = False
        break

        # update the timer
    timer.update()

# append time and frame numbers of movements in dataframe
for i in range(0, len(time_stamp)-1, 2):
    time_of_movements = time_of_movements.append(
        {"Start": time_stamp[i], "End": time_stamp[i + 1]}, ignore_index=True)
    frames_of_movements = frames_of_movements.append(
        {"Start": frames[i], "End": frames[i + 1]}, ignore_index=True)

# create a CSV file in which time and frame numbers of movements
# will be saved
time_of_movements.to_csv(os.path.sep.join(["time", "{}.csv".format(dir)]))
frames_of_movements.to_csv(os.path.sep.join(["frames", "{}.csv".format(dir)]))

# stop the timer and display FPS information
timer.stop()
print("[INFO] elasped time: {:.2f}".format(timer.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(timer.fps()))

# when everything done, release the capture and destroy all the windows
print("[INFO] cleaning up...")
video.release()
cv2.destroyAllWindows()
