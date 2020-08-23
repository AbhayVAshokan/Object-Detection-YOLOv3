# Object Detection using YOLOv3

> A website to provide an interface for object detection algorithm using YOLOv3.

+ The input video is first checked for motion detection. If motion is detected, every second frame is processed using YOLOv3 object detection algorithm loaded with pre-trained weights. The processed video is stored in the path specified. 

+ Start button sends an API request to the `NodeJS` server which opens a virtual python shell and runs the python code corresponding to the object detection.

+ Stop button send another API request to force stop the running python shell.

+ The Object detection algorithm loads `videos/cctv_footage.mp4` and `yolo-coco/yolov3.weights` and writes to `output/Test.avi` by default unless specified otherwise. In addition, confidence value and threshold value can also be specified which are kept at `0` by default.

**Screenshot**

<img src="https://raw.githubusercontent.com/abilash-sajeev/Object-Detection-YOLOv3/master/.github/screenshot.png" width=75%>

### Setup
1. The link contains the pretrained weights executing the YOLOv3 object detection algorithm: [pre-trained weights](https://pjreddie.com/media/files/yolov3.weights)

    Download the file and copy it into your `yolo-coco` folder.
<br>

2. Install the following dependencies
    ```
    apt-get install nodejs
    apt-get install npm
    pip3 install opencv-python
    ```
<br>

3. Install the `npm` dependencies.
    ```
    npm install
    ```
<br>

4. Run the NodeJS server.
    ```
    node app
    ```

The website is available at `localhost:3000`.
<br><br>

**Python Implementation (without UI)**
In order to run object detection algorithm without user interface execute the following command.

```
python3 detect.py
```
<br>
<p> The following command gives more control over the parameters. </p>

```
python3 detect.py --input videos/cctv_.mp4 --output outputs/Test.avi --yolo yolo-coco/yolov3.weights --confidence 0 --threshold 0
```

<br>

### Generated folders
**frames**: Stores frame numbers of images processed after motion detection.

**snapshots**: Stores images in which the detection is performed.

**time**: Stores the timestamps of frames in `frames` directory.

**output**: Stores the output video with detection performed.
<br><br>

### Important notes

1. The project has been tested and developed on `Chrome` browser. We cannot guarantee that it works in other browsers as well.

2. The API request times out in exactly 2 minutes. If the python code executes more more than 2 minutes on a video, it cannot generate a response. Use short videos and test using the python code directly if running time exceeds 2 minutes.
