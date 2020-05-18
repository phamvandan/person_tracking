#! /usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function, absolute_import

import warnings
import cv2
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet
import time
from track_function import visual_one_tracker

warnings.filterwarnings('ignore')
import sys
import darknet
from support_function import convertBack,showFrame,load_model
import os

def get_detection(frame, boxs, encoder):
    features = encoder(frame, boxs)
    detections = [Detection(bbox, 1.0, feature) for bbox, feature in zip(boxs, features)]
    return detections

netMain = None
metaMain = None
altNames = None

## per step_frame detect one time
step_frame = 1
thresh = 0.8
## big version 992*544
# version = 1
## small version 576*320
version = 2
video3 = False

write_flag = True


def main():
    ## khoang cach cosine
    max_cosine_distance = 0.9
    nn_budget = None
    model_filename = 'model_data/mars-small128.pb'
    encoder = gdet.create_box_encoder(model_filename, batch_size=1)
    metric_left = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    tracker_left = Tracker(metric_left)
    video_capture = cv2.VideoCapture(sys.argv[1])
    frame_count = -1
    tic = time.time()
    fps = 0.0
    sum = 0
    detections = None
    ## load YOLO model
    global metaMain, netMain, altNames
    darknet_image, metaMain, netMain, altNames = load_model(metaMain, netMain, altNames, version)

    if write_flag:
        folder_name = sys.argv[1]+""
        folder_name = folder_name[:len(folder_name)-4]
        folder_name = folder_name + "_tracked"
        if not os.path.exists(folder_name):
            os.mkdir(folder_name)
        folder_name = folder_name + "/"
        print(folder_name)
        frame_width = int(video_capture.get(3))
        frame_height = int(video_capture.get(4))
        # Define the codec and create VideoWriter object.The output is stored in 'outpy.avi' file.
        out = cv2.VideoWriter(folder_name + 'output_tracked.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (frame_width,frame_height))
        f = open(folder_name + "person_count_tracked.txt","w+")
        f2 = open(folder_name + "fps_tracked.txt","w+")
    else:
        f = None
    frame_counter = -1

    while True:
        ret, frame = video_capture.read()  # frame shape 640*480*3
        if ret != True:
            if write_flag:
                fps = frame_counter/sum
                # calculate an exponentially decaying average of fps number
                f2.write(str(fps) + "\n")
            break
        visual_frame = frame.copy()
        t1 = time.time()
        frame_count = frame_count +1
        frame_counter = frame_counter + 1
        tic = time.time()
        frame = cv2.resize(frame, (darknet.network_width(netMain),
                                   darknet.network_height(netMain)),
                           interpolation=cv2.INTER_LINEAR)
        if frame_count % step_frame == 0:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_resized = cv2.resize(frame_rgb,(darknet.network_width(netMain),
                                        darknet.network_height(netMain)),
                                    interpolation=cv2.INTER_LINEAR)
            darknet.copy_image_from_bytes(darknet_image,frame_resized.tobytes())
            detections = darknet.detect_image(netMain, metaMain, darknet_image, thresh=thresh)
            boxes = []
            for detection_left in detections:
                if "person" in str(detection_left[0]):
                    x, y, w, h = detection_left[2][0],\
                        detection_left[2][1],\
                        detection_left[2][2],\
                        detection_left[2][3]
                    xmin, ymin, xmax, ymax = convertBack(
                        int(x), int(y), int(w), int(h))
                    boxes.append((xmin,ymin,xmax-xmin,ymax-ymin))
            detections = get_detection(frame, boxes, encoder)
            frame_count = 0
        # Call the tracker
        tracker_left.predict()
        tracker_left.update(detections)
        toc = time.time()
        sum = sum + toc - tic
        # if write_flag and ((video3 and frame_counter % 30 == 0) or (video3 is False and frame_counter % 25 == 0)):
        if write_flag:
            visual_one_tracker(tracker_left,visual_frame,(255,0,0),f,frame_counter,darknet.network_width(netMain),darknet.network_height(netMain),step_frame)
        else:
            visual_one_tracker(tracker_left,visual_frame,(255,0,0),None,frame_counter,darknet.network_width(netMain),darknet.network_height(netMain),step_frame)
        fps_text = 'FPS: {:.2f}'.format(fps)
        cv2.putText(visual_frame, fps_text, (0, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)
        cv2.imshow("demo", visual_frame)

        if write_flag:
            out.write(visual_frame)
        key = cv2.waitKey(1)
        if key==ord('q'):
            break
        elif key==32:
            key = cv2.waitKey(0)
            if key == ord('s'):
                cv2.imwrite("frame_read.jpg",visual_frame)

    video_capture.release()
    cv2.destroyAllWindows()



if __name__ == '__main__':
    main()
