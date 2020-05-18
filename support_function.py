import cv2,os,darknet
import numpy as np
import math,time

matrix_tranform = np.load("H_matrix.npy")

def showFrame(fps,frame,tic):
    # out.write(frame)
    #calculate FPS
    toc = time.time()
    curr_fps = 1.0 / (toc - tic)
    # calculate an exponentially decaying average of fps number
    fps = curr_fps if fps == 0.0 else (fps*0.95 + curr_fps*0.05)
    tic = toc
    fps_text = 'FPS: {:.2f}'.format(fps)
    cv2.putText(frame, fps_text, (0,30), cv2.FONT_HERSHEY_SIMPLEX,1, (0, 0, 0), 3)
    cv2.imshow("demo", frame)
    return fps,tic

def convertBack(x, y, w, h):
    xmin = int(round(x - (w / 2)))
    xmax = int(round(x + (w / 2)))
    ymin = int(round(y - (h / 2)))
    ymax = int(round(y + (h / 2)))
    return xmin, ymin, xmax, ymax

def cvDrawBoxes(detections, img,new_width,new_height):
    height,width = img.shape[:2]
    number_box = 0
    for detection in detections:
        if "person" in detection[0].decode():
            x, y, w, h = detection[2][0],\
                detection[2][1],\
                detection[2][2],\
                detection[2][3]
            xmin, ymin, xmax, ymax = convertBack(
                float(x), float(y), float(w), float(h))

            xmin = xmin*width/new_width
            ymin = ymin*height/new_height
            xmax = xmax*width/new_width
            ymax = ymax*height/new_height
            pt1 = (int(xmin), int(ymin))
            pt2 = (int(xmax), int(ymax))
            
            cv2.rectangle(img, pt1, pt2, (255, 0, 0), 2)
            number_box = number_box + 1
    return img,number_box

def load_model(metaMain, netMain, altNames,version):
    if version == 1:
        configPath = "./cfg/yolov4.cfg"
    else:
        configPath = "./cfg/small_yolov4.cfg"
    weightPath = "./weights/yolov4.weights"
    metaPath = "./cfg/coco.data"
    if not os.path.exists(configPath):
        raise ValueError("Invalid config path `" +
                         os.path.abspath(configPath) + "`")
    if not os.path.exists(weightPath):
        raise ValueError("Invalid weight path `" +
                         os.path.abspath(weightPath) + "`")
    if not os.path.exists(metaPath):
        raise ValueError("Invalid data file path `" +
                         os.path.abspath(metaPath) + "`")
    if netMain is None:
        netMain = darknet.load_net_custom(configPath.encode(
            "ascii"), weightPath.encode("ascii"), 0, 1)  # batch size = 1
    if metaMain is None:
        metaMain = darknet.load_meta(metaPath.encode("ascii"))
    if altNames is None:
        try:
            with open(metaPath) as metaFH:
                metaContents = metaFH.read()
                import re
                match = re.search("names *= *(.*)$", metaContents,
                                  re.IGNORECASE | re.MULTILINE)
                if match:
                    result = match.group(1)
                else:
                    result = None
                try:
                    if os.path.exists(result):
                        with open(result) as namesFH:
                            namesList = namesFH.read().strip().split("\n")
                            altNames = [x.strip() for x in namesList]
                except TypeError:
                    pass
        except Exception:
            pass
    # Create an image we reuse for each detect
    darknet_image = darknet.make_image(darknet.network_width(netMain),
                                       darknet.network_height(netMain), 3)
    return darknet_image,metaMain, netMain, altNames

def reference_point(src_point, matrix_tranform):
    x_src, y_src = src_point[0], src_point[1]
    x_des = (matrix_tranform[0][0] * x_src + matrix_tranform[0][1] * y_src +
             matrix_tranform[0][2]) / (
                    matrix_tranform[2][0] * x_src + matrix_tranform[2][
                1] * y_src + matrix_tranform[2][2])
    y_des = (matrix_tranform[1][0] * x_src + matrix_tranform[1][1] * y_src +
             matrix_tranform[1][2]) / (
                    matrix_tranform[2][0] * x_src + matrix_tranform[2][
                1] * y_src + matrix_tranform[2][2])
    x_des = int(x_des)
    y_des = int(y_des)
    return [x_des, y_des]

def is_near(point1, point2, distance=10):
    # print("point1",point1)
    # print("point2",point2)
    dis = math.sqrt(pow(point1[0] - point2[0], 2) + pow(point1[1] - point2[1], 2))
    print("dis",dis)
    return  dis< distance


def bb_intersection_over_union(boxA, boxB):
	# determine the (x, y)-coordinates of the intersection rectangle
	xA = max(boxA[0], boxB[0])
	yA = max(boxA[1], boxB[1])
	xB = min(boxA[2], boxB[2])
	yB = min(boxA[3], boxB[3])
	# compute the area of intersection rectangle
	interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
	# compute the area of both the prediction and ground-truth
	# rectangles
	boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
	boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
	# compute the intersection over union by taking the intersection
	# area and dividing it by the sum of prediction + ground-truth
	# areas - the interesection area
	iou = interArea / float(boxBArea)
	# return the intersection over union value
	return iou


def get_ref_box(box):
    x1,y1,x2,y2 = box
    (x1,y1) = reference_point((x1,y1),matrix_tranform)
    (x2, y2) = reference_point((x2, y2), matrix_tranform)
    return (x1,y1,x2,y2)

def boxes_filter(frame,detections,ori_width = 1280,ori_height=480,new_width=832,new_height=416):
    boxes = []
    for detection in detections:
        if "person" in detection[0].decode():
            x, y, w, h = detection[2][0],\
                detection[2][1],\
                detection[2][2],\
                detection[2][3]
            xmin, ymin, xmax, ymax = convertBack(
                float(x), float(y), float(w), float(h))
            xmin = int(xmin*ori_width/new_width)
            xmax = int(xmax*ori_width/new_width)
            ymin = int(ymin*ori_height/new_height)
            ymax = int(ymax*ori_height/new_height)
            boxes.append((xmin,ymin,xmax,ymax))
    length = len(boxes)
    if length == 1:
        return length
    boxes_left = []
    boxes_right = []
    for box in boxes:
        if box[0]<=ori_width//2:
            boxes_left.append(box)
        else:
            boxes_right.append(box)
    print("boxes_left",boxes_left)
    print("boxes_right",boxes_right)
    count_same = 0
    color_left = [0]*len(boxes_left)
    color_right = [0]*len(boxes_right)
    # for index_left, box_left in enumerate(boxes_left):
    #     if color_left[index_left]==0:
    #         max = [0,0,0]
    #         for index_right, box_right in enumerate(boxes_right):
    #             if color_right[index_right]==0:
    #                 (x1, y1, x2, y2) = box_right
    #                 # print("boxes_right", box_right)
    #                 box_right = (x1-ori_width//2, y1, x2-ori_width//2, y2)
    #                 box_right = get_ref_box(box_right)
    #                 (x1, y1, x2, y2) = box_right
    #                 cv2.rectangle(frame, (int(x1),int(y1)), (int(x2),int(y2)), (255, 0, 255), 2)
    #                 # print("boxes_left", box_left)
    #                 # print("boxes_right", box_right)
    #                 iou = bb_intersection_over_union(box_left,box_right)
    #                 print("iou",iou)
    #                 if iou>=0.1:
    #                     if iou > max[0]:
    #                         max[0] = iou
    #                         max[1] = index_right
    #                         max[2] = index_left
    #         if max[0] != 0:
    #             count_same = count_same + 1
    #             color_left[max[2]] = 1
    #             color_right[max[1]] = 1

    for index_left,box_left in enumerate(boxes_left):
        (x1, y1, x2, y2) = box_left
        pt_left = ((x1+x2)//2,(y1+y2)//2)
        for index_right,box_right in enumerate(boxes_right):
            (x1,y1,x2,y2) = box_right
            pt_right = ((x1+x2)//2-ori_width//2,(y1+y2)//2)
            pt_ref = reference_point(pt_right,matrix_tranform)
            if is_near(pt_left,pt_ref,distance=20):
                count_same = count_same + 1
                color_left[index_left] = 1
                color_right[index_right] = 1
            cv2.circle(frame, pt_left, 5, (0, 0, 0), thickness=-1)
            cv2.circle(frame, ((x1+x2)//2,(y1+y2)//2), 5, (0, 0, 255), thickness=-1)
            cv2.circle(frame, tuple(pt_ref), 5, (0, 255, 0), thickness=-1)

    ## draw boxes
    for index_left, box_left in enumerate(boxes_left):
        pt1 = box_left[:2]
        pt2 = (box_left[2],box_left[3])
        if color_left[index_left]:
            cv2.rectangle(frame, pt1, pt2, (0,0,255), 2)
        else:
            cv2.rectangle(frame, pt1, pt2, (0, 255, 0), 2)
    for index_left, box_left in enumerate(boxes_right):
        pt1 = box_left[:2]
        pt2 = (box_left[2],box_left[3])
        if color_right[index_left]:
            cv2.rectangle(frame, pt1, pt2, (0,0,255), 2)
        else:
            cv2.rectangle(frame, pt1, pt2, (0, 255, 0), 2)
    return length - count_same
