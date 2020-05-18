import math
import numpy as np
import cv2
import random


def point_inside_polygon(point, poly, include_edges=True):
    '''
    Check if point (x,y) is inside polygon poly.

    poly is N-vertices polygon defined as
    [[x1,y1],...,[xN,yN]] or [[x1,y1],...,[xN,yN],[x1,y1]]
    (function works fine in both cases)

    Geometrical idea: point is inside polygon if horisontal beam
    to the right from point crosses polygon even number of times.
    Works fine for non-convex polygons.
    '''
    n = len(poly)
    inside = False
    x, y = point[0], point[1]
    p1x, p1y = poly[0][0], poly[0][1]
    for i in range(1, n + 1):
        p2x, p2y = poly[i % n][0], poly[i % n][1]
        if p1y == p2y:
            if y == p1y:
                if min(p1x, p2x) <= x <= max(p1x, p2x):
                    # point is on horisontal edge
                    # inside = include_edges
                    inside = True
                    break
                elif x < min(p1x,
                             p2x):  # point is to the left from current edge
                    inside = not inside
        else:  # p1y!= p2y
            if min(p1y, p2y) <= y <= max(p1y, p2y):
                xinters = (y - p1y) * (p2x - p1x) / float(p2y - p1y) + p1x

                if x == xinters:  # point is right on the edge
                    # inside = include_edges
                    inside = True
                    break

                if x < xinters:  # point is to the left from current edge
                    inside = not inside

        p1x, p1y = p2x, p2y

    return inside


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
    return [x_des, y_des]


def is_near(point1, point2, distance=10):
    # print("distance ", math.sqrt(pow(point1[0] - point2[0], 2) + pow(point1[1] - point2[1], 2)))
    return math.sqrt(pow(point1[0] - point2[0], 2) + pow(point1[1] - point2[1], 2)) < distance


def is_the_same(left_point, right_point, matrix_tranform, width=640, limit_distance=15):
    right_point = list(right_point)
    right_point = reference_point(right_point, matrix_tranform)
    if is_near(left_point, right_point, limit_distance):
        return True
    return False


height, width = 480, 640
matrix_tranform = np.load('H_matrix.npy')
number_of_colors = 100
mycolor = [(122, 200, 231), (229, 13, 78), (32, 5, 155), (80, 57, 55), (153, 109, 99), (15, 18, 58), (180, 61, 198),
           (219, 128, 28), (218, 8, 162), (173, 70, 219), (115, 147, 104), (185, 82, 236), (252, 28, 89),
           (205, 192, 78), (155, 24, 134), (157, 153, 176), (86, 25, 101), (201, 239, 113), (179, 215, 149),
           (48, 47, 122), (84, 139, 4), (50, 52, 201), (72, 39, 77), (224, 144, 18), (32, 81, 159), (206, 82, 181),
           (32, 6, 179), (154, 149, 92), (9, 26, 108), (250, 24, 82), (138, 14, 207), (135, 254, 43), (255, 2, 233),
           (203, 201, 7), (16, 229, 219), (220, 193, 182), (209, 157, 230), (158, 59, 116), (195, 60, 57),
           (78, 146, 228), (72, 161, 91), (239, 136, 31), (226, 59, 252), (87, 153, 21), (37, 158, 48), (85, 226, 129),
           (133, 199, 67), (200, 60, 37), (208, 217, 100), (224, 35, 203), (142, 194, 35), (236, 173, 104),
           (165, 55, 209), (109, 65, 53), (145, 59, 2), (82, 122, 207), (191, 106, 227), (202, 87, 115), (197, 95, 177),
           (66, 216, 81), (221, 181, 71), (231, 167, 102), (172, 2, 202), (9, 136, 139), (95, 139, 143), (213, 231, 28),
           (147, 81, 2), (109, 187, 210), (170, 57, 57), (238, 126, 191), (3, 42, 134), (210, 168, 28), (217, 159, 28),
           (181, 147, 199), (173, 70, 91), (48, 237, 21), (55, 94, 15), (163, 226, 244), (222, 32, 165),
           (181, 125, 122), (236, 177, 67), (105, 81, 18), (88, 13, 9), (21, 140, 32), (193, 8, 104), (19, 116, 8),
           (24, 188, 3), (39, 63, 13), (237, 112, 165), (216, 145, 45), (201, 10, 74), (26, 17, 93), (22, 84, 108),
           (212, 196, 46), (133, 57, 221), (31, 88, 89), (225, 229, 52), (32, 155, 203), (185, 56, 85),
           (174, 54, 207), ]


def visual_overlap(frame, points):
    for index, point in enumerate(points):
        cv2.circle(frame, tuple(point), radius=5, color=(0, 255, 0), thickness=-1)
        if index == 3:
            cv2.line(frame, tuple(point), tuple(points[0]), color=(0, 0, 255), thickness=2)
            # cv2.line(frame, tuple(point), tuple(points[0]), color=255, thickness=2)
            continue
        cv2.line(frame, tuple(point), tuple(points[index + 1]), color=(0, 0, 255), thickness=2)
        # cv2.line(frame, tuple(point), tuple(points[index + 1]), color=255,thickness=2)

def visual_one_tracker(tracker, frame, this_color,f,frame_counter,new_width,new_height,step_frame):
    (ori_height,ori_width) = frame.shape[:2]
    count = 0
    for track in tracker.tracks:
        if track.time_since_update > 1:
            continue
        count = count + 1
        this_color = mycolor[track.track_id%100]
        bbox = track.to_tlbr()
        (x1, y1, x2, y2) = bbox
        x1 = int(x1*ori_width/new_width)
        x2 = int(x2*ori_width/new_width)
        y1 = int(y1*ori_height/new_height)
        y2 = int(y2*ori_height/new_height)
        cv2.rectangle(frame, (x1,y1), (x2,y2), this_color, 2)
        center_point = ((x1 + x2) // 2, (y1 + y2) // 2)
        cv2.circle(frame, center_point, radius=5, color=(0, 255, 0), thickness=-1)
        cv2.putText(frame, str(track.track_id), center_point, 0, 5e-3 * 200, (0, 255, 0), 2)
    if f is not None:
        f.write(str(frame_counter) + ".jpg" + "-" + str(count) + "\n")


def visual_tracker(tracker, frame, this_color):
    for track in tracker.tracks:
        # if not track.is_confirmed() or track.time_since_update > 1:
        #     continue
        bbox = track.to_tlbr()
        (x1, y1, x2, y2) = bbox
        (x1, y1, x2, y2) = (int(x1), int(y1), int(x2), int(y2))
        if track.in_overlap:
            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), mycolor[track.father_id],2)
        else:
            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), this_color, 2)
        center_point = ((x1 + x2) // 2, (y1 + y2) // 2)
        cv2.circle(frame, center_point, radius=5, color=(0, 255, 0), thickness=-1)
        cv2.putText(frame, str(track.father_id), ((x1 + x2) // 2, y2), 0, 1, (0, 255, 0), 2)
        cv2.putText(frame, str(track.track_id), ((x1 + x2) // 2, y1), 0, 5e-3 * 200, (0, 255, 0), 2)


def get_center(track, right=False):
    (x1, y1, x2, y2) = track.to_tlbr()
    if not right:
        point = ((x1 + x2) // 2, (y1 + y2) // 2)
    else:
        point = ((x1 + x2) // 2 - 640, (y1 + y2) // 2)
    return point


## check inside polygon
def check_inside_polygon(tracker, overlap_poly, right=False):
    for index, track in enumerate(tracker.tracks):
        point = get_center(track, right)
        if not point_inside_polygon(point, overlap_poly):
            tracker.tracks[index].in_overlap = False
        else:
            tracker.tracks[index].in_overlap = True


def allocate_father_id(tracker_left, tracker_right, id):
    for index, track in enumerate(tracker_left.tracks):
        if track.father_id is None and track.in_overlap is False:
            tracker_left.tracks[index].father_id = id
            id = id + 1
    for index, track in enumerate(tracker_right.tracks):
        if track.father_id is None and track.in_overlap is False:
            tracker_right.tracks[index].father_id = id
            id = id + 1
    for index_left, track_left in enumerate(tracker_left.tracks):
        if track_left.in_overlap:
            center_left = get_center(track_left)
            for index_right, track_right in enumerate(tracker_right.tracks):
                if track_right.in_overlap:
                    center_right = get_center(track_right, right=True)
                    # print("index left ", index_left, "right", index_right)
                    if is_the_same(center_left, center_right, matrix_tranform, limit_distance=30):
                        ## refer left here
                        if track_left.father_id is not None and track_right.father_id is None:
                            tracker_right.tracks[index_right].father_id = track_left.father_id
                        elif track_left.father_id is None and track_right.father_id is not None:
                            tracker_left.tracks[index_left].father_id = track_right.father_id
                        elif track_left.father_id is not None and track_right.father_id is not None:
                            tracker_right.tracks[index_right].father_id = track_left.father_id
                        else:
                            tracker_left.tracks[index_left].father_id = id
                            tracker_right.tracks[index_right].father_id = id
                            id = id + 1
                    ## if not the same => allocate new
                    if track_left.father_id is None:
                        tracker_left.tracks[index_left].father_id = id
                        id = id + 1
                    if track_right.father_id is None:
                        tracker_right.tracks[index_right].father_id = id
                        id = id + 1
    return id
