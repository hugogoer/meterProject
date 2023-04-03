import sys
import argparse
import torch.backends.cudnn as cudnn
import cv2 as cv
import numpy as np
import math
import os
from scipy import spatial
from numpy.linalg import det
from utils.datasets import *
from utils.utils import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5 import QtCore, QtGui, QtWidgets
# ./inference/images
# ./Identification results folder
def distance(x1, y1, x2, y2):
    return math.sqrt((x1-x2)**2 + (y1 - y2)**2)

def resizeimg(img):
    height, width, channels = img.shape
    if width > 1500 or width < 600:
        scale = 500 / width
        ui.printf("图片的尺寸由 %dx%d, 调整到 %dx%d" % (width, height, width * scale, height * scale))
        scaled = cv.resize(img, (0, 0), fx=scale, fy=scale)
        return scaled,scale

def preliminary_pretreatment(img):
    dst = cv.pyrMeanShiftFiltering(img, 10, 100)
    cimage = cv.cvtColor(dst, cv.COLOR_BGR2GRAY)
    return cimage

def center(pre):
    height, width = pre.shape[0],pre.shape[1]
    circles, param, x, y, r = [], 50, 0, 0, 0
    while 1:
        circles = cv.HoughCircles(pre, cv.HOUGH_GRADIENT, 1, 20, param1=100, param2=param, minRadius=100, maxRadius=300)
        if circles is None:
            param = param - 5
            continue
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            if i[2] > r and i[2] < width / 2:
                r = i[2]
                x = i[0]
                y = i[1]
        break

    ui.printf("表盘大致位置已找到，坐标为：（ %d, %d ），半径为： %d" % (x,y,r))


    return x,y,r

def create_hue_mask(image, lower_color, upper_color):
    lower = np.array(lower_color, np.uint8)
    upper = np.array(upper_color, np.uint8)
    mask = cv.inRange(image, lower, upper)
    return mask

def findcolor(img,h0,s0,v0,h1,s1,v1,h2,s2,v2,h3,s3,v3):
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    lower_color_hue = create_hue_mask(hsv, [h0,s0,v0],[h1,s1,v1])
    higher_color_hue = create_hue_mask(hsv, [h2,s2,v2],[h3,s3,v3])
    mask = cv.bitwise_or(lower_color_hue, higher_color_hue)
    output = cv.bitwise_and(img, img, mask = mask)
    return output

def pretreatment(img):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    img_thresh = cv.GaussianBlur(gray, (5, 5), 0)
    kernel = np.ones((3, 3), np.float32) / 25#5-3
    img_thresh = cv.filter2D(img_thresh, -1, kernel)
    edges = cv.Canny(img_thresh, 15, 100, apertureSize=3)#24,80  10,80  10,90
    Matrix = np.ones((2, 2), np.uint8)
    img_edge = cv.morphologyEx(edges, cv.MORPH_CLOSE, Matrix)
    return img_edge

def find_farpoint(point):
    point = np.array(point)
    candidates = point[spatial.ConvexHull(point).vertices]
    dist_mat = spatial.distance_matrix(candidates, candidates)
    p1, p2 = np.unravel_index(dist_mat.argmax(), dist_mat.shape)
    return (candidates[p1][0], candidates[p1][1]),(candidates[p2][0], candidates[p2][1])

def find_scale1(img_copy1,x,y,r):
    for i in range(img_copy1.shape[0]):
        for j in range(img_copy1.shape[1]):
            if distance(j, i, x, y) >= r:
                img_copy1[i, j] = 0
    output1 = findcolor(img_copy1, 50, 170, 46, 77, 255, 255, 50, 170, 46, 77, 255, 255)  # green
    point_g = []
    for i in range(output1.shape[0]):
        for j in range(output1.shape[1]):
            if output1[i, j].all() > 0:
                point_g.append((j, i))
    point_g = np.array(point_g)
    candidates = point_g[spatial.ConvexHull(point_g).vertices]
    dist_mat = spatial.distance_matrix(candidates, candidates)
    p1, p2 = np.unravel_index(dist_mat.argmax(), dist_mat.shape)
    scale = distance(candidates[p1][0], candidates[p1][1], candidates[p2][0], candidates[p2][1])
    return candidates,p1,p2,scale

def findEllipse(x,y,r,c1,c2,scale,candidates,p1,p2,img,img_copy):
    contours, hierarchy = cv.findContours(img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    X, Y, ma, MA,angle = 9999, 9999, 9999, 9999, 9999
    height, width, channels = img_copy.shape
    for ind, cont in enumerate(contours):
        if (len(cont) > 5):
            (X0, Y0), (MA0, ma0), angle0 = cv.fitEllipse(cont)
            mindistance = min(distance(X0,Y0,candidates[p1][0],candidates[p1][1]),distance(X0,Y0,candidates[p2][0],candidates[p2][1]))
            if ma0 < min(width,height) and MA0 < max(width,height) and distance(X0,Y0,int((c1[0]+c2[0])/2),int((c1[1]+c2[1])/2)) < 1/2*distance(c1[0],c1[1],c2[0],c2[1]) and distance(X0, Y0, x, y) < 1 / 2 * r  and mindistance <= scale and max(ma0,MA0) >= 2.5*scale and max(ma0,MA0) <= 4*scale and min(ma0,MA0) >= 2*scale and max(ma0,MA0) <= max(ma,MA):
                X, Y, MA, ma, angle = X0, Y0, MA0, ma0, angle0
    ui.printf("表盘详细位置已找到，椭圆中心坐标为：（ %d, %d ），长轴为： %d，短轴为： %d，倾斜角度为： %.2f°" % (X,Y,max(ma,MA),min(ma,MA),angle))
    return X,Y,MA,ma,angle,img_copy

def findvertex(img_copy,X,Y,MA,ma,angle):
    points = []
    img1 = np.zeros((img_copy.shape[0], img_copy.shape[1]), dtype=np.uint8)
    cv.ellipse(img1, (int(X), int(Y)), (int(MA / 2), int(ma / 2)), angle, 0, 360, (255, 255, 255), 2)
    img2 = np.zeros((img_copy.shape[0], img_copy.shape[1]), dtype=np.uint8)
    cv.line(img2, (int(X - math.cos(angle) * ma), int(Y + math.sin(angle) * ma)),
            (int(X + math.cos(angle) * ma), int(Y - math.sin(angle) * ma)), (255, 255, 255), 1)
    cv.line(img2, (int(X + math.sin(angle) * MA), int(Y + math.cos(angle) * MA)),
            (int(X - math.sin(angle) * MA), int(Y - math.cos(angle) * MA)), (255, 255, 255), 1)
    for i in range(img_copy.shape[0]):
        for j in range(img_copy.shape[1]):
            if img1[i, j] > 0 and img2[i, j] > 0:
                points.append((j, i))
    point = list([])
    n = points[0][0]
    for i in range(len(points)):
        if abs(points[i][0] - n) > 2:
            point.append(points[i])
            n = points[i][0]
    point.append(points[0])
    img3 = np.zeros((img_copy.shape[0], img_copy.shape[1]), dtype=np.uint8)
    cv.ellipse(img3, (int(X), int(Y)), (int(MA / 2), int(ma / 2)), angle, 0, 360, (255, 255, 255), -1)
    for i in range(img_copy.shape[0]):
        for j in range(img_copy.shape[1]):
            if img3[i, j] == 0:
                img_copy[i,j] = 255
    order = []
    order.append(point[np.argmin(point, axis=0)[1]])
    order.append(point[np.argmax(point, axis=0)[1]])
    order.append(point[np.argmin(point, axis=0)[0]])
    order.append(point[np.argmax(point, axis=0)[0]])
    return img_copy,order

def perspective_transformation(img_copy,point):
    w = min(img_copy.shape[0], img_copy.shape[1])
    pts1 = np.float32([[point[0][0], point[0][1]], [point[1][0], point[1][1]], [point[2][0], point[2][1]],
                       [point[3][0], point[3][1]]])

    pts2 = np.float32([[w / 2, 0], [w / 2, w], [0, w / 2], [w, w / 2]])
    M = cv.getPerspectiveTransform(pts1, pts2)
    dst = cv.warpPerspective(img_copy, M, (w, w))
    for i in range(dst.shape[0]):
        for j in range(dst.shape[1]):
            if dst[i, j].all() == 0 and distance(i,j,w/2,w/2) > w/2:
                dst[i, j] = 255
            if distance(i, j, w / 2, w / 2) > w / 2:
                dst[i, j] = 255
    return dst

def find_scale(dst):
    w = min(dst.shape[0],dst.shape[1])
    for i in range(dst.shape[0]):
        for j in range(dst.shape[1]):
            if distance(i,j,w/2,w/2) > (w/2)*3/4:
                dst[i, j] = 0
    output0 = findcolor(dst,11, 245, 46, 30, 255, 255,   11, 245, 46, 30, 255, 255)#yellow
    output1 = findcolor(dst,50, 80, 46, 77, 255, 255,    50, 80, 46, 77, 255, 255)#green
    output3 = findcolor(dst,175, 210, 46, 180, 255, 250,    0, 210, 46, 10, 255, 250)#red
    output4 = cv.bitwise_or(output0,output1);output = cv.bitwise_or(output3,output4)
    return output,output0,output1,output3

def points2circle(p1, p2, p3):
    p1 = np.array(p1)
    p2 = np.array(p2)
    p3 = np.array(p3)
    num1 = len(p1)
    num2 = len(p2)
    num3 = len(p3)
    if (num1 == num2) and (num2 == num3):
        if num1 == 2:
            p1 = np.append(p1, 0)
            p2 = np.append(p2, 0)
            p3 = np.append(p3, 0)
        elif num1 != 3:
            print('\t仅支持二维或三维坐标输入')
            return None
    else:
        print('\t输入坐标的维数不一致')
        return None
    temp01 = p1 - p2
    temp02 = p3 - p2
    temp03 = np.cross(temp01, temp02)
    temp = (temp03 @ temp03) / (temp01 @ temp01) / (temp02 @ temp02)
    if temp < 10**-6:
        print('\t三点共线, 无法确定圆')
        return None
    temp1 = np.vstack((p1, p2, p3))
    temp2 = np.ones(3).reshape(3, 1)
    mat1 = np.hstack((temp1, temp2))
    m = +det(mat1[:, 1:])
    n = -det(np.delete(mat1, 1, axis=1))
    p = +det(np.delete(mat1, 2, axis=1))
    q = -det(temp1)
    temp3 = np.array([p1 @ p1, p2 @ p2, p3 @ p3]).reshape(3, 1)
    temp4 = np.hstack((temp3, mat1))
    temp5 = np.array([2 * q, -m, -n, -p, 0])
    mat2 = np.vstack((temp4, temp5))
    A = +det(mat2[:, 1:])
    B = -det(np.delete(mat2, 1, axis=1))
    C = +det(np.delete(mat2, 2, axis=1))
    D = -det(np.delete(mat2, 3, axis=1))
    E = +det(mat2[:, :-1])
    pc = -np.array([B, C, D]) / 2 / A
    r = np.sqrt(B * B + C * C + D * D - 4 * A * E) / 2 / abs(A)
    return pc, r

def farpoint(point,point0):
    dis, xx, yy = 0, 0, 0
    for I in point:
        if distance(I[0], I[1] ,point0[0] ,point0[1]) >= dis:
            dis = distance(I[0], I[1],point0[0],point0[1])
            xx = I[0]
            yy = I[1]
    return xx, yy

def nearpoint(point,point0):
    dis, xx, yy = 99999, 0, 0
    for I in point:
        if distance(I[0], I[1], point0[0], point0[1]) <= dis:
            dis = distance(I[0], I[1], point0[0], point0[1])
            xx = I[0]
            yy = I[1]
    return xx, yy

def alignment(img_copy,output):
    point_k = []
    for i in range(output.shape[0]):
        for j in range(output.shape[1]):
            if output[i, j].all() > 0:
                point_k.append((j, i))
    point1,point2 = find_farpoint(point_k)
    xlen = point1[0] - point2[0]
    ylen = point1[1] - point2[1]
    rad = math.atan2(ylen, xlen)
    deg = math.degrees(rad)
    image_center = tuple(np.array(img_copy.shape)[:2] / 2)
    rot_mat = cv.getRotationMatrix2D(image_center, deg, 1)
    dst_copy = cv.warpAffine(img_copy, rot_mat, img_copy.shape[:2], flags=cv.INTER_LINEAR)
    output = cv.warpAffine(output, rot_mat, output.shape[:2], flags=cv.INTER_LINEAR)
    return dst_copy,output

def revise(output,output0,output1,output3,dst_copy1):
    num = 0
    for i in range(int(output1.shape[0] / 2)):
        for j in range(output1.shape[1]):
            if output1[i, j].all() > 0:
                num += 1
    if num < 100:
        deg = 180
        image_center = tuple(np.array(output.shape)[:2] / 2)
        rot_mat = cv.getRotationMatrix2D(image_center, deg, 1)
        dst_copy1 = cv.warpAffine(dst_copy1, rot_mat, dst_copy1.shape[:2], flags=cv.INTER_LINEAR)
        output = cv.warpAffine(output, rot_mat, output.shape[:2], flags=cv.INTER_LINEAR)
        output0 = cv.warpAffine(output0, rot_mat, output0.shape[:2], flags=cv.INTER_LINEAR)
        output1 = cv.warpAffine(output1, rot_mat, output1.shape[:2], flags=cv.INTER_LINEAR)
        output3 = cv.warpAffine(output3, rot_mat, output3.shape[:2], flags=cv.INTER_LINEAR)

    for i in range(int(output1.shape[0])):
        for j in range(output1.shape[1]):
            if i >= int(output1.shape[0]/2):
                output[i, j],output0[i, j],output1[i, j],output3[i, j] = 0,0,0,0

    return output,output0,output1,output3,dst_copy1

def cal_ang(point_1, point_2, point_3):
    a=math.sqrt((point_2[0]-point_3[0])*(point_2[0]-point_3[0])+(point_2[1]-point_3[1])*(point_2[1] - point_3[1]))
    b=math.sqrt((point_1[0]-point_3[0])*(point_1[0]-point_3[0])+(point_1[1]-point_3[1])*(point_1[1] - point_3[1]))
    c=math.sqrt((point_1[0]-point_2[0])*(point_1[0]-point_2[0])+(point_1[1]-point_2[1])*(point_1[1]-point_2[1]))
    B=math.degrees(math.acos((b*b-a*a-c*c)/(-2*a*c)))
    return B

def find_scale_point(output,output0,output1,output3,dst_copy1):
    point_y, point_g, point_r, point_k, point_z = [], [], [], [], []
    for i in range(output.shape[0]):
        for j in range(output.shape[1]):
            if output[i, j].all() > 0:
                point_k.append((j, i))
    for i in range(output0.shape[0]):
        for j in range(output0.shape[1]):
            if output0[i, j].all() > 0:
                point_y.append((j, i))
    for i in range(output1.shape[0]):
        for j in range(output1.shape[1]):
            if output1[i, j].all() > 0:
                point_g.append((j, i))
    for i in range(output3.shape[0]):
        for j in range(output3.shape[1]):
            if output3[i, j].all() > 0:
                point_r.append((j, i))
    point1, point2 = find_farpoint(point_k)
    xx4, yy4 = farpoint(point_y, point2)
    xx3, yy3 = farpoint(point_g, point2)
    xx5, yy5 = farpoint(point_g, point1)
    pc, r = points2circle(point1, (int(xx3), int(yy3)), point2)
    xz0, yz0 = farpoint(point_r, (int(pc[0]), int(pc[1])))
    xz1, yz1 = nearpoint(point_r, (int(pc[0]), int(pc[1])))
    img3 = np.zeros((dst_copy1.shape[0], dst_copy1.shape[1]), dtype=np.uint8)
    cv.circle(img3, (int(pc[0]), int(pc[1])), int(r), (255, 255, 255), 2)
    img4 = np.zeros((dst_copy1.shape[0], dst_copy1.shape[1]), dtype=np.uint8)
    cv.line(img4, (int(xz0), int(yz0)), (int(xz1), int(yz1)), (255, 255, 255), 1)
    for i in range(dst_copy1.shape[0]):
        for j in range(dst_copy1.shape[1]):
            if img3[i, j] > 0 and img4[i, j] > 0:
                point_z.append((j, i))
    point_scale = np.array(
        [(int(point1[0]), int(point1[1])), (int(point2[0]), int(point2[1])), (int(xx3), int(yy3)),
         (int(xx4), int(yy4)), (int(xx5), int(yy5))])
    point_scale = point_scale[np.lexsort(point_scale[:, ::-1].T)]
    point_pointer = np.array((int(point_z[0][0]), int(point_z[0][1])))
    point_circle = np.array((int(pc[0]), int(pc[1])))
    r_circle = int(r)
    return point_scale,point_pointer,point_circle,r_circle

def read_dial(point_scale,point_pointer,point_circle,r_circle,dst_copy1):
    degree0 = cal_ang(point_scale[0], point_circle, point_scale[-1])
    degreex = cal_ang(point_scale[0], point_circle, ((point_circle[0] - 10), point_circle[1]))
    for i in range(int(degree0 * 10)):
        cv.circle(dst_copy1, (int(point_circle[0] + r_circle * math.cos(math.radians(i / 10 + 180 + degreex))),
                              int(point_circle[1] + r_circle * math.sin(math.radians(i / 10 + 180 + degreex)))), 1,
                  (255, 255, 0), -1)
    i = -2
    for j in point_scale:
        i += 1
        if j[0] >= point_pointer[0]:
            break
    cv.line(dst_copy1, (int(point_circle[0]), int(point_circle[1])), (int(point_scale[i][0]), int(point_scale[i][1])),
            (255, 255, 0), 2)
    cv.line(dst_copy1, (int(point_circle[0]), int(point_circle[1])), (int(j[0]), int(j[1])), (255, 255, 0), 2)
    cv.line(dst_copy1, (int(point_circle[0]), int(point_circle[1])), (int(point_scale[0][0]), int(point_scale[0][1])),
            (255, 255, 0), 2)
    cv.line(dst_copy1, (int(point_circle[0]), int(point_circle[1])), (int(point_scale[-1][0]), int(point_scale[-1][1])),
            (255, 255, 0), 2)
    cv.line(dst_copy1, (int(point_circle[0]), int(point_circle[1])), (int(point_pointer[0]), int(point_pointer[1])),
            (0, 255, 255), 2)
    for k in point_scale:
        cv.circle(dst_copy1, (int(k[0]), int(k[1])), 3, (255, 255, 255), -1)  # white
    ui.printf("刻度盘位置已找到，起点坐标为：（ %d，%d ），终点的坐标为：（ %d，%d ） " % (point_scale[0][0],point_scale[0][1],point_scale[-1][0],point_scale[-1][1]))
    ui.printf("指针的位置已找到，处于第 %d 分区，坐标为： （ %d，%d ） " % (i+1,point_pointer[0],point_pointer[1]))
    if i == -1:
        indication = 0
    else:
        degreez = cal_ang(point_scale[i], point_circle, point_pointer)
        degreem = cal_ang(point_scale[i], point_circle, j)
        if i == 0:
            indication = 0.2 * degreez / degreem
        elif i == 1:
            indication = 0.2 + 1.3 * degreez / degreem
        elif i == 2:
            indication = 1.5 + 0.5 * degreez / degreem
        elif i == 3:
            if degreez / degreem < 0.531:
                indication = 2 + degreez / (degreem * 0.531)
            else:
                indication = 3 + (degreez - (degreem * 0.531)) / (degreem - (degreem * 0.531))
    return indication

def compensation_amendment(indication,inclination_angle):
    return  indication

def plot_one_box(scale,x, img, color=None, label=None, line_thickness=None):
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(int(x[0])*scale), int(int(x[1])*scale)), (int(int(x[2])*scale), int(int(x[3])*scale))
    a1 ,a2 = c1 ,c2
    cv.rectangle(img, c1, c2, color, thickness=tl, lineType=cv.LINE_AA)
    if label:
        tf = max(tl - 1, 1)
        t_size = cv.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv.rectangle(img, c1, c2, color, -1, cv.LINE_AA)
        cv.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv.LINE_AA)
    return a1,a2


def detect(info1,info2):
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='weights/best.pt', help='model.pt path')
    parser.add_argument('--source', type=str, default=info1,  help='source')
    parser.add_argument('--output', type=str, default=info2, help='output folder')
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.4, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='IOU threshold for NMS')
    parser.add_argument('--fourcc', type=str, default='mp4v', help='output video codec (verify ffmpeg support)')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    opt = parser.parse_args()
    opt.img_size = check_img_size(opt.img_size)
    out, source, weights, view_img, save_txt, imgsz = \
        opt.output, opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size
    webcam = source == '0' or source.startswith('rtsp') or source.startswith('http') or source.endswith('.txt')
    device = torch_utils.select_device(opt.device)
    if os.path.exists(out):
        shutil.rmtree(out)
    os.makedirs(out)
    half = device.type != 'cpu'
    model = torch.load(weights, map_location=device)['model'].float()  # load to FP32
    model.to(device).eval()
    if half:
        model.half()
    classify = False
    if classify:
        modelc = torch_utils.load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model'])  # load weights
        modelc.to(device).eval()
    if webcam:
        view_img = True
        cudnn.benchmark = True
        dataset = LoadStreams(source, img_size=imgsz)
    else:
        save_img = True
        dataset = LoadImages(source, img_size=imgsz)
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]
    img = torch.zeros((1, 3, imgsz, imgsz), device=device)
    _ = model(img.half() if half else img) if device.type != 'cpu' else None
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()
        img /= 255.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        t1 = torch_utils.time_synchronized()
        pred = model(img, augment=opt.augment)[0]
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t2 = torch_utils.time_synchronized()
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)
        for i, det in enumerate(pred):
            if webcam:
                p, s, im0 = path[i], '%g: ' % i, im0s[i].copy()
            else:
                p, s, im0 = path, '', im0s

            save_path = str(Path(out) / Path(p).name)
            txt_path = str(Path(out) / Path(p).stem) + ('_%g' % dataset.frame if dataset.mode == 'video' else '')
            s += '%gx%g ' % img.shape[2:]
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]
            if det is not None and len(det):
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()
                    s += '%g %ss, ' % (n, names[int(c)])
                for *xyxy, conf, cls in det:
                    if save_img or view_img:
                        label = '%s %.2f' % (names[int(cls)], conf)
                        im0,scale = resizeimg(im0)
                        img_input = im0.copy()
                        c1,c2 = plot_one_box(scale,xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)
            ui.printf("原始图片加载完成")
            resize = img_input
            original_picture = resize.copy()
            img_copy = resize
            img_copy1 = img_copy.copy()
            pre = preliminary_pretreatment(resize)
            ui.printf('%sDone. (%.3fs)' % (s, t2 - t1))
            x, y, r = center(pre)
            candidates, p1, p2, scale = find_scale1(img_copy1,x,y,r)
            img_edge = pretreatment(img_copy)
            ui.showimg(original_picture)
            X, Y, MA, ma, angle, img_copy = findEllipse(x, y, r,c1,c2, scale, candidates, p1, p2, img_edge, img_copy)
            inclination_angle = math.degrees(math.asin(min(ma, MA) / max(ma, MA)))
            ui.printf("表盘拍摄倾斜角度为： %.2f°" % (90 - inclination_angle))
            img_copy, point = findvertex(img_copy, X, Y, MA, ma, angle)
            dst = perspective_transformation(img_copy, point)
            dst_copy = dst.copy()
            output, output0, output1, output3 = find_scale(dst)
            dst_copy, output = alignment(dst_copy, output)
            dst_copy1 = dst_copy.copy()
            ui.showimg(im0)
            output, output0, output1, output3 = find_scale(dst_copy)
            output, output0, output1, output3, dst_copy1 = revise(output, output0, output1, output3, dst_copy1)
            point_scale, point_pointer, point_circle, r_circle = find_scale_point(output, output0, output1, output3,
                                                                                  dst_copy1)
            indication = read_dial(point_scale, point_pointer, point_circle, r_circle,dst_copy1)
            indication = compensation_amendment(indication,inclination_angle)
            string = "%.2f mA" % indication
            cv.putText(dst_copy1, string, (int(point_circle[0] * 2 / 5), int(point_circle[1] + 70)),
                       cv.FONT_HERSHEY_SIMPLEX, 2,
                       (0, 215, 255), 3)
            ui.printf("该表盘图片识别完成")
            ui.printf("表盘的读数为： %.2f mA" % indication)
            c1, c2 = plot_one_box(scale, xyxy, dst_copy1, label=label, color=colors[int(cls)], line_thickness=3)
            ui.showimg(dst_copy1)
            if save_img:
                if dataset.mode == 'images':
                    cv.imwrite(save_path, dst_copy1)

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1113, 848)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.textBrowser = QtWidgets.QTextBrowser(self.centralwidget)
        self.textBrowser.setGeometry(QtCore.QRect(240, 30, 621, 81))
        self.textBrowser.setStyleSheet("background-color: rgb(181, 255, 253);")
        self.textBrowser.setObjectName("textBrowser")
        self.layoutWidget = QtWidgets.QWidget(self.centralwidget)
        self.layoutWidget.setGeometry(QtCore.QRect(110, 202, 901, 41))
        self.layoutWidget.setObjectName("layoutWidget")
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout(self.layoutWidget)
        self.horizontalLayout_2.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.label_2 = QtWidgets.QLabel(self.layoutWidget)
        self.label_2.setStyleSheet("font: 12pt \"3ds\";\n"
                                   "background-color: rgb(253, 255, 211);")
        self.label_2.setObjectName("label_2")
        self.horizontalLayout_2.addWidget(self.label_2)
        self.textEdit_2 = QtWidgets.QTextEdit(self.layoutWidget)
        self.textEdit_2.setObjectName("textEdit_2")
        self.horizontalLayout_2.addWidget(self.textEdit_2)
        self.pushButton = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton.setGeometry(QtCore.QRect(240, 260, 191, 41))
        self.pushButton.setStyleSheet("background-color: rgb(0,255,0);\n"
                                      "font: 20pt \"3ds\";")
        self.pushButton.setObjectName("pushButton")
        self.layoutWidget_2 = QtWidgets.QWidget(self.centralwidget)
        self.layoutWidget_2.setGeometry(QtCore.QRect(520, 330, 491, 21))
        self.layoutWidget_2.setObjectName("layoutWidget_2")
        self.verticalLayout_3 = QtWidgets.QVBoxLayout(self.layoutWidget_2)
        self.verticalLayout_3.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        self.label_5 = QtWidgets.QLabel(self.layoutWidget_2)
        self.label_5.setStyleSheet("background-color: rgb(255, 255, 255);\n"
                                   "font: 12pt \"3ds\";")
        self.label_5.setObjectName("label_5")
        self.verticalLayout_3.addWidget(self.label_5)
        self.layoutWidget1 = QtWidgets.QWidget(self.centralwidget)
        self.layoutWidget1.setGeometry(QtCore.QRect(110, 150, 901, 41))
        self.layoutWidget1.setObjectName("layoutWidget1")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.layoutWidget1)
        self.horizontalLayout.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.label = QtWidgets.QLabel(self.layoutWidget1)
        self.label.setStyleSheet("font: 12pt \"3ds\";\n"
                                 "background-color: rgb(253, 255, 211);")
        self.label.setObjectName("label")
        self.horizontalLayout.addWidget(self.label)
        self.textEdit = QtWidgets.QTextEdit(self.layoutWidget1)
        self.textEdit.setObjectName("textEdit")
        self.horizontalLayout.addWidget(self.textEdit)
        self.label_6 = QtWidgets.QLabel(self.centralwidget)
        self.label_6.setGeometry(QtCore.QRect(520, 360, 491, 401))
        self.label_6.setText("")
        self.label_6.setObjectName("label_6")
        self.label_3 = QtWidgets.QLabel(self.centralwidget)
        self.label_3.setGeometry(QtCore.QRect(108, 331, 311, 16))
        self.label_3.setStyleSheet("background-color: rgb(255, 255, 255);\n"
                                   "font: 12pt \"3ds\";")
        self.label_3.setObjectName("label_3")
        self.textBrowser_2 = QtWidgets.QTextBrowser(self.centralwidget)
        self.textBrowser_2.setGeometry(QtCore.QRect(108, 353, 311, 421))
        self.textBrowser_2.setObjectName("textBrowser_2")
        self.pushButton_3 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_3.setGeometry(QtCore.QRect(670, 260, 191, 41))
        self.pushButton_3.setStyleSheet("background-color: rgb(255, 0, 0);\n"
                                        "font: 20pt \"3ds\";")
        self.pushButton_3.setObjectName("pushButton_3")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1113, 23))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)
        self.pushButton.clicked.connect(self.handleCalc)
        self.pushButton_3.clicked.connect(self.handleCalc3)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.textBrowser.setHtml(_translate("MainWindow",
                                            "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
                                            "<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
                                            "p, li { white-space: pre-wrap; }\n"
                                            "</style></head><body style=\" font-family:\'SimSun\'; font-size:9pt; font-weight:400; font-style:normal;\">\n"
                                            "<p align=\"center\" style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:20pt; font-weight:600;\">基于YOLOv5和OpenCV的指针式表盘读数识别程序（过程演示）</span></p></body></html>"))
        self.label_2.setText(_translate("MainWindow", "存放识别图片的文件夹位置："))
        self.pushButton.setText(_translate("MainWindow", "开始识别"))
        self.label_5.setText(_translate("MainWindow", "                          识别过程"))
        self.label.setText(_translate("MainWindow", "存放原始图片的文件夹位置："))
        self.label_3.setText(_translate("MainWindow", "                 日志"))
        self.pushButton_3.setText(_translate("MainWindow", "停止识别"))


    def handleCalc(self):
        info1 = self.textEdit.toPlainText()
        info2 = self.textEdit_2.toPlainText()
        for line in info1.splitlines():
            if not line.strip():
                continue
            parts = line.split(' ')
        for line in info2.splitlines():
            if not line.strip():
                continue
            parts = line.split(' ')
        detect(info1,info2)


    def handleCalc3(self):
        os._exit(0)

    def printf(self,text):
        self.textBrowser_2.append(text)
        self.cursor = self.textBrowser_2.textCursor()
        self.textBrowser_2.moveCursor(self.cursor.End)
        QtWidgets.QApplication.processEvents()

    def showimg(self,img):
        img2 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        _image = QtGui.QImage(img2[:], img2.shape[1], img2.shape[0], img2.shape[1] * 3,
                              QtGui.QImage.Format_RGB888)
        n_width = _image.width()
        n_height = _image.height()
        if n_width / 600 >= n_height / 400:
            ratio = n_width / 600
        else:
            ratio = n_height / 400
        new_width = int(n_width / ratio)
        new_height = int(n_height / ratio)
        new_img = _image.scaled(new_width, new_height, Qt.KeepAspectRatio)
        self.label_6.setPixmap(QPixmap.fromImage(new_img))

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
