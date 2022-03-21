import sys
import cv2
import os
import torch
import json
import pickle
from time import time
from PIL import Image
from diamondSpace import DiamondSpace
from vpd_utils import cvt_diamond_space, start_end_line, draw_point_line, draw_points, computeCameraCalibration, \
    get_intersections
from edges import neighborhood
import matplotlib.pylab as plt
import numpy as np

sys.path.append(r'E:\Pycharm\mine\Vehicle_Orientation_Detect\SSD')

from SSD.inference import time_synchronized, get_data_transform, load_model

good_features_parameters = dict(maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7, useHarrisDetector=True,
                                k=0.04)
draw_circle_parameters = dict(radius=3, color=(0, 0, 255), thickness=3, )
optical_flow_parameters = dict(winSize=(21, 21), minEigThreshold=1e-4)
hough_lines_parameters = dict(rho=1.0, theta=np.pi / 180, threshold=100, minLineLength=50, maxLineGap=15)


class Tracks(object):
    def __init__(self, video_src, roi, detect_interval=20, track_interval=10):

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.tracks = []  # vehicle tracks
        self.edgelets = []  # vehicle edges
        self.background = None
        self.detectInterval = detect_interval
        self.track_interval = track_interval
        self.roi = np.array(roi)
        self.camera = cv2.VideoCapture(video_src)
        self.frame_count = 0
        self.detectModel = None
        self.init_frame = None
        self.current_frame = None
        self.previous_frame = None
        self.features = []
        self.diff_threshold = 1
        self.maxFeatureNum = 50
        self.frame_height = 0
        self.frame_width = 0
        self.feature_determine_threshold = 0.1
        self.tracks_filter_threshold = 0.005

        # vanish point
        self.vp_1 = []
        self.vp_2 = []
        self.vp_3 = []

        self.principal_point = None

        self.DiamondSpace = None

    def run(self):
        self.frame_count = 0
        while True:
            start = time()

            flag, frame = self.camera.read()
            h, w = frame.shape[:2]
            if h * w > 1e6:
                frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)
            if self.frame_count == 0:
                self.init_frame = frame
                self.frame_height, self.frame_width = frame.shape[:2]
                self.principal_point = [self.frame_height / 2, self.frame_width / 2]
                self.DiamondSpace = DiamondSpace(d=min(self.frame_height, self.frame_width) / 2, size=256)

            self.current_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            if not flag:
                print('no frame grabbed!')
                return 0
            if len(self.features) > 0:
                # KLT Tracker
                pimg, cimg = self.previous_frame, self.current_frame
                last_feature = np.float32([f[-1] for f in self.features])
                feature_c, status_c, _ = cv2.calcOpticalFlowPyrLK(pimg, cimg, last_feature, None,
                                                                  **optical_flow_parameters)
                feature_recover, status_r, _ = cv2.calcOpticalFlowPyrLK(cimg, pimg, feature_c, None,
                                                                        **optical_flow_parameters)

                good_features_index = [1e-5 < diff < self.diff_threshold for diff in
                                       abs(feature_recover - last_feature).reshape(-1, 2).max(-1)]
                # new_tracks = []
                for fp, fc, isGood in zip(self.features, feature_c, good_features_index):
                    if not isGood:
                        continue
                    fp.append(tuple(fc.ravel().tolist()))

                    if len(fp) > self.maxFeatureNum:
                        del fp[0]
                        if self.frame_count % self.track_interval == 0:
                            if self.get_track_length(fp) > 30:
                                temp = fp.copy()
                                self.tracks.append(temp)
                    # new_tracks.append(fp)
                # cv2.polylines(frame, [np.int32(tr) for tr in new_tracks], False, (0, 255, 0))

            if self.frame_count % self.detectInterval == 0:
                mask = np.zeros_like(self.current_frame)
                boxes, classes_str = self.detect_car(frame)
                if len(boxes) > 0:
                    for box, box_class in zip(boxes, classes_str):
                        if box_class in ['car', "bus"]:
                            mask[box[1]:box[3], box[0]:box[2]] = 255
                            points = self.lines_from_box(frame, box)
                            if points is not None:
                                self.edgelets.append(points)
                            # self.edgelets.append(lines)
                            # cv2.imshow('line', line_img)
                    for x, y in [np.int32(tr[-1]) for tr in self.features]:
                        cv2.circle(mask, (x, y), 5, 0, -1)

                # good tracker
                p = cv2.goodFeaturesToTrack(self.current_frame, mask=mask, **good_features_parameters)
                if p is not None:
                    # print(len(p))
                    for x, y in np.float32(p).reshape(-1, 2):
                        # 特征点测试
                        # cv2.circle(frame, (int(x), int(y)), radius=5, color=(0, 0, 255), thickness=3)
                        self.features.append([(x, y)])

            self.frame_count += 1
            self.previous_frame = self.current_frame.copy()
            cv2.imshow('vehicle tracks', frame)
            # print(f'fps:{1 / (time() - start)}')
            if cv2.waitKey(1) & 0xFF == 27:
                self.camera.release()
                cv2.destroyAllWindows()
                break

    def load_ssd_model(self, class_json):
        assert os.path.exists(class_json), "file '{}' dose not exist.".format(class_json)
        json_file = open(json_path, 'r')
        class_dict = json.load(json_file)
        self.category_index = {v: k for k, v in class_dict.items()}
        self.detectModel = load_model(self.device)
        self.data_transform = get_data_transform()

    def detect_car(self, frame, threshold=0.5):
        image_RBG = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        original_img = Image.fromarray(image_RBG)  # 网络输入为rgb顺序
        img, _ = self.data_transform(original_img)
        # expand batch dimension
        img = torch.unsqueeze(img, dim=0)
        with torch.no_grad():
            # time_start = time_synchronized()
            predictions = self.detectModel(img.to(self.device))[0]  # bboxes_out, labels_out, scores_out
            # time_end = time_synchronized()
            # print("inference+NMS time: {}".format(time_end - time_start))
            predict_boxes = predictions[0].to("cpu").numpy()
            predict_boxes[:, [0, 2]] = predict_boxes[:, [0, 2]] * original_img.size[0]
            predict_boxes[:, [1, 3]] = predict_boxes[:, [1, 3]] * original_img.size[1]
            predict_classes = predictions[1].to("cpu").numpy()
            predict_scores = predictions[2].to("cpu").numpy()
            if len(predict_boxes) == 0:
                print("没有检测到任何目标!")
            filter_boxes = predict_boxes[predict_scores > threshold].astype(int)
            classes_str = [self.category_index[i] for i in predict_classes[predict_scores > threshold]]
        return filter_boxes, classes_str

    @staticmethod
    def get_track_length(track):
        """
        获得track的平均长度，用于过滤已消失在监控中的特征点仍在跟踪的情况
        :param track: the track of the feature point
        :return: the average length of track
        """
        x, y = track[0]
        length = 0
        for j in np.arange(1, len(track), 1):
            xn = track[j][0]
            yn = track[j][1]
            diff_x = (xn - x)
            diff_y = (yn - y)
            length += np.sqrt(diff_x * diff_x + diff_y * diff_y)
        return length / len(track)

    def draw_all_tracks(self, save_name='all_tracks.jpg', save_path='./', save_tracks_data=False):
        display = self.init_frame.copy()
        cv2.polylines(display, [np.int32(tr) for tr in self.tracks], False, (0, 255, 0))
        cv2.imwrite(os.path.join(save_path, save_name), display)
        if save_tracks_data:
            with open(os.path.join(save_path, 'tracks.data'), 'wb') as f:
                pickle.dump(self.tracks, f)

    def lines_from_box(self, frame, box, drawFlag=False):
        vehicle_img = frame[box[1]:box[3], box[0]:box[2]]
        height, width = vehicle_img.shape[:2]
        edges = cv2.Canny(vehicle_img, 200, 200, L2gradient=True)
        orientation, quality = neighborhood(edges)
        unique = np.unique(quality)
        sum = 0
        for u in unique:
            if sum > 1 / 2 * height * width:
                break
            sum += np.sum(quality == u)
        edges[quality > u] = 255
        edges[quality <= u] = 0
        # thres, edges = cv2.threshold(quality, 0, 255, cv2.THRESH_OTSU)
        # lines = get_lines(edges, orientation, box)
        points = cv2.HoughLinesP(edges, 1.0, np.pi / 180, 30, minLineLength=30, maxLineGap=20)
        if points is not None:
            points = points.reshape(-1, 4)
            points[:, [0, 2]] += box[0]
            points[:, [1, 3]] += box[1]
            if drawFlag:
                for p in points:
                    x1, y1, x2, y2 = p
                    cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                return frame
        return points

    def detect_lines(self, frame):

        img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cv2.fillPoly(img_gray, [self.roi], 0)
        img_filter = cv2.medianBlur(img_gray, 7)
        sobel = cv2.Sobel(img_filter, -1, 0, 1)
        _, res = cv2.threshold(sobel, 0, 255, cv2.THRESH_OTSU)
        points = cv2.HoughLinesP(res, **hough_lines_parameters)
        return points

    def get_vp1(self, visualize=False):
        lines = cvt_diamond_space(self.tracks)
        self.DiamondSpace.insert(lines)
        vps, values, vpd_s = self.DiamondSpace.find_peaks(t=0.9, )
        # vps中权重最大的一个点取为第一消失点
        self.vp_1 = vps[0][:2]

        if visualize:
            img = cv2.cvtColor(self.init_frame, cv2.COLOR_BGR2RGB)
            print("numbers of vps", len(vps))
            size = self.DiamondSpace.size
            scale = self.DiamondSpace.scale

            # 第一消失点可视化
            _, ax = plt.subplots(1, 2, figsize=(20, 10))
            ax[0].imshow(self.DiamondSpace.attach_spaces(), cmap="Greys", extent=(
                (-size + 0.5) / scale, (size - 0.5) / scale, (size - 0.5) / scale,
                (-size + 0.5) / scale))
            ax[0].set(title="Accumulator", xticks=np.linspace(-size + 1, size - 1, 5) / scale,
                      yticks=np.linspace(-size + 1, size - 1, 5) / scale)
            ax[0].plot(vpd_s[0, 0] / scale, vpd_s[0, 1] / scale, "ro", markersize=11)
            ax[0].plot(vpd_s[1:, 0] / scale, vpd_s[1:, 1] / scale, "go", markersize=11)
            ax[0].invert_yaxis()

            ax[1].imshow(img)
            ax[1].set(title="first vanishing point in image")
            ax[1].plot(vps[0, 0], vps[0, 1], 'ro', markersize=11)
            ax[1].plot(vps[1:, 0], vps[1:, 1], 'go', markersize=11)

            plt.savefig('first_vp1.jpg')

    def get_vp2(self, visualize=False):
        points = np.vstack(self.edgelets)
        lines = start_end_line(points)
        # test_img = draw_lines(self.init_frame, lines)
        index = self.DiamondSpace.filter_lines_from_peak(self.vp_1, lines)

        vps, values, vpd_s = self.DiamondSpace.find_peaks(t=0.9, )
        # vps中权重最大的一个点取为第二消失点
        self.vp_2 = vps[0][:2]
        if visualize:
            img = cv2.cvtColor(self.init_frame, cv2.COLOR_BGR2RGB)
            print("numbers of vps", len(vps))
            size = self.DiamondSpace.size
            scale = self.DiamondSpace.scale

            # 第一消失点可视化
            _, ax = plt.subplots(1, 2, figsize=(20, 10))
            ax[0].imshow(self.DiamondSpace.attach_spaces(), cmap="Greys", extent=(
                (-size + 0.5) / scale, (size - 0.5) / scale, (size - 0.5) / scale,
                (-size + 0.5) / scale))
            ax[0].set(title="Accumulator", xticks=np.linspace(-size + 1, size - 1, 5) / scale,
                      yticks=np.linspace(-size + 1, size - 1, 5) / scale)
            ax[0].plot(vpd_s[0, 0] / scale, vpd_s[0, 1] / scale, "ro", markersize=11)
            # ax[0].plot(vpd_s[1:, 0] / scale, vpd_s[1:, 1] / scale, "go", markersize=11)
            ax[0].invert_yaxis()

            ax[1].imshow(img)
            ax[1].set(title="first vanishing point in image")

            ax[1].plot(vps[0, 0], vps[0, 1], 'ro', markersize=11)
            # ax[1].plot(vps[1:, 0], vps[1:, 1], 'go', markersize=11)
            # ax[1].plot()

            plt.savefig('first_vp2.jpg')
        # cv2.imshow('test', test_img)
        # cv2.waitKey(0)
        return

    def save_calibration(self, visualize=False):
        vp1, vp2, vp3, pp, roadPlane, focal = computeCameraCalibration(self.vp_1, self.vp_2, self.principal_point)

        calibration = dict(vp1=vp1.tolist(), vp2=vp2.tolist(), vp3=vp3.tolist(), principal_point=pp.tolist(),
                           roadPlane=roadPlane.tolist(), focal=focal)
        with open('./calibrations.json', 'w') as f:
            json.dump(calibration, f)

        if visualize:
            img = cv2.cvtColor(self.init_frame, cv2.COLOR_BGR2RGB)
            plt.figure(figsize=(20, 20))
            plt.imshow(img)
            plt.plot(vp1[0], vp1[1], 'ro', markersize=11)
            plt.plot(vp2[0], vp2[1], 'ro', markersize=11)
            plt.plot(vp3[0], vp3[1], 'ro', markersize=11)
            plt.savefig("all_vps.jpg")
        return calibration

    def transform(self, ):
        # todo: transform to bird view
        p1, p2, p3, p4 = self.roi
        # height, width = self.init_frame.shape[:2]
        height, width = 480, 854

        # trans = []
        # trans.append(p1[::-1])
        # trans.append(get_interaction(np.stack([self.vp_2, p3[::-1]]).ravel(), np.stack([p1[::-1], p2[::-1]]).ravel()))
        # trans.append(p3[::-1])
        # trans.append(get_interaction(np.stack([self.vp_2, p1[::-1]]).ravel(), np.stack([p3[::-1], p4[::-1]]).ravel()))

        trans = []
        trans.append(get_intersections(np.stack([self.vp_2, p4[::-1]]).ravel(), np.stack([p1[::-1], p2[::-1]]).ravel()))
        trans.append(get_intersections(np.stack([self.vp_2, p3[::-1]]).ravel(), np.stack([p1[::-1], p2[::-1]]).ravel()))
        trans.append(p3[::-1])
        trans.append(p4[::-1])

        dst_points = np.array([[0, height],
                               [0, 0],
                               [width, 0],
                               [width, height]])
        img = cv2.imread('pics/video2_vp2_filter_line.jpg')
        # img = draw_points(img, np.vstack(trans))

        M = cv2.getPerspectiveTransform(np.vstack(trans).astype(np.float32), dst_points.astype(np.float32))
        wrap = cv2.warpPerspective(img, M, (width, height))
        cv2.imshow('warp', wrap)
        cv2.waitKey(0)


if __name__ == '__main__':
    # read class_indict
    json_path = "../SSD/pascal_voc_classes.json"
    # video_path = '../1-2-981_Trim.mp4'
    video_path = r"E:\datasets\surveilliance\v5.avi"
    # roi = [(221, 9), (789, 1431), (2560, 1440), (2560, 0)]
    roi = [(278, 1), (90, 700), (88, 783), (477, 679)]
    # video_path = '../sherbrooke_video.avi'
    # video_path = '../rouen_video.avi'
    cameraTrack = Tracks(video_path, roi)
    cameraTrack.load_ssd_model(json_path)
    cameraTrack.run()
    # cameraTrack.draw_all_tracks()
    cameraTrack.get_vp1()
    # print(cameraTrack.vp_1)
    cameraTrack.get_vp2()
    # print(cameraTrack.vp_2)
    cameraTrack.save_calibration()
