import logging
import math
import cv2
import os
import torch
import numpy as np
import matplotlib.pylab as plt

from detection_model.calibration_utils import cvt_diamond_space, start_end_line, draw_point_line, \
    computeCameraCalibration, draw_points
from utils import get_pair_keypoints, scale_image
from detection_model.edgelets import neighborhood, accumulate_orientation
from detection_model.diamondSpace import DiamondSpace
from IPM.utils import convertToBirdView
from yolov5.model import YoloTensorrt
#from openpifpaf.predictor import Predictor
from time import time

import warnings
from detection_model.lifting_3d import *

import random
from skimage import data, segmentation, color
from skimage import graph
from networkx.algorithms.coloring import greedy_color
from skimage import data, segmentation, color, graph, filters
#from skimage.future import graph as graph_future
#from matplotlib import pyplot as plt

from time import time

# Disable treating warnings as errors in this file
with warnings.catch_warnings():
    warnings.filterwarnings("default")  # or use warnings.resetwarnings()


debug = False#True

#good_features_parameters = dict(maxCorners=50, qualityLevel=0.3, minDistance=7, blockSize=7, useHarrisDetector=True,  k=0.04)
# good_features_parameters = dict(maxCorners=200, qualityLevel=0.3, minDistance=7, blockSize=5, useHarrisDetector=True, k=0.04)
                                
# optical_flow_parameters = dict(winSize=(21, 21), minEigThreshold=1e-4)

good_features_parameters = dict(maxCorners=200, qualityLevel=0.1, minDistance=1, blockSize=3, useHarrisDetector=True, k=0.04)
optical_flow_parameters = dict(winSize=(21, 21), minEigThreshold=1e-4)



def get_warp():
    # read input
    img = cv2.imread("/home/dzhura/ComputerVision/Vehicle_Orientation_Detect/street/frame_0000.jpg")
    hh, ww = img.shape[:2]
    
    # specify input coordinates for corners of red quadrilateral in order TL, TR, BR, BL as x,
    input = np.float32([[200, 861], [507,632], [1403, 618], [1625, 850]])
    
    # get top and left dimensions and set to output dimensions of red rectangle
    width = round(math.hypot(input[0,0]-input[1,0], input[0,1]-input[1,1]))
    height = round(math.hypot(input[0,0]-input[3,0], input[0,1]-input[3,1]))
    print("width:",width, "height:",height)
    
    # set upper left coordinates for output rectangle
    x = input[0,0]
    y = input[0,1]
    
    # specify output coordinates for corners of red quadrilateral in order TL, TR, BR, BL as x,
    output = np.float32([[x,y], [x+width-1,y], [x+width-1,y+height-1], [x,y+height-1]])
    
    # compute perspective matrix
    matrix = cv2.getPerspectiveTransform(input, output)
    print(matrix)
    warp = cv2.warpPerspective(img, matrix, [x+width-1,y+height-1])
    cv2.imwrite("test_warp")

def wait_until_space_pressed():
    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == ord(' '):  # Check if the space key is pressed
            break
            
def ransac_vanishing_point(edgelets, height, num_ransac_iter=2000, threshold_inlier=5):
    """Estimate vanishing point using Ransac.

    Parameters
    ----------
    edgelets: tuple of ndarrays
        (locations, directions, strengths) as computed by `compute_edgelets`.
    num_ransac_iter: int
        Number of iterations to run ransac.
    threshold_inlier: float
        threshold to be used for computing inliers in degrees.

    Returns
    -------
    best_model: ndarry of shape (3,)
        Best model for vanishing point estimated.

    Reference
    ---------
    Chaudhury, Krishnendu, Stephen DiVerdi, and Sergey Ioffe.
    "Auto-rectification of user photos." 2014 IEEE International Conference on
    Image Processing (ICIP). IEEE, 2014.
    """
    locations, directions, strengths = edgelets
    lines = edgelet_lines(edgelets)
    #print("lines :", lines)

    num_pts = strengths.size
    #print("num_pts :", num_pts)

    arg_sort = np.argsort(-strengths)
    first_index_space = arg_sort[:num_pts // 5]
    second_index_space = arg_sort[:num_pts // 2]

    best_model = None
    best_votes = np.zeros(num_pts)
    
    if len(first_index_space) == 0 or len(second_index_space) == 0:
        return best_model

    for ransac_iter in range(num_ransac_iter):
        ind1 = np.random.choice(first_index_space)
        ind2 = np.random.choice(second_index_space)

        l1 = lines[ind1]
        l2 = lines[ind2]

        current_model = np.cross(l1, l2)
        #print(current_model)

        if np.sum(current_model**2) < 1 or current_model[2] == 0: # or current_model[1] / current_model[2] > height/2:
            # reject degenerate candidates
            continue

        current_votes = compute_votes(
            edgelets, current_model, threshold_inlier)

        if current_votes.sum() > best_votes.sum():
            best_model = current_model
            best_votes = current_votes
            # logging.info("Current best model has {} votes at iteration {}".format(
                # current_votes.sum(), ransac_iter))

    return best_model

def vis_edgelets(image, edgelets, color=cv_colors.RED.value, output_path="vis_image.png", show=False):
    """Helper function to visualize edgelets using OpenCV and optionally store the result to a file."""
    
    locations, directions, strengths = edgelets
    print("locations.shape: ", locations.shape)

    # Create a copy of the image to draw on
    vis_image = np.copy(image)

    for i in range(locations.shape[0]):
        # Calculate start and end points for each edgelet
        start_point = (int(locations[i, 0] - directions[i, 0] * strengths[i] / 2),
                       int(locations[i, 1] - directions[i, 1] * strengths[i] / 2))
        end_point = (int(locations[i, 0] + directions[i, 0] * strengths[i] / 2),
                     int(locations[i, 1] + directions[i, 1] * strengths[i] / 2))

        # Draw the edgelet on the image
        cv2.line(vis_image, start_point, end_point, color=color, thickness=1)

    # Display the image
    if show:
        cv2.imshow('Edgelets', vis_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # Save the result to a file if the output_path is provided
    if output_path:
        cv2.imwrite(output_path, vis_image)
        print(f"Result saved to: {output_path}")
        

def compute_edgelets(prev_points, next_points, box1, box2, scale=5, threshold=3.0):

    flow_vectors = next_points - prev_points

    print("Shapes before filtering:")
    print("prev_points shape:", prev_points.shape)
    print("next_points shape:", next_points.shape)
    print("flow_vectors shape:", flow_vectors.shape)

    # Filter flow vectors below the threshold
    magnitude = np.linalg.norm(flow_vectors, axis=1)
    mask = magnitude >= threshold

    # Filter points within box1
    mask_box1 = (
        (prev_points[:, 0] >= box1[0]) & (prev_points[:, 0] <= box1[2]) &
        (prev_points[:, 1] >= box1[1]) & (prev_points[:, 1] <= box1[3])
    )

    # Filter points within box2
    mask_box2 = (
        (next_points[:, 0] >= box2[0]) & (next_points[:, 0] <= box2[2]) &
        (next_points[:, 1] >= box2[1]) & (next_points[:, 1] <= box2[3])
    )

    mask = mask & mask_box1 & mask_box2

    prev_points = prev_points[mask]
    next_points = next_points[mask]
    flow_vectors = flow_vectors[mask]

    print("Shapes after filtering:")
    print("prev_points shape:", prev_points.shape)
    print("next_points shape:", next_points.shape)
    print("flow_vectors shape:", flow_vectors.shape)

    lines = np.concatenate([prev_points[:, None], next_points[:, None]], axis=1)
    lines = lines.reshape(-1, 2, 2)
    #lines = np.int32(lines + 0.5)

    # Uncomment the following lines to draw the lines on the frame
    # for (x1, y1), (x2, y2) in lines:
    #     cv2.line(frame, (x1, y1), (x2, y2), (255, 255, 255), 1)

    locations = []
    directions = []
    strengths = []

    for (x1, y1), (x2, y2) in lines:
        p0, p1 = np.array([x1, y1]), np.array([x2, y2])
        locations.append((p0 + p1) / 2)
        directions.append(p1 - p0)
        strengths.append(np.linalg.norm(p1 - p0))

    # convert to numpy arrays and normalize if directions is not empty
    if directions:
        directions = directions / np.linalg.norm(directions, axis=1)[:, np.newaxis]
        
    locations = np.array(locations)
    directions = np.array(directions)
    strengths = np.array(strengths)

    return locations, directions, strengths



def vis_model(image, edgelets, model, output_path="vis_model.png", show=False):
    """Helper function to visualize computed model using OpenCV and optionally store the result to a file."""

    # Create a copy of the image to draw on
    vis_image = image #np.copy(image)

    # Visualize edgelets with green color
    vis_edgelets(vis_image, edgelets, color=(0, 255, 0))

    # Get inliers based on the computed votes
    inliers = compute_votes(edgelets, model, 10) > 0

    # Extract inlier edgelets
    edgelets = (edgelets[0][inliers], edgelets[1][inliers], edgelets[2][inliers])
    locations, directions, strengths = edgelets

    # Visualize inlier edgelets with red color
    vis_edgelets(vis_image, edgelets, color=(0, 0, 255))

    # Get vanishing point in homogeneous coordinates
    vp = model / model[2]

    # Draw vanishing point as a blue circle
    cv2.circle(vis_image, (int(vp[0]), int(vp[1])), 5, (255, 0, 0), -1)

    # Draw lines from edgelet locations to vanishing point
    for i in range(locations.shape[0]):
        cv2.line(vis_image, (int(locations[i, 0]), int(locations[i, 1])),
                 (int(vp[0]), int(vp[1])), (255, 0, 0), 2)

    # Display the image
    if show:
        cv2.imshow('Model Visualization', vis_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # Save the result to a file if the output_path is provided
    if output_path:
        cv2.imwrite(output_path, vis_image)
        print(f"Result saved to: {output_path}")


def edgelet_lines(edgelets):
    """Compute lines in homogenous system for edglets.

    Parameters
    ----------
    edgelets: tuple of ndarrays
        (locations, directions, strengths) as computed by `compute_edgelets`.

    Returns
    -------
    lines: ndarray of shape (n_edgelets, 3)
        Lines at each of edgelet locations in homogenous system.
    """
    locations, directions, _ = edgelets
    normals = np.zeros_like(directions)
    normals[:, 0] = directions[:, 1]
    normals[:, 1] = -directions[:, 0]
    p = -np.sum(locations * normals, axis=1)
    lines = np.concatenate((normals, p[:, np.newaxis]), axis=1)
    return lines
    

def compute_votes(edgelets, model, threshold_inlier=5):
    """Compute votes for each of the edgelet against a given vanishing point.

    Votes for edgelets which lie inside threshold are same as their strengths,
    otherwise zero.

    Parameters
    ----------
    edgelets: tuple of ndarrays
        (locations, directions, strengths) as computed by `compute_edgelets`.
    model: ndarray of shape (3,)
        Vanishing point model in homogenous cordinate system.
    threshold_inlier: float
        Threshold to be used for computing inliers in degrees. Angle between
        edgelet direction and line connecting the  Vanishing point model and
        edgelet location is used to threshold.

    Returns
    -------
    votes: ndarry of shape (n_edgelets,)
        Votes towards vanishing point model for each of the edgelet.

    """
    vp = model[:2] / model[2]

    locations, directions, strengths = edgelets

    est_directions = locations - vp
    dot_prod = np.sum(est_directions * directions, axis=1)
    abs_prod = np.linalg.norm(directions, axis=1) * \
        np.linalg.norm(est_directions, axis=1)
    abs_prod[abs_prod == 0] = 1e-5

    cosine_theta = dot_prod / abs_prod
    
    theta = np.zeros_like(cosine_theta)
    
    # Iterate element by element to compute theta
    for i in range(len(cosine_theta)):
        theta[i] = np.arccos(np.abs(cosine_theta[i])) if cosine_theta[i] < 1 and cosine_theta[i] > -1 else threshold_inlier * np.pi / 180
    
    #theta = np.arccos(np.abs(cosine_theta))

    theta_thresh = threshold_inlier * np.pi / 180
    return (theta < theta_thresh) * strengths


# def segment(img, n = 40):

    # labels1 = segmentation.slic(img, compactness=30, n_segments=n, start_label=1)
    # out1 = color.label2rgb(labels1, img, kind='avg', bg_label=0)
    
    # g = graph.rag_mean_color(img, labels1)
    # labels2 = graph.cut_threshold(labels1, g, 29)
    # #colors = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for _ in range(n)]
    # # coloring = greedy_color(g)
    # # labels = np.unique(list(coloring.values()))
    # # n_label = len(labels)
    # # palette = np.random.random((n_label, 3))  # just for demonstration, one should use a set of distinguishable colors
    # # color_idx = [val for key, val in sorted(coloring.items())]
    # # colors = palette[color_idx]

    # #out2 = color.label2rgb(labels2) #, img, colors = colors)
    # out2 = color.label2rgb(labels2, img, kind='avg', bg_label=0)
    # #out2 = color.label2rgb(labels2, img, kind='overlay', colors=['red','green','blue','cyan','magenta','yellow'])
    
    # return out2

def generate_random_colors(num_colors):
    # Generate a list of random RGB colors in the range [0, 255]
    colors = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for _ in range(num_colors)]
    return colors
    
def segment(image, compactness_val = 30, n = 2*400,  bg_color= (0, 0, 0) ):
    # SLIC segmentation
    labels1 = segmentation.slic(image, compactness=compactness_val, n_segments=n, start_label=1)
    #out1 = color.label2rgb(labels1, image, kind='avg', bg_label=0)

    # RAG and threshold-cut segmentation
    g = graph.rag_mean_color(image, labels1)
    labels2 = graph.cut_threshold(labels1, g, 5)
    # out2 = color.label2rgb(labels2, image, kind='overlay', colors=['red','green','blue','cyan','magenta','yellow'])

    # return cv2.cvtColor(np.uint8(out2), cv2.COLOR_RGB2BGR)

    # Change colors to random in the second segmentation
    random_colors = generate_random_colors(np.max(labels2) + 1)
    out2 = np.zeros_like(image)
    for label, c in zip(np.unique(labels2), random_colors):
        out2[labels2 == label] = c
 
    out2[labels2 == 0] = bg_color
    # Set background label to 0 and make the background white
    labels2[labels2 == 0] = 0
    out2 = color.label2rgb(labels2, out2, kind='avg', bg_label=0)#, bg_color=bg_color)  # Set bg_color to white      
     
    return out2

def spectral_cluster(img, compactness_val=30, n=60):
    '''
    Apply spectral clustering to a given image using k-means clustering and
    display results.

    Args:
        filename: name of the image to segment.

        compactness_val: Controls the "boxyness" of each segment. Higher values
          mean a more boxed shape.

        n = number of clusters.
     '''
    #img = misc.imread(filename)
    labels1 = segmentation.slic(img, compactness=compactness_val, n_segments=n)
    out1 = color.label2rgb(labels1, img, kind='overlay', colors=['red','green','blue','cyan','magenta','yellow'])
    

    # Change colors to random in the second segmentation
    random_colors = generate_random_colors(np.max(labels1) + 1)
    out2 = np.zeros_like(img)
    for label, c in zip(np.unique(labels1), random_colors):
        out2[labels1 == label] = c

    # # Display the images using OpenCV
    # cv2.imshow("Segmentation with Average Colors", out1)
    # cv2.imshow("Segmentation with Random Colors", out2)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    
    return out2

def weight_boundary(graph, src, dst, n):
    """
    Handle merging of nodes of a region boundary region adjacency graph.

    This function computes the `"weight"` and the count `"count"`
    attributes of the edge between `n` and the node formed after
    merging `src` and `dst`.

    Parameters
    ----------
    graph : RAG
        The graph under consideration.
    src, dst : int
        The vertices in `graph` to be merged.
    n : int
        A neighbor of `src` or `dst` or both.

    Returns
    -------
    data : dict
        A dictionary with the "weight" and "count" attributes to be
        assigned for the merged node.

    """
    default = {'weight': 0.0, 'count': 0}

    count_src = graph[src].get(n, default)['count']
    count_dst = graph[dst].get(n, default)['count']

    weight_src = graph[src].get(n, default)['weight']
    weight_dst = graph[dst].get(n, default)['weight']

    count = count_src + count_dst
    return {
        'count': count,
        'weight': (count_src * weight_src + count_dst * weight_dst)/count
    }

def merge_boundary(graph, src, dst):
    """Call back called before merging 2 nodes.

    In this case we don't need to do any computation here.
    """
    pass

def segment_boundary(img):
    edges = filters.sobel(color.rgb2gray(img))
    labels = segmentation.slic(img, compactness=30, n_segments=400, start_label=1)
    g = graph.rag_boundary(labels, edges)

    # plt.imshow(img)
    # plt.title('Initial Image')
    # plt.show()

    # graph.show_rag(labels, g, img)
    # plt.title('Initial RAG')

    labels2 = graph.merge_hierarchical(labels, g, thresh=0.08, rag_copy=False,
                                       in_place_merge=True,
                                       merge_func=merge_boundary,
                                       weight_func=weight_boundary)

    # graph.show_rag(labels2, g, img)
    # plt.title('RAG after hierarchical merging')

    # plt.figure()
    out = color.label2rgb(labels2, img, kind='avg', bg_label=0)
    # plt.imshow(out)
    # plt.title('Final Segmentation')

    # plt.show()

    return cv2.cvtColor(out, cv2.COLOR_RGB2BGR)

def compute_optical_flow(prev_frame, next_frame):
    # Convert frames to grayscale
    prev_gray = prev_frame #cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    next_gray = next_frame #cv2.cvtColor(next_frame, cv2.COLOR_BGR2GRAY)

    # Compute dense optical flow using Farneback method
    flow = cv2.calcOpticalFlowFarneback(prev_gray, next_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)

    return flow

def flow_to_rgb(flow):
    # Compute magnitude and angle of flow vectors
    magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])

    # Set the color based on the angle and normalize the magnitude
    hsv = np.zeros((flow.shape[0], flow.shape[1], 3), dtype=np.uint8)
    hsv[..., 0] = angle * 180 / np.pi / 2
    hsv[..., 1] = 255
    hsv[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)

    # Convert HSV to RGB
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    return rgb

def draw_flow(image, flow, step=16):
    h, w = image.shape[:2]
    y, x = np.mgrid[step//2:h:step, step//2:w:step].reshape(2, -1).astype(int)

    # Extract flow vectors at specified points
    fx, fy = flow[y, x].T

    # Create lines representing the flow vectors
    lines = np.vstack([x, y, x + fx, y + fy]).T.reshape(-1, 2, 2)
    lines = np.int32(lines)

    # Draw the flow vectors on the image
    drawn_image = image.copy()
    cv2.polylines(drawn_image, lines, 0, (0, 255, 0), 2, cv2.LINE_AA)

    return drawn_image


def CLAHE(img, ):
    """
    均衡化
    :param img:
    :return:
    """
    b, g, r = cv2.split(img)
    clahe = cv2.createCLAHE(clipLimit=0.2, tileGridSize=(8, 8))
    b = clahe.apply(b)
    g = clahe.apply(g)
    r = clahe.apply(r)
    new = cv2.merge([b, g, r])
    return new


class Calibration_Yolo(object):
    def __init__(self, video_src, input_shape=(900, 1600), detect_interval=20, track_interval=10):

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.tracks = []  # vehicle tracks
        self.edgelets = []  # vehicle edges
        self.features = []  # Harris corners
        self.background = None
        self.detectInterval = detect_interval
        self.track_interval = track_interval
        self.camera = cv2.VideoCapture(video_src)

        self.frame_count = 0
        self.detectModel = None
        self.init_frame = None
        self.current_frame = None
        self.previous_frame = None
        #tracker
        self.projections = None
        self.diff_threshold = 1
        self.maxFeatureNum = 50

        self.feature_determine_threshold = 0.1
        self.tracks_filter_threshold = 0.005

        # calibration
        self.frame_width = int(self.camera.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.scale = scale_image((self.frame_height, self.frame_width), input_shape, scaleUp=True)
        # DiamondSpace
        self.DiamondSpace = None
        # vanish point
        self.vp_1 = []
        self.vp_2 = []
        self.vp_3 = []
        self.principal_point = None
        self.target_shape = None

        # preprocess
        self.roi = None
        # self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))  # 均衡化

        # logger
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)

    def setROI(self, roi):
        self.roi = np.array(roi).astype(np.int32).reshape(1, -1, 2)
        if self.scale:
            self.mask = np.zeros((int(self.frame_height * self.scale), int(self.frame_width * self.scale)),
                                 dtype=np.uint8)
        else:
            self.mask = np.zeros((self.frame_height, self.frame_width), dtype=np.uint8)
        self.mask = cv2.fillPoly(self.mask, self.roi, 255)

    def run(self, threshold=0.5, view_process=False):
        self.logger.info("starting calibrate from video......")
        self.frame_count = 0
        _, frame = self.camera.read()
        # if not self.roi:
        #
        if self.scale != 1:
            frame = cv2.resize(frame, (0, 0), fx=self.scale, fy=self.scale, interpolation=cv2.INTER_LINEAR)
        self.init_frame = frame
        if self.roi is not None:
            frame = cv2.bitwise_and(frame, frame, mask=self.mask)
        background = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        background = cv2.medianBlur(background, 3)
        self.principal_point = np.array([self.frame_height * self.scale / 2, self.frame_width * self.scale / 2])
        self.DiamondSpace = DiamondSpace(
            d=min(self.frame_height * self.scale, self.frame_width * self.scale) / 2, size=256*2)
        while True:
            start = time()

            flag, frame = self.camera.read()

            if not flag:
                print('no frame grabbed!')
                break

            if self.scale != 1:
                frame = cv2.resize(frame, (0, 0), fx=self.scale, fy=self.scale, interpolation=cv2.INTER_LINEAR)
            if self.roi is not None:
                frame = cv2.bitwise_and(frame, frame, mask=self.mask)
            self.current_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            current_medianBlur = cv2.medianBlur(self.current_frame, 3)
            background = cv2.addWeighted(current_medianBlur, 0.05, background, 0.95, 0)

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
                            if self.get_track_length(fp) > 1:
                                temp = fp.copy()
                                self.tracks.append(temp)

            if self.frame_count % self.detectInterval == 0:
                mask = np.zeros_like(self.current_frame)
                boxes = self.detect_car(frame)
                if len(boxes) > 0:
                    for box in boxes:
                        mask[box[1]:box[3], box[0]:box[2]] = 255
                        points = self.lines_from_box(current_medianBlur, background, box, threshold=threshold- 0.3,
                                                     winSize=9,
                                                     drawFlag=view_process)
                        if points is not None:
                            self.edgelets.append(points)
                    for x, y in [np.int32(tr[-1]) for tr in self.features]:
                        cv2.circle(mask, (x, y), 10, 0, -1)

                # good tracker
                p = cv2.goodFeaturesToTrack(self.current_frame, mask=mask, **good_features_parameters)
                if p is not None:
                    for x, y in np.float32(p).reshape(-1, 2):
                        self.features.append([(x, y)])
                print(f'final_time:{(time() - start):.6f}')

            self.frame_count += 1
            self.previous_frame = self.current_frame.copy()
            cv2.imshow('frame', frame)
            print(f'fps:{1 / (time() - start)}')
            if cv2.waitKey(1) == 27:
                # cv2.imwrite('frame1.jpg', frame)
                self.camera.release()
                cv2.destroyAllWindows()
                break


    def find_corresponding_box(self, current_box, threshold = 50):
            # Implement logic to find corresponding box based on tracked points and previous boxes
            # You may use similarity metrics or other matching techniques
            # For simplicity, this example assumes one-to-one matching based on proximity
    
            corresponding_prj = None
            min_distance = threshold
            box_corners = [i for i in current_box]
            xmin = box_corners[0]
            ymin = box_corners[1]
            xmax = box_corners[2]
            ymax = box_corners[3]
            centroid_current = np.array([xmin + (xmax -xmin)/2, ymin + (ymax-ymin)/2])
            corresponding_centroid = None
    
            for prj in self.projections:
    
                prev_box = prj.box
                box_corners = [i for i in prev_box]
                xmin = box_corners[0]
                ymin = box_corners[1]
                xmax = box_corners[2]
                ymax = box_corners[3]
                centroid_previous = np.array([xmin + (xmax -xmin)/2, ymin + (ymax-ymin)/2])
    
                distance = np.linalg.norm(centroid_current - centroid_previous)
                print("distance :", distance)
    
                if distance < min_distance:  # Set an appropriate threshold
                    corresponding_prj = prj
                    corresponding_centroid = centroid_previous
                    min_distance = distance
    
    
            return corresponding_prj, centroid_current, corresponding_centroid


    def detect_orientation(self):
        self.frame_count = 0
        orientations = []
        # Variable for color to draw optical flow track
        color = (0, 255, 0)
        # self.perspective = np.float32([[-2.63354909e+00, -1.14758510e+01,  5.44305490e+03],
                                         # [-4.70089448e-01, -2.34697869e+01,  1.15890480e+04],
                                         # [-1.08570046e-04, -3.75704350e-03,  1.00000000e+00]]
                                        # )
        # self.target_shape = [4600, 3900]                                                                 

        while True:
            flag, original_frame = self.camera.read()
            if not flag:
                self.logger.error('no frame grabbed!')
                print('no frame grabbed!')
                break
            else:
                frame = original_frame.copy()
                vp = self.vp_1[:2]

                if self.scale != 1:
                    frame = cv2.resize(frame, (0, 0), fx=self.scale, fy=self.scale, interpolation=cv2.INTER_LINEAR)
                    vp = self.vp_1 * self.scale
                    #cv2.circle(frame, tuple(map(int, vp[:2])), 5, (255, 0, 0), -1)

                in_frame = frame.copy()
                if self.roi is not None:
                    frame = cv2.bitwise_and(frame, frame, mask=self.mask)

                start = time()


                # Create a list of Projection instances
                projection_list = []
                self.current_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                # Creates an image filled with zero intensities with the same dimensions as the frame - for later drawing purposes
                mask = np.zeros_like(frame)
                output = frame.copy()
                warp = cv2.warpPerspective(frame, self.perspective, self.target_shape)


                mask = np.zeros_like(self.current_frame)
                boxes = self.detect_car(frame)
    
                # Initialize an empty NumPy array for points outside the loop
                points = np.empty((0, 2), dtype=np.float32)
                print(f"detect: {time() - start}")
                
                for i, box in enumerate(boxes):
    
                    t = get_lr(box, vp[:2])
                    #print("type :", t)
                    if t == 'r':
                        color = cv_colors.RED.value
                    else:
                        color = cv_colors.BLUE.value
                        
                    cv2.rectangle(frame, box[:2], box[2:], color, 1)
                    #prj = Projection(box = box, type = t)
                    projection_list.append(Projection(box = box, type = t))
    
                    xmin, ymin, xmax, ymax = box
    
    
    
                    print("vp :", vp)
                    #vis_model(frame, edgelets1, vp1, output_path = None)
                    #vp = vp1 / vp1[2]
                    print(f"vp : {vp}")
                
                    # Draw arrowed line from vp1 through the middle of the lower edge of the box on image 1
                    middle_lower_edge = ((box[0] + box[2]) // 2, box[3])
                    direction = middle_lower_edge - vp[:2]
                    direction = direction / np.linalg.norm(direction)
                    scale = 70
                    #cv2.arrowedLine(frame, middle_lower_edge,  tuple(map(int,middle_lower_edge + scale * direction)), (255, 0, 0), 2) #, tipLength=0.1)
                    #cv2.arrowedLine(frame, tuple(map(int, vp[:2])), middle_lower_edge, (255, 0, 0), 2)
    
                    diff = middle_lower_edge - vp[:2]# - middle_lower_edge#p2 - p1
                    #orientation = math.degrees(np.arctan2(diff[1],  diff[0]))
                    
                    # Concatenate the starting and ending points into a single matrix
                    # if p1[1] > p2[1]:
                        # p1, p2 = p2, p1
                    points = np.array([middle_lower_edge, middle_lower_edge + diff/10, [xmin, ymax], [xmin, ymin], [xmax, ymin], [xmax, ymax]], dtype=np.float64)
                    #points /=  self.scale
                    pointsW = np.concatenate([points, np.ones((len(points), 1))], axis=1)
                    points_IPM = pointsW @ self.perspective.T
                    points_IPM = points_IPM / points_IPM[:, -1].reshape(-1, 1)
                    diff = points_IPM[1] - points_IPM[0]  # opposite!!!
                    orientation_IPM = np.arctan2(diff[1], diff[0])
    
                    # # Draw the longest vector on the frame
                    # # Convert points to integers if needed
                    pt1 = tuple(points_IPM[0, :2].astype(int))
                    pt2 = tuple(points_IPM[1, :2].astype(int))
                    
                    # # Compute the adjusted endpoint to control the length of the line
                    arrow_length = 10
                    adjusted_endpoint = pt1[0] + int(arrow_length * (-pt2[0] + pt1[0])), pt1[1] + int(arrow_length * (-pt2[1] + pt1[1]))
                    
                    # # Draw the line on the warp image
                    warp = cv2.arrowedLine(warp, pt1, adjusted_endpoint, color, 1)
                    box_ipm = np.array([points_IPM[2, :2], points_IPM[3, :2], points_IPM[4, :2], points_IPM[5, :2]], dtype=np.float64)

                    warp = cv2.polylines(warp, [np.array(box_ipm, dtype=np.int64)], isClosed=True, color=cv_colors.BLUE.value, thickness=1)
    
                    projection_list[-1].quadrangle = box_ipm
                    projection_list[-1].orientation = diff[:2]
    
                    
                    # Find corresponding box based on tracked points and previous boxes
                    if self.projections is not None:
                        prev_prj, centroid_current, centroid_previous = self.find_corresponding_box(box)
                        if prev_prj is not None:
                            if debug:
                                cv2.rectangle(frame, prev_prj.box[:2], prev_prj.box[2:], cv_colors.YELLOW.value, 1)
                                # Draw lines connecting centroids
                                # print("centroid_current:", centroid_current)
                                # print("centroid_previous:", centroid_previous)
                                cv2.line(frame, (int(centroid_current[0]), int(centroid_current[1])), (int(centroid_previous[0]), int(centroid_previous[1])), cv_colors.PURPLE.value, 5)
                            
                            # if prev_prj.quadrangle is None:
                                # continue
    
                            prev_box_ipm = prev_prj.quadrangle
                            warp = cv2.polylines(warp, [np.array(prev_box_ipm, dtype=np.int64)], isClosed=True, color=(0, 255, 255), thickness=1)
                            
                            try:
                                solutions = find_rectangle_sides(prev_box_ipm, prev_prj.orientation, prev_prj.type,  box_ipm, diff[:2], t)
                            except:
                                print("solution error")
                                continue
    
                            assert len(solutions) < 2,  f"Invalid number of solutions: {len(solutions)}"
                            if len(solutions) == 0:
                                length = prev_prj.length
                            else:
                                length = solutions[0][0]
                            if length is not None: #for solution in solutions:
   
                                projection_list[-1].lenght = length

                                if t == 'r':
                                    w_points = get_bottom_r(box_ipm, orientation_IPM, length = length, bev_input = warp, color = cv_colors.RED.value)
                                else:
                                    w_points = get_bottom_l(box_ipm, orientation_IPM, length = length, bev_input = warp, color = cv_colors.RED.value)  #get_bottom2(box_ipm, orientation_IPM, width = w, length = l, bev_input = warp, color = cv_colors.RED.value, thickness = 50, epsilon = 10)
                                
                                if w_points is None:
                                    #projection_list.append(prj)
                                    continue


                                
                                pointsW = np.concatenate([w_points, np.ones((len(w_points), 1))], axis=1)
                                #print(pointsW.dtype)
    
                                lower_face = pointsW @ np.linalg.inv(self.perspective).T
                                lower_face = lower_face / lower_face[:, -1].reshape(-1, 1)
                                print("Lower Face:\n", lower_face)
                                
                                # Assuming get_upper_face is a function that extracts the upper face based on box and lower_face[:, :2]
                                #left_van = vp[:2]
                                if t == 'r':
                                    upper_face = get_upper_face(box, lower_face[:, :2], left_van = vp[:2], im = frame)
                                else:
                                    upper_face = get_upper_face_l(box, lower_face[:, :2], left_van = vp[:2], im = frame)
    
                                print("Upper Face:\n", upper_face)
                                
                                frame = draw_cube(frame, lower_face[: ,:2], upper_face)
                                
                                warp = cv2.polylines(warp, [np.array(w_points, dtype=np.int64)], isClosed=True, color=cv_colors.RED.value, thickness=2)
                                    
    
    
                            
                            print(f"keypoint: {time() - start}")

                  
 

            self.previous_frame = self.current_frame.copy()
            self.projections = projection_list.copy()

            self.frame_count += 1
            
            x_scale, y_scale = scale_image(warp.shape, (2*300, 2*600), force=True)
            warp = cv2.resize(warp, (-1, -1), fx=x_scale, fy=y_scale)

            # print(f"fps:{(1 / (time() - start)):.3f}")
            cv2.imshow('warp', warp)
            cv2.imshow('frame', frame)
            
            cv2.imwrite(f'original/frame_{self.frame_count}.png', in_frame)
            cv2.imwrite(f'output/frame_{self.frame_count}.png', frame)
            cv2.imwrite(f'output/warp_{self.frame_count}.png', warp)

            if cv2.waitKey(1) & 0xFF == 27:
                print(f"平均角度为{np.average(np.abs(orientations))}")
                cv2.imwrite('frame1.jpg', frame)
                self.camera.release()
                cv2.destroyAllWindows()
                break


    def detect_orientation_seg(self):
        self.frame_count = 0
        orientations = []
        # Variable for color to draw optical flow track
        color = (0, 255, 0)

        while True:
            flag, original_frame = self.camera.read()
            if not flag:
                self.logger.error('no frame grabbed!')
                print('no frame grabbed!')
                break
            else:
                frame = original_frame.copy()
                # frame =  drawCalibration(frame, self.vp_1[:2], self.vp_2[:2], self.vp_3[:2])
                # frame = drawViewpoint(frame, self.principal_point,
                                                        # self.vp_1,
                                                        # self.vp_2, self.vp_3, scale = 1)

                if self.scale != 1:
                    frame = cv2.resize(frame, (0, 0), fx=self.scale, fy=self.scale, interpolation=cv2.INTER_LINEAR)

                if self.roi is not None:
                    frame = cv2.bitwise_and(frame, frame, mask=self.mask)

                start = time()

                self.current_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                # Creates an image filled with zero intensities with the same dimensions as the frame - for later drawing purposes
                mask = np.zeros_like(frame)
                output = frame.copy()
                warp = cv2.warpPerspective(frame, self.perspective, self.target_shape)

                if self.previous_frame is not None:
                    pimg, cimg = self.previous_frame, self.current_frame
                    prev = cv2.goodFeaturesToTrack(pimg, mask=None, **good_features_parameters)
                    next_points, status, error = cv2.calcOpticalFlowPyrLK(pimg, cimg, prev, None, **optical_flow_parameters)
                
                    # Selects good feature points for previous position
                    good_old = prev[status == 1].astype(int)
                
                    # Selects good feature points for next position
                    good_new = next_points[status == 1].astype(int)
                    displacement_vectors = good_new - good_old
                    vector_magnitudes_squared = np.sum(displacement_vectors**2, axis=1)
                    
                    # # Draw displacement vectors on the combined frame
                    # for i in range(len(good_old)):
                        # pt1 = tuple(good_old[i])
                        # pt2 = tuple(good_new[i])
                        # cv2.arrowedLine(frame, pt1, pt2, color, 2)
                 
                    mask = np.zeros_like(self.current_frame)
                    boxes, classes = self.detect_car(frame)
    
                    # Initialize an empty NumPy array for points outside the loop
                    points = np.empty((0, 2), dtype=np.float32)
                    print(f"detect: {time() - start}")
    
                    # Declare feature_c outside the loop
                    # feature_c = None
                    # good_features_index = None

                    for i, box in enumerate(boxes):

                        cls = classes[i]

                        box_corners = [i for i in box]
                        xmin = box_corners[0]
                        ymin = box_corners[1]
                        xmax = box_corners[2]
                        ymax = box_corners[3]
                        
                        #drawViewpoint(frame, [xmin, ymin, 1], self.vp_1, self.vp_2, self.vp_3)
                        
                        
                        # Create a boolean mask based on the ranges
                        range_mask = (
                            (good_old[:, 0] >= xmin) & (good_old[:, 0] <= xmax) &
                            (good_old[:, 1] >= ymin) & (good_old[:, 1] <= ymax)
                        )
                        # Check if there are any points in the range
                        if not np.any(range_mask):
                            continue  # Skip this box if there are no points in the range
                            
                        # Find the index of the point with the maximum y-coordinate
                        index_of_max_y = np.argmax(good_new[:, 1])
                        
                        # Extract the point with the maximum y-coordinate
                        point_with_max_y = good_new[index_of_max_y]
                        
                                                
                        # Draw the vector or point on the frame
                        frame = cv2.arrowedLine(frame, good_old[index_of_max_y], point_with_max_y, cv_colors.RED.value, 5)
    
                        # Calculate the squared magnitude of each vector for the filtered vectors
                        filtered_vector_magnitudes_squared = vector_magnitudes_squared[range_mask]

                        # Find the index of the vector with the maximum magnitude
                        index_of_longest_vector = np.argmax(filtered_vector_magnitudes_squared)
                        
                        # Extract the longest vector
                        longest_vector = displacement_vectors[range_mask][index_of_longest_vector]

                        # # Get the starting and ending points of the longest vector
                        p1, p2 = good_old[range_mask][index_of_longest_vector], good_new[range_mask][index_of_longest_vector]
                        
                        # Draw the longest vector on the frame
                        frame = cv2.arrowedLine(frame, p1, p2, cv_colors.BLUE.value, 2)
                        
                        diff = p2 - p1
                        orientation = math.degrees(np.arctan2(diff[1],  diff[0]))
                        
                        # Concatenate the starting and ending points into a single matrix

                        points = np.array([p1, p2, [xmin, ymax], [xmin, ymin], [xmax, ymin], [xmax, ymax]], dtype=np.float64)
                        #points /=  self.scale
                        pointsW = np.concatenate([points, np.ones((len(points), 1))], axis=1)
                        points_IPM = pointsW @ self.perspective.T
                        points_IPM = points_IPM / points_IPM[:, -1].reshape(-1, 1)
                        diff = points_IPM[1] - points_IPM[0]
                        orientation_IPM = math.degrees(np.arctan2(diff[1], diff[0]))

                        # # Draw the longest vector on the frame
                        # # Convert points to integers if needed
                        pt1 = tuple(points_IPM[0, :2].astype(int))
                        pt2 = tuple(points_IPM[1, :2].astype(int))
                        
                        # Compute the adjusted endpoint to control the length of the line
                        # arrow_length = 10
                        # adjusted_endpoint = pt1[0] + int(arrow_length * (pt2[0] - pt1[0])), pt1[1] + int(arrow_length * (pt2[1] - pt1[1]))
                        
                        # # Draw the line on the warp image
                        # warp = cv2.arrowedLine(warp, pt1, adjusted_endpoint, color, 5)
                        box_ipm = [points_IPM[2, :2], points_IPM[3, :2], points_IPM[4, :2], points_IPM[5, :2]]
                        
                        # Draw the polyline on the warp image
                        warp = cv2.polylines(warp, [np.array(box_ipm, dtype=np.int64)], isClosed=True, color=color, thickness=1)

                        print(f"keypoint: {time() - start}")

                        if debug:
                            cv2.putText(frame, "angle:" + str(np.around(orientation, 3)), (xmin, ymax),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
                            cv2.putText(frame, "angle ipm:" + str(np.around(orientation_IPM, 3)), (xmax, ymax),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                            cv2.putText(frame, "class:" + cls, (xmin, ymin),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                            cv2.rectangle(frame, box[:2], box[-2:], (0, 255, 0), 1)

                        # error, w_points = get_bottom2(box_ipm, math.radians(orientation_IPM), w = 100, h = 200, bev_in = warp, color = cv_colors.RED.value, thickness = 50, eps = 10)
                        # pointsW = np.concatenate([w_points, np.ones((len(w_points), 1))], axis=1)

                        # lower_face = pointsW @ np.linalg.inv(self.perspective).T
                        # lower_face = lower_face / lower_face[:, -1].reshape(-1, 1)
                        # print("Lower Face:\n", lower_face)
                        
                        # # Assuming get_upper_face is a function that extracts the upper face based on box and lower_face[:, :2]
                        # upper_face = get_upper_face(box, lower_face[:, :2], frame)
                        # print("Upper Face:\n", upper_face)
                        
                        # #frame = draw_cube(frame, lower_face[: ,:2], upper_face)
                        
                        # warp = cv2.polylines(warp, [np.array(w_points, dtype=np.int64)], isClosed=True, color=cv_colors.RED.value, thickness=2)
            
                    # Updates previous good feature points
                    prev = good_new.reshape(-1, 1, 2)
                    
                    
                    # # Compute optical flow
                    # optical_flow = compute_optical_flow(self.previous_frame, self.current_frame)
                    
                    # #Convert flow field to RGB for visualization
                    # rgb_flow = flow_to_rgb(optical_flow)
                    
                    # #Draw flow vectors on the original frame
                    # result_image = draw_flow(rgb_flow, optical_flow)
                    
                    # #Display the result
                    # cv2.imshow('Optical Flow', result_image)

                    # warp_flow = cv2.warpPerspective(result_image, self.perspective, self.target_shape)
                    # # #warp_flow = segment(warp_flow, bg_color= (0, 0, 0))
                    # # #warp_flow = cv2.addWeighted(warp.astype(np.float32), 0.2, warp_flow.astype(np.float32), 0.8, 0)

                    # # x_scale, y_scale = scale_image(warp_flow.shape, (2*300, 2*600), force=True)
                    # # warp_flow = cv2.resize(warp_flow, (-1, -1), fx=x_scale, fy=y_scale)
    
                    # # # print(f"fps:{(1 / (time() - start)):.3f}")
                    # cv2.imshow('warp flow 1', warp_flow.astype(np.uint8))

                    # warp_prev = cv2.warpPerspective(self.previous_frame, self.perspective, self.target_shape)
                    # warp_curr = cv2.warpPerspective(self.current_frame, self.perspective, self.target_shape)
                    
                    # x_scale, y_scale = scale_image(warp_prev.shape, (300, 600), force=True)
                    # warp_prev = cv2.resize(warp_prev, (-1, -1), fx=x_scale, fy=y_scale)
                    # x_scale, y_scale = scale_image(warp_curr.shape, (300, 600), force=True)
                    # warp_curr = cv2.resize(warp_curr, (-1, -1), fx=x_scale, fy=y_scale)
                    
                    # x_scale, y_scale = scale_image(warp.shape, (300, 600), force=True)
                    # warp = cv2.resize(warp, (-1, -1), fx=x_scale, fy=y_scale)
                    
                    
                    # ############################################
                    # flow = compute_optical_flow(warp_prev, warp_curr) #cv2.calcOpticalFlowFarneback(gray1, gray2, None, 0.5, 3, 15, 3, 5, 1.2, 0)

                    # # Compute magnitude and angle using cv2.cartToPolar()
                    # magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
                    # #Create a mask in HSV Color Space.
                    # hsv = np.zeros((300, 600, 3), dtype=np.uint32) #     np.zeros_like(warp_curr)
                    # # Sets image saturation to maximum.
                    # hsv[..., 1] = 255
                    # # Sets image hue according to the optical flow direction
                    # hsv[..., 0] = angle * 180 / np.pi / 2
                    # # Sets image value according to the optical flow magnitude (normalized)
                    # hsv[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
                    
                    # # Normalize magnitude to the range [0, 1]
                    # normalized_magnitude =  cv2.normalize(magnitude, None, 0, 180, cv2.NORM_MINMAX)
                    
                    # # Convert angles from radians to degrees and perform modulo 180
                    # angle_degrees = np.degrees(angle) % 180
                    
                    # # Normalize angle to the range [0, 180] degrees
                    # normalized_angle_degrees = cv2.normalize(angle_degrees, None, 0, 180, cv2.NORM_MINMAX)
                    
                    # # Create an HSV image with the same shape as the original flow
                    # normalized_hsv = np.zeros((flow.shape[0], flow.shape[1], 3), dtype=np.uint32)
                    
                    # # Sets image hue according to the normalized optical flow direction
                    # normalized_hsv[..., 0] = (normalized_angle_degrees).astype(np.uint32)
                    
                    # # Sets image value according to the normalized optical flow magnitude
                    # normalized_hsv[..., 2] = (normalized_magnitude).astype(np.uint32)
                    
                    # normalized_hsv[..., 1] = 255
                    
                    # # from random import random
                    
                    # # # Disable all logging output
                    # # logging.disable(logging.CRITICAL)
                    
                    # sigma = 2.0
                    # neighbor = 8
                    # K = 1000*1000.0
                    # min_comp_size = 5
                    # seg, forest, graph_edges, th = get_segmented_array(sigma, neighbor, K, min_comp_size, hsv, normalized_hsv)
                    # warp_seg = cv2.addWeighted(warp.astype(np.float32), 0.3, seg.astype(np.float32), 0.7, 0)
                    
                    # x_scale, y_scale = scale_image(warp_seg.shape, (2*300, 2*600), force=True)
                    # warp_seg = cv2.resize(warp_seg, (-1, -1), fx=x_scale, fy=y_scale)
                    # #cv2.imwrite("output/seg.jpg", seg)
                    # cv2.imshow("cust seg" , warp_seg.astype(np.uint8))
                    #############################################
                    
                    # #Convert flow field to RGB for visualization
                    # warp_flow = compute_optical_flow(warp_prev, warp_curr)
                    # warp_rgb_flow = flow_to_rgb(warp_flow)
                    # warp_rgb_flow = segment(warp_rgb_flow, bg_color= (1, 1, 1))
                    # warp_rgb_flow = cv2.addWeighted(warp.astype(np.float32), 0.2, warp_rgb_flow.astype(np.float32), 0.8, 0)
                    
                    # x_scale, y_scale = scale_image(warp_rgb_flow.shape, (2*300, 2*600), force=True)
                    # warp_rgb_flow = cv2.resize(warp_rgb_flow, (-1, -1), fx=x_scale, fy=y_scale)
    
                    # # print(f"fps:{(1 / (time() - start)):.3f}")
                    # cv2.imshow('warp rgb flow ', warp_rgb_flow.astype(np.uint8))

                    #cv2.waitKey(0)

                self.previous_frame = self.current_frame.copy()
                self.frame_count += 1
                
                x_scale, y_scale = scale_image(warp.shape, (2*300, 2*600), force=True)
                warp = cv2.resize(warp, (-1, -1), fx=x_scale, fy=y_scale)

                # print(f"fps:{(1 / (time() - start)):.3f}")
                #cv2.imshow('warp', warp)
                cv2.imshow('detect', frame)
                # cv2.imshow('warp', warp)

                if cv2.waitKey(1) & 0xFF == 27:
                    ## 计算检测得到的平均方向角度
                    print(f"平均角度为{np.average(np.abs(orientations))}")
                    cv2.imwrite('frame1.jpg', frame)
                    self.camera.release()
                    cv2.destroyAllWindows()
                    break

    def load_yolo_model(self, engine_file, class_json, verbose=False):
        self.logger.info("loading yolov5 tensorRT engine.......")
        self.yolo = YoloTensorrt(engine_file, class_json, verbose=verbose)
        self.logger.info("yolov5 tensorRT model loads successfully")

    def detect_car(self, frame, threshold=0.5):
        self.yolo.reload_images(frame)
        cls, bs = self.yolo.infer(threshold=threshold)
        if cls:
            [classes], [boxes] = cls, bs
            # filter cars
            index = [True if c in ['car', 'bus', 'truck', ] else False for c in classes]
            cars_boxes = boxes[index]
            return cars_boxes
        else:
            return []

    def load_keypoint_model(self, ):
        self.logger.info("loading keypoint detector model")
        self.perdictor = Predictor(checkpoint="shufflenetv2k16-apollo-24")

        # warmup
        img = np.ones((640, 640, 3), dtype=np.uint8)
        self.perdictor.numpy_image(img)

    def get_keypoints(self, image, all=False):
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        predictions, gt_anns, image_meta = self.perdictor.numpy_image(img)
        if predictions:
            data = np.vstack(predictions[0].data)
            index = np.where(data[:, -1] > 0)[0]
            # pair_flag, pair_keypoints = get_pair_keypoints(index, ktype='vertical')
            pair_flag, pair_keypoints = get_pair_keypoints(index, ktype='horizontal')
            if pair_flag:
                if all:
                    return pair_flag, data[index]
                else:
                    return pair_flag, data[pair_keypoints[0]]
            else:
                return pair_flag, None
        else:
            return False, None

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

    def draw_all_tracks(self, save_name='all_tracks.jpg', save_path='./'):
        display = self.init_frame.copy()
        cv2.polylines(display, [np.int32(tr) for tr in self.tracks], False, (0, 255, 0))
        cv2.imwrite(os.path.join(save_path, save_name), display)

    def lines_from_box(self, current, background, box, winSize=9, threshold=0.5, drawFlag=False):

        vehicle_current = current[box[1]:box[3], box[0]:box[2]]

        vehicle_background = background[box[1]:box[3], box[0]:box[2]]
        # vehicle_current_preprocess = self.clahe.apply(vehicle_current)
        # vehicle_background_preprocess = self.clahe.apply(vehicle_background)
        origin_edges = cv2.Canny(vehicle_current, 255 / 2, 255)
        edges = cv2.Canny(vehicle_current, 255 / 2, 255) - cv2.Canny(vehicle_background, 255 / 2, 255)
        orientation, quality = neighborhood(edges, winSize=winSize)
        accumulation, t = accumulate_orientation(orientation, quality, winSize=winSize, threshold=threshold)

        res = cv2.addWeighted(accumulation.astype(np.float32), 0.9, edges.astype(np.float32), 0.1, 0)
        _, res = cv2.threshold(res.astype(np.uint8), 127, 255, cv2.THRESH_OTSU)
        #_, res = cv2.threshold(res, np.percentile(res[res!=0], 100 * (1 - threshold)), 255, cv2.THRESH_BINARY)
        # thres, edges = cv2.threshold(quality, 0, 255, cv2.THRESH_OTSU)
        # lines = get_lines(edges, orientation, box)
        points = cv2.HoughLinesP(res, 1.0, np.pi / 180, 30, minLineLength=10, maxLineGap=20)

        if points is not None:
            points = points.reshape(-1, 4)
            points[:, [0, 2]] += box[0]
            points[:, [1, 3]] += box[1]
            if drawFlag:
                mask = np.zeros(current.shape)
                for p in points:
                    x1, y1, x2, y2 = p
                    cv2.line(mask, (x1, y1), (x2, y2), 255, 2)
                cv2.imshow('frame', mask)
                cv2.imshow('car', vehicle_current)
                cv2.imshow('edges', edges)
                cv2.imshow('res', res)
                if cv2.waitKey(0) == 27:
                    cv2.imwrite('./origin_edge.jpg', origin_edges)
                    cv2.imwrite('./edges.jpg', edges)
                    cv2.imwrite('./res.jpg', res)
                cv2.destroyWindow('frame')
                cv2.destroyWindow('car')
                cv2.destroyWindow('edges')
                cv2.destroyWindow('res')
        return points

    def get_vp1(self, save_path, visualize=False):
        lines = cvt_diamond_space(self.tracks)
        self.DiamondSpace.insert(lines)
        vps, values, vpd_s = self.DiamondSpace.find_peaks(t=0.8, )
        if len(vps) <= 0: vps, values, vpd_s = self.DiamondSpace.find_peaks(t=0.7, )
        if len(vps) <= 0: vps, values, vpd_s = self.DiamondSpace.find_peaks(t=0.5, )
        #if len(vps) <= 0: vps, values, vpd_s = self.DiamondSpace.find_peaks(t=0.3, )
        if len(vps) <= 0:
            raise Exception("Fail to detect the first vanishing point.")
        # vps中权重最大的一个点取为第一消失点
        self.vp_1 = vps[0][:2]

        if visualize:
            self.draw_all_tracks(save_path=f"{save_path}/new/")
            img = cv2.cvtColor(self.init_frame, cv2.COLOR_BGR2RGB)
            print("numbers of vps", len(vps))
            size = self.DiamondSpace.size
            scale = self.DiamondSpace.scale

            # 第一消失点可视化
            _, ax = plt.subplots(1, 2, figsize=(30, 20))
            ax[0].imshow(self.DiamondSpace.attach_spaces(), cmap="Greys", extent=(
                (-size + 0.5) / scale, (size - 0.5) / scale, (size - 0.5) / scale,
                (-size + 0.5) / scale))
            ax[0].set(xticks=np.linspace(-size + 1, size - 1, 5) / scale,
                      yticks=np.linspace(-size + 1, size - 1, 5) / scale)
            ax[0].plot(vpd_s[0, 0] / scale, vpd_s[0, 1] / scale, "ro", markersize=11)
            # ax[0].plot(vpd_s[1:, 0] / scale, vpd_s[1:, 1] / scale, "go", markersize=11)
            ax[0].invert_yaxis()

            ax[1].imshow(img)
            ax[1].set(title="first vanishing point in image")
            ax[1].plot(vps[0, 0], vps[0, 1], 'ro', markersize=11)
            # ax[1].plot(vps[1:, 0], vps[1:, 1], 'go', markersize=11)

            ax[0].set_title(label="Accumulator", fontsize=30)
            ax[0].tick_params(axis='x', labelsize=20)
            ax[0].tick_params(axis='y', labelsize=20)
            ax[1].tick_params(axis='x', labelsize=20)
            ax[1].tick_params(axis='y', labelsize=20)
            ax[1].set_title("first vanishing point in image", fontsize=30)

            plt.savefig(f'{save_path}/new/first_vp1.jpg')

    def get_vp2(self, save_path, visualize=False):
        points = np.vstack(self.edgelets)
        lines = start_end_line(points)
        # index = self.DiamondSpace.filter_lines_from_vp(self.vp_1, lines, 40)
        index = self.DiamondSpace.filter_lines_from_vp(self.vp_1, lines, min(self.frame_width * self.scale,
                                                                             self.frame_height * self.scale) / 2)
        vps, values, vpd_s = self.DiamondSpace.find_peaks(t=0.8, )
        # vps中权重最大的一个点取为第二消失点
        if len(vps) <= 0:
            raise Exception("Fail to detect the second vanishing point.")

        ## get the best
        # first_filter = vps[np.bitwise_and(values == values.max(), vps[:, -1] == 1)]
        # self.vp_2 = first_filter[np.linalg.norm(first_filter, axis=1).argmin()][:2]

        self.vp_2 = vps[0][:2]

        if visualize:
            img = cv2.cvtColor(self.init_frame, cv2.COLOR_BGR2RGB)
            print("numbers of vps", len(vps))
            size = self.DiamondSpace.size
            scale = self.DiamondSpace.scale
            edgelets = draw_point_line(self.init_frame, points, visualFlag=False)
            edgelets_filter = draw_point_line(self.init_frame, points[index], visualFlag=False)
            cv2.imwrite(f'{save_path}/new/edgelets.jpg', edgelets)
            cv2.imwrite(f'{save_path}/new/edgelets_filter.jpg', edgelets_filter)
            # 第二消失点可视化
            _, ax = plt.subplots(1, 2, figsize=(20, 10))
            ax[0].imshow(self.DiamondSpace.attach_spaces(), cmap="Greys", extent=(
                (-size + 0.5) / scale, (size - 0.5) / scale, (size - 0.5) / scale,
                (-size + 0.5) / scale))
            ax[0].set(xticks=np.linspace(-size + 1, size - 1, 5) / scale,
                      yticks=np.linspace(-size + 1, size - 1, 5) / scale)
            ax[0].plot(vpd_s[0, 0] / scale, vpd_s[0, 1] / scale, "ro", markersize=11)
            # ax[0].plot(vpd_s[1:, 0] / scale, vpd_s[1:, 1] / scale, "go", markersize=11)
            ax[0].invert_yaxis()

            ax[1].imshow(img)
            ax[1].set(xticks=np.linspace(-18e4, 0, 5))
            # ax[1].set(title=)

            ax[1].plot(vps[0, 0], vps[0, 1], 'ro', markersize=11)

            ax[0].set_title(label="Accumulator", fontsize=30)
            ax[0].tick_params(axis='x', labelsize=20)
            ax[0].tick_params(axis='y', labelsize=20)
            ax[1].tick_params(axis='x', labelsize=20)
            ax[1].tick_params(axis='y', labelsize=20)
            ax[1].set_title("second vanishing point in image", fontsize=30)
            plt.savefig(f'{save_path}/new/first_vp2.jpg')

    def save_calibration(self, save_path, visualize=False):
        vp1, vp2, vp3, pp, roadPlane, focal, intrinsic_matrix, rotation_matrix = computeCameraCalibration(
            self.vp_1 / self.scale,
            self.vp_2 / self.scale,
            self.principal_point / self.scale)

        calibration = dict(vp1=vp1, vp2=vp2, vp3=vp3, principal_point=pp,
                           roadPlane=roadPlane, focal=focal, intrinsic=intrinsic_matrix, rotation=rotation_matrix)
        with open(f'{save_path}/new/calibrations.npy', 'wb') as f:
            np.save(f, calibration)

        if visualize:
            img = cv2.cvtColor(self.init_frame, cv2.COLOR_BGR2RGB)
            plt.figure(figsize=(20, 20))
            plt.imshow(img)
            plt.plot(vp1[0], vp1[1], 'ro', markersize=11)
            plt.plot(vp2[0], vp2[1], 'ro', markersize=11)
            plt.plot(vp3[0], vp3[1], 'ro', markersize=11)
            plt.savefig(f'{save_path}/new/all_vps.jpg')
        return calibration

    def calibrate(self, save_path, visualize=False, ):
        self.logger.info(f"saving the calibration information to {save_path}")

        if not os.path.exists(os.path.join(save_path, 'new')):
            os.makedirs(os.path.join(save_path, 'new'), )
        self.get_vp1(save_path, visualize)
        self.get_vp2(save_path, visualize)
        self.save_calibration(save_path, visualize)

    def load_calibration(self, path, dst_shape=(1600, 900), strict=False):
        self.logger.info(f"loading the calibration information from {path}")
        try:
            with open(path, 'rb') as f:
                calibration = np.load(f, allow_pickle=True).tolist()
            print(calibration)
            self.vp_1 = calibration.get("vp1")
            self.vp_2 = calibration.get("vp2")
            self.vp_3 = calibration.get("vp3")
            self.principal_point = calibration.get('principal_point')
            self.focal = calibration.get('focal')
            self.roadPlane = calibration.get('roadPlane')
            self.intrinsic = calibration.get('intrinsic')
            self.rotation = calibration.get('rotation')
            target_shape = dst_shape if dst_shape[0] * dst_shape[1] > self.frame_width * self.frame_height else (
                self.frame_width, self.frame_height)
            self.perspective = convertToBirdView(self.intrinsic, self.rotation, (self.frame_width, self.frame_height),
                                                 target_shape=target_shape, strict=strict)
        except Exception as e:
            self.logger.error(f'failed to load calibration, error message{e}')
            raise e
