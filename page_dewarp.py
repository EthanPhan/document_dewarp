#!/usr/bin88/env python
######################################################################
# page_dewarp.py - Proof-of-concept of page-dewarping based on a
# "cubic sheet" model. Requires OpenCV (version 3 or greater),
# PIL/Pillow, and scipy.optimize.
######################################################################
# Author:  Matt Zucker
# Date:    July 2016
# License: MIT License (see LICENSE.txt)
######################################################################

import os
import sys
import datetime
import cv2
from PIL import Image
import numpy as np
import scipy.optimize
from scipy.integrate import quad
from scipy.optimize import minimize, least_squares, root
from numpy.linalg import norm
import levmar

# for some reason pylint complains about cv2 members being undefined :(
# pylint: disable=E1101

PAGE_MARGIN_X = 50       # reduced px to ignore near L/R edge
PAGE_MARGIN_Y = 20       # reduced px to ignore near T/B edge

OUTPUT_ZOOM = 1.0        # how much to zoom output relative to *original* image
OUTPUT_DPI = 300         # just affects stated DPI of PNG, not appearance
REMAP_DECIMATE = 16      # downscaling factor for remapping image

ADAPTIVE_WINSZ = 55      # window size for adaptive threshold in reduced px

TEXT_MIN_WIDTH = 15      # min reduced px width of detected text contour
TEXT_MIN_HEIGHT = 2      # min reduced px height of detected text contour
TEXT_MIN_ASPECT = 1.5    # filter out text contours below this w/h ratio
TEXT_MAX_THICKNESS = 10  # max reduced px thickness of detected text contour

EDGE_MAX_OVERLAP = 1.0   # max reduced px horiz. overlap of contours in span
EDGE_MAX_LENGTH = 100.0  # max reduced px length of edge connecting contours
EDGE_ANGLE_COST = 10.0   # cost of angles in edges (tradeoff vs. length)
EDGE_MAX_ANGLE = 7.5     # maximum change in angle allowed between contours

RVEC_IDX = slice(0, 3)   # index of rvec in params vector
TVEC_IDX = slice(3, 6)   # index of tvec in params vector
SURF_IDX = slice(6, 11)  # index of cubic slopes in params vector

SPAN_MIN_WIDTH = 30      # minimum reduced px width for span
SPAN_PX_PER_STEP = 10    # reduced px spacing for sampling along spans
FOCAL_LENGTH = 1.2       # normalized focal length of camera

DEBUG_LEVEL = 0          # 0=none, 1=some, 2=lots, 3=all
DEBUG_OUTPUT = 'file'    # file, screen, both

WINDOW_NAME = 'Dewarp'   # Window name for visualization

# nice color palette for visualizing contours, etc.
CCOLORS = [
    (255, 0, 0),
    (255, 63, 0),
    (255, 127, 0),
    (255, 191, 0),
    (255, 255, 0),
    (191, 255, 0),
    (127, 255, 0),
    (63, 255, 0),
    (0, 255, 0),
    (0, 255, 63),
    (0, 255, 127),
    (0, 255, 191),
    (0, 255, 255),
    (0, 191, 255),
    (0, 127, 255),
    (0, 63, 255),
    (0, 0, 255),
    (63, 0, 255),
    (127, 0, 255),
    (191, 0, 255),
    (255, 0, 255),
    (255, 0, 191),
    (255, 0, 127),
    (255, 0, 63),
]

# default intrinsic parameter matrix
K = np.array([
    [FOCAL_LENGTH, 0, 0],
    [0, FOCAL_LENGTH, 0],
    [0, 0, 1]], dtype=np.float32)


params = None


def g(x, a):
    """ The polynomial function that describe the document surface

    :returns: Z value
    :rtype: scalar

    """

    X = np.array([np.power(x, i) for i in range(5)])
    return np.dot(a, X)


def g_prime(x, a):
    X = np.array([i * np.power(x, i - 1) for i in range(1, 5)])
    return np.dot(a[1:], X)


def compute_u_from_x(x, pvec):
    A = pvec[SURF_IDX]

    def func(t, a):
        return np.sqrt(1 + g_prime(t, a) ** 2)
    u = quad(func, 0, x, args=(A,))[0]
    return u


# Newton-Raphson method
def newton(func, x0, fprime=None, args=(), tol=1.48e-8, maxiter=100,
           fprime2=None):
    x = x0
    F_value = func(x)
    F_norm = np.linalg.norm(F_value, ord=2)  # l2 norm of vector
    iteration_counter = 0
    while abs(F_norm) > tol and iteration_counter < maxiter:
        delta = np.linalg.solve(fprime(x), -F_value)
        x = x + delta
        F_value = func(x)
        F_norm = np.linalg.norm(F_value, ord=2)
        iteration_counter += 1

    # Here, either a solution is found, or too many iterations
    if abs(F_norm) > tol:
        iteration_counter = -1
    return x, iteration_counter


def remap_xy(xy_coords, pvec):
    alpha, beta = xy_coords[0][0], xy_coords[0][1]

    A = pvec[SURF_IDX]
    R = pvec[RVEC_IDX].reshape((1, 3))
    rmat, _ = cv2.Rodrigues(R)
    P = np.array([alpha, beta, -FOCAL_LENGTH], dtype=np.float)
    O = np.array([0, 0, FOCAL_LENGTH], dtype=np.float)

    def intersect_func(x):
        """ Function to find the intersect between the document surface
        and the ray
        X*r1 + Y * r2 + g(X) * r3 + O - p * t = 0

        :param x: [Xik, Yik, tik]
        :returns: the value of the function
        :rtype: a vector of shape 3

        """
        XYZ = np.array([x[0], x[1], g(x[0], A)])
        ret = np.matmul(XYZ, rmat) + O - P * x[2]
        return ret

    def intersect_func_prime(x):
        """ function that compute derevative of the intersect_func
        dx = r1 + g_prime(x)
        dy = r2
        dt = - p

        :param x: [Xik, Yik, tik]
        :returns: the value of the derevative function

        """
        dx = rmat.T[:, 0] + g_prime(x[0], A) * rmat.T[:, 2]
        dy = rmat.T[:, 1]
        dt = -P
        dx = np.expand_dims(dx, axis=0)
        dy = np.expand_dims(dy, axis=0)
        dt = np.expand_dims(dt, axis=0)
        ret = np.concatenate((dx, dy, dt), axis=0).T
        return ret

    x0 = np.array([0, 0, remap_xy.t])
    root, iteration = newton(intersect_func, x0, fprime=intersect_func_prime)
    X = root[0]
    Y = root[1]
    '''
    print(alpha, beta)
    print(root)
    print('--------------------------------------------')
    '''
    remap_xy.t = root[2]
    v = Y
    u = compute_u_from_x(X, pvec)
    return u, v, remap_xy.t, X, Y


remap_xy.t = 0


def get_projected_xy(xy_coords, pvec):
    u, v, _, _, _ = remap_xy(xy_coords, pvec)
    return [-u, -v]


def get_l_from_pvec(line_idx, pvec):
    return pvec[11 + line_idx]


def straight_text_line_cost(pvec, line_points):
    ret = []
    for i, points in enumerate(line_points):
        l = get_l_from_pvec(i, pvec)
        for p in points:
            _, v, _, _, _ = remap_xy(p, pvec)
            ret.append(v - l)

    ret = np.array(ret)
    print('----------- cost', np.mean(np.abs(ret)))
    return ret


def dtdphi(i, rmat, drdphi, t, p, X, a):
    dr1dphi_i = drdphi[i, :3]
    dr3dphi_i = drdphi[i, 6:9]
    dr13dphi_i = drdphi[i, 2]
    dr33dphi_i = drdphi[i, 8]
    A = np.dot(dr1dphi_i, p * t) - dr13dphi_i * FOCAL_LENGTH
    B = np.dot(rmat[0, :], p)
    C = np.dot(dr3dphi_i, p * t) - dr33dphi_i * FOCAL_LENGTH
    D = np.dot(rmat[2, :], p)
    gpr_x = g_prime(X, a)

    ret = - (C - gpr_x * A) / (D - gpr_x * B)
    return ret


def dEdphi(pvec, point):
    R = pvec[RVEC_IDX].reshape((1, 3))
    A = pvec[SURF_IDX]
    rmat, drdphi = cv2.Rodrigues(R)
    ret = []
    for i in range(3):
        _, v, t, X, _ = remap_xy(point, pvec)
        alpha, beta = point[0][0], point[0][1]
        p = np.array([alpha, beta, -FOCAL_LENGTH], dtype=np.float)
        dr2 = drdphi[i, 3:6]

        r2 = rmat[1, :]
        dt = dtdphi(i, rmat, drdphi, t, p, X, A)

        dr23 = drdphi[i, 5]
        jac = np.dot(dr2, t * p) + np.dot(r2, p * dt) - dr23 * FOCAL_LENGTH

        ret.append(jac)
    return np.array(ret)


def dtda(rmat, p, X, a):
    ret = []
    for i in range(5):
        r3 = rmat[2, :]
        r1 = rmat[0, :]
        gpr_x = g_prime(X, a)
        jac = (X ** i) / np.dot((r3 - gpr_x * r1), p)
        ret.append(jac)

    return np.array(ret)


def dEda(pvec, point):
    R = pvec[RVEC_IDX].reshape((1, 3))
    A = pvec[SURF_IDX]
    rmat, _ = cv2.Rodrigues(R)

    _, v, t, X, _ = remap_xy(point, pvec)
    alpha, beta = point[0][0], point[0][1]
    p = np.array([alpha, beta, -FOCAL_LENGTH], dtype=np.float)

    r2 = rmat[1, :]
    dt = dtda(rmat, p, X, A)
    ret = np.dot(r2, p) * dt
    return ret


def dEdl(pvec, line_points):
    ret = []
    for j, points in enumerate(line_points):
        l = get_l_from_pvec(j, pvec)
        jac = 0
        for pt in points:
            _, v, _, _, _ = remap_xy(pt, pvec)
            jac += -2 * (v - l)
        ret.append(jac)
    return np.array(ret)


def straight_text_line_cost_prime(pvec, line_points):
    ret = []
    for j, points in enumerate(line_points):
        for pt in points:
            # compute dE/d_phi1
            dphi = dEdphi(pvec, pt)
            # print('drvec', dphi)
            # print('rvec', pvec[RVEC_IDX])

            # compute dE/dam
            da = dEda(pvec, pt) * 0.2
            # da = np.zeros(5)
            da[0] = 0.0
            # print('da', da)

            dtvec = np.zeros(3)

            # compute dE/Dl
            dl = np.zeros_like(pvec[11:])
            dl[j] = -1
            # print(dl)

            ret.append(np.hstack((dphi, dtvec, da, dl)))

    return np.array(ret)


def objective(pvec, line_points):
    ret = straight_text_line_cost(pvec, line_points)
    return ret


def objective_prime(pvec, line_points):
    return straight_text_line_cost_prime(pvec, line_points)


def optimize_params(pvec, line_points):
    '''
    pvec = minimize(objective,
                    pvec, method='Powell', args=(line_points,)).x
    '''
    pvec = least_squares(objective, pvec, args=(line_points,),
                         method='lm', jac=objective_prime, verbose=1).x

    return pvec


def debug_show(name, step, text, display):

    if DEBUG_OUTPUT != 'screen':
        filetext = text.replace(' ', '_')
        outfile = name + '_debug_' + str(step) + '_' + filetext + '.png'
        cv2.imwrite(outfile, display)

    if DEBUG_OUTPUT != 'file':

        image = display.copy()
        height = image.shape[0]

        cv2.putText(image, text, (16, height-16),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                    (0, 0, 0), 3, cv2.LINE_AA)

        cv2.putText(image, text, (16, height-16),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                    (255, 255, 255), 1, cv2.LINE_AA)

        cv2.imshow(WINDOW_NAME, image)

        while cv2.waitKey(5) < 0:
            pass


def round_nearest_multiple(i, factor):
    i = int(i)
    rem = i % factor
    if not rem:
        return i
    else:
        return i + factor - rem


def pix2norm(shape, pts):
    height, width = shape[:2]
    scl = 2.0/(max(height, width))
    offset = np.array([width, height], dtype=pts.dtype).reshape((-1, 1, 2))*0.5
    return (pts - offset) * scl


def norm2pix(shape, pts, as_integer):
    height, width = shape[:2]
    scl = max(height, width)*0.5
    offset = np.array([0.5*width, 0.5*height],
                      dtype=pts.dtype).reshape((-1, 1, 2))
    rval = pts * scl + offset
    if as_integer:
        return (rval + 0.5).astype(int)
    else:
        return rval


def fltp(point):
    return tuple(point.astype(int).flatten())


def draw_correspondences(img, dstpoints, projpts):

    display = img.copy()
    dstpoints = norm2pix(img.shape, dstpoints, True)
    projpts = norm2pix(img.shape, projpts, True)

    for pts, color in [(projpts, (255, 0, 0)),
                       (dstpoints, (0, 0, 255))]:

        for point in pts:
            cv2.circle(display, fltp(point), 3, color, -1, cv2.LINE_AA)

    for point_a, point_b in zip(projpts, dstpoints):
        cv2.line(display, fltp(point_a), fltp(point_b),
                 (255, 255, 255), 1, cv2.LINE_AA)

    return display


def estimate_l_from_y(span_points):
    ret = []
    for span in span_points:
        l = np.mean(span[:, :, 1])
        ret.append(-l)
    return np.array(ret)


def get_default_params(small, span_points):

    # page width and height
    page_width = small.shape[1]
    page_height = small.shape[0]
    rough_dims = (page_width, page_height)

    # our initial guess for the surface's coefficients
    surface_params = [0.0, 0.0, 0.0, 0.0, 0.0]

    # object points of flat page in 3D coordinates
    corners_object3d = np.array([
        [0, 0, 0],
        [page_width, 0, 0],
        [page_width, page_height, 0],
        [0, page_height, 0]], dtype=np.float32)
    corners = np.array([
        [0, 0],
        [page_width, 0],
        [page_width, page_height],
        [0, page_height]], dtype=np.float32)

    # estimate rotation and translation from four 2D-to-3D point
    # correspondences
    _, rvec, tvec = cv2.solvePnP(corners_object3d,
                                 corners, K, np.zeros(5))

    tvec = [0, 0, FOCAL_LENGTH]
    rvec = np.zeros(3)
    # initialize l for all the spans
    ls = estimate_l_from_y(span_points)
    params = np.hstack((np.array(rvec).flatten(),
                        np.array(tvec).flatten(),
                        np.array(surface_params).flatten(),
                        ls.flatten()))
    return params


def project_xy(xy_coords, pvec):

    # get cubic polynomial coefficients given
    #
    #  f(0) = 0, f'(0) = alpha
    #  f(1) = 0, f'(1) = beta

    alpha, beta = tuple(pvec[CUBIC_IDX])

    poly = np.array([
        alpha + beta,
        -2*alpha - beta,
        alpha,
        0])

    xy_coords = xy_coords.reshape((-1, 2))
    z_coords = np.polyval(poly, xy_coords[:, 0])

    objpoints = np.hstack((xy_coords, z_coords.reshape((-1, 1))))

    image_points, _ = cv2.projectPoints(objpoints,
                                        pvec[RVEC_IDX],
                                        pvec[TVEC_IDX],
                                        K, np.zeros(5))

    return image_points


def project_keypoints(pvec, keypoint_index):
    xy_coords = pvec[keypoint_index]
    xy_coords[0, :] = 0

    return project_xy(xy_coords, pvec)


def resize_to_screen(src, maxw=1280, maxh=700, copy=False):

    height, width = src.shape[:2]

    scl_x = float(width)/maxw
    scl_y = float(height)/maxh

    scl = int(np.ceil(max(scl_x, scl_y)))

    if scl > 1.0:
        inv_scl = 1.0/scl
        img = cv2.resize(src, (0, 0), None, inv_scl, inv_scl, cv2.INTER_AREA)
    elif copy:
        img = src.copy()
    else:
        img = src

    return img


def box(width, height):
    return np.ones((height, width), dtype=np.uint8)


def get_page_extents(small):

    height, width = small.shape[:2]

    xmin = PAGE_MARGIN_X
    ymin = PAGE_MARGIN_Y
    xmax = width-PAGE_MARGIN_X
    ymax = height-PAGE_MARGIN_Y

    page = np.zeros((height, width), dtype=np.uint8)
    cv2.rectangle(page, (xmin, ymin), (xmax, ymax), (255, 255, 255), -1)

    outline = np.array([
        [xmin, ymin],
        [xmin, ymax],
        [xmax, ymax],
        [xmax, ymin]])

    return page, outline


def get_mask(name, small):

    sgray = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY)

    mask = cv2.adaptiveThreshold(sgray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                 cv2.THRESH_BINARY_INV,
                                 ADAPTIVE_WINSZ,
                                 25)

    if DEBUG_LEVEL >= 3:
        debug_show(name, 0.1, 'thresholded', mask)

    mask = cv2.dilate(mask, box(9, 1))

    if DEBUG_LEVEL >= 3:
        debug_show(name, 0.2, 'dilated', mask)

    mask = cv2.erode(mask, box(1, 3))

    if DEBUG_LEVEL >= 3:
        debug_show(name, 0.3, 'eroded', mask)

    return mask


def interval_measure_overlap(int_a, int_b):
    return min(int_a[1], int_b[1]) - max(int_a[0], int_b[0])


def angle_dist(angle_b, angle_a):

    diff = angle_b - angle_a

    while diff > np.pi:
        diff -= 2*np.pi

    while diff < -np.pi:
        diff += 2*np.pi

    return np.abs(diff)


def blob_mean_and_tangent(contour):

    moments = cv2.moments(contour)

    area = moments['m00']

    mean_x = moments['m10'] / area
    mean_y = moments['m01'] / area

    moments_matrix = np.array([
        [moments['mu20'], moments['mu11']],
        [moments['mu11'], moments['mu02']]
    ]) / area

    _, svd_u, _ = cv2.SVDecomp(moments_matrix)

    center = np.array([mean_x, mean_y])
    tangent = svd_u[:, 0].flatten().copy()

    return center, tangent


class ContourInfo(object):

    def __init__(self, contour, rect, mask):

        self.contour = contour
        self.rect = rect
        self.mask = mask

        self.center, self.tangent = blob_mean_and_tangent(contour)

        self.angle = np.arctan2(self.tangent[1], self.tangent[0])

        clx = [self.proj_x(point) for point in contour]

        lxmin = min(clx)
        lxmax = max(clx)

        self.local_xrng = (lxmin, lxmax)

        self.point0 = self.center + self.tangent * lxmin
        self.point1 = self.center + self.tangent * lxmax

        self.pred = None
        self.succ = None

    def proj_x(self, point):
        return np.dot(self.tangent, point.flatten()-self.center)

    def local_overlap(self, other):
        xmin = self.proj_x(other.point0)
        xmax = self.proj_x(other.point1)
        return interval_measure_overlap(self.local_xrng, (xmin, xmax))


def generate_candidate_edge(cinfo_a, cinfo_b):

    # we want a left of b (so a's successor will be b and b's
    # predecessor will be a) make sure right endpoint of b is to the
    # right of left endpoint of a.
    if cinfo_a.point0[0] > cinfo_b.point1[0]:
        tmp = cinfo_a
        cinfo_a = cinfo_b
        cinfo_b = tmp

    x_overlap_a = cinfo_a.local_overlap(cinfo_b)
    x_overlap_b = cinfo_b.local_overlap(cinfo_a)

    overall_tangent = cinfo_b.center - cinfo_a.center
    overall_angle = np.arctan2(overall_tangent[1], overall_tangent[0])

    delta_angle = max(angle_dist(cinfo_a.angle, overall_angle),
                      angle_dist(cinfo_b.angle, overall_angle)) * 180/np.pi

    # we want the largest overlap in x to be small
    x_overlap = max(x_overlap_a, x_overlap_b)

    dist = np.linalg.norm(cinfo_b.point0 - cinfo_a.point1)

    if (dist > EDGE_MAX_LENGTH or
            x_overlap > EDGE_MAX_OVERLAP or
            delta_angle > EDGE_MAX_ANGLE):
        return None
    else:
        score = dist + delta_angle*EDGE_ANGLE_COST
        return (score, cinfo_a, cinfo_b)


def make_tight_mask(contour, xmin, ymin, width, height):

    tight_mask = np.zeros((height, width), dtype=np.uint8)
    tight_contour = contour - np.array((xmin, ymin)).reshape((-1, 1, 2))

    cv2.drawContours(tight_mask, [tight_contour], 0,
                     (1, 1, 1), -1)

    return tight_mask


def get_contours(name, small, line_img):
    #mask = get_mask(name, small)
    # cv2.imwrite('mask.jpg', mask)
    retval, mask = cv2.threshold(line_img, 12, 255, cv2.THRESH_BINARY)
    #retval, mask = cv2.threshold(mask, 12, 255, cv2.THRESH_BINARY)

    _, contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                      cv2.CHAIN_APPROX_NONE)

    contours_out = []

    for contour in contours:

        rect = cv2.boundingRect(contour)
        xmin, ymin, width, height = rect

        if (width < TEXT_MIN_WIDTH or
                height < TEXT_MIN_HEIGHT or
                width < TEXT_MIN_ASPECT*height):
            continue

        tight_mask = make_tight_mask(contour, xmin, ymin, width, height)

        if tight_mask.sum(axis=0).max() > TEXT_MAX_THICKNESS:
            continue

        contours_out.append(ContourInfo(contour, rect, tight_mask))

    visualize_contours(name, small, contours_out)

    return contours_out


def frange(start, stop, step=1.0):
    while start < stop:
        yield start
        start += step


def sample_spans(shape, spans):

    span_points = []

    for span in spans:
        contour_points = []

        for cinfo in span:
            yvals = np.arange(cinfo.mask.shape[0]).reshape((-1, 1))
            totals = (yvals * cinfo.mask).sum(axis=0)
            means = totals / cinfo.mask.sum(axis=0)

            xmin, ymin = cinfo.rect[:2]

            step = SPAN_PX_PER_STEP
            start = ((len(means)-1) % step) // 2

            for x in range(start, len(means), step):
                contour_points.append((x+xmin, means[x]+ymin))

        contour_points = np.array(contour_points,
                                  dtype=np.float32).reshape((-1, 1, 2))

        contour_points = pix2norm(shape, contour_points)

        span_points.append(contour_points)
    return span_points


def keypoints_from_samples(name, small, span_points):

    all_evecs = np.array([[0.0, 0.0]])
    all_weights = 0

    for points in span_points:
        print(points)
        _, evec = cv2.PCACompute(points.reshape((-1, 2)),
                                 None, maxComponents=1)
        weight = np.linalg.norm(points[-1] - points[0])

    evec = all_evecs / all_weights

    x_dir = evec.flatten()

    if x_dir[0] < 0:
        x_dir = -x_dir

    y_dir = np.array([-x_dir[1], x_dir[0]])

    pagecoords = cv2.convexHull(page_outline)
    pagecoords = pix2norm(pagemask.shape, pagecoords.reshape((-1, 1, 2)))
    pagecoords = pagecoords.reshape((-1, 2))

    px_coords = np.dot(pagecoords, x_dir)
    py_coords = np.dot(pagecoords, y_dir)

    px0 = px_coords.min()
    px1 = px_coords.max()

    py0 = py_coords.min()
    py1 = py_coords.max()

    p00 = px0 * x_dir + py0 * y_dir
    p10 = px1 * x_dir + py0 * y_dir
    p11 = px1 * x_dir + py1 * y_dir
    p01 = px0 * x_dir + py1 * y_dir

    corners = np.vstack((p00, p10, p11, p01)).reshape((-1, 1, 2))

    ycoords = []
    xcoords = []

    for points in span_points:
        pts = points.reshape((-1, 2))
        px_coords = np.dot(pts, x_dir)
        py_coords = np.dot(pts, y_dir)
        ycoords.append(py_coords.mean() - py0)
        xcoords.append(px_coords - px0)

    visualize_span_points(name, small, span_points, corners)

    return corners, np.array(ycoords), xcoords


def visualize_contours(name, small, cinfo_list):

    regions = np.zeros_like(small)

    for j, cinfo in enumerate(cinfo_list):

        cv2.drawContours(regions, [cinfo.contour], 0,
                         CCOLORS[j % len(CCOLORS)], -1)

    mask = (regions.max(axis=2) != 0)

    display = small.copy()
    display[mask] = (display[mask]/2) + (regions[mask]/2)

    for j, cinfo in enumerate(cinfo_list):
        color = CCOLORS[j % len(CCOLORS)]
        color = tuple([c/4 for c in color])

        cv2.circle(display, fltp(cinfo.center), 3,
                   (255, 255, 255), 1, cv2.LINE_AA)

        cv2.line(display, fltp(cinfo.point0), fltp(cinfo.point1),
                 (255, 255, 255), 1, cv2.LINE_AA)

    cv2.imwrite('contours.jpg', display)

    debug_show(name, 1, 'contours', display)


def visualize_spans(name, small, pagemask, spans):

    regions = np.zeros_like(small)

    for i, span in enumerate(spans):
        contours = [cinfo.contour for cinfo in span]
        cv2.drawContours(regions, contours, -1,
                         CCOLORS[i*3 % len(CCOLORS)], -1)

    mask = (regions.max(axis=2) != 0)

    display = small.copy()
    display[mask] = (display[mask]/2) + (regions[mask]/2)
    # display[pagemask == 0] /= 4

    cv2.imwrite('span.jpg', display)
    debug_show(name, 2, 'spans', display)


def visualize_span_points(name, small, span_points, corners):

    display = small.copy()

    for i, points in enumerate(span_points):

        points = norm2pix(small.shape, points, False)

        mean, small_evec = cv2.PCACompute(points.reshape((-1, 2)),
                                          None,
                                          maxComponents=1)

        dps = np.dot(points.reshape((-1, 2)), small_evec.reshape((2, 1)))
        dpm = np.dot(mean.flatten(), small_evec.flatten())

        point0 = mean + small_evec * (dps.min()-dpm)
        point1 = mean + small_evec * (dps.max()-dpm)

        for point in points:
            cv2.circle(display, fltp(point), 3,
                       CCOLORS[i % len(CCOLORS)], -1, cv2.LINE_AA)

        cv2.line(display, fltp(point0), fltp(point1),
                 (255, 255, 255), 1, cv2.LINE_AA)

    cv2.polylines(display, [norm2pix(small.shape, corners, True)],
                  True, (255, 255, 255))

    cv2.imwrite('span_points.jpg', display)
    debug_show(name, 3, 'span points', display)


def imgsize(img):
    height, width = img.shape[:2]
    return '{}x{}'.format(width, height)


def make_keypoint_index(span_counts):

    nspans = len(span_counts)
    npts = sum(span_counts)
    keypoint_index = np.zeros((npts+1, 2), dtype=int)
    start = 1

    for i, count in enumerate(span_counts):
        end = start + count
        keypoint_index[start:start+end, 1] = 8+i
        start = end

    keypoint_index[1:, 0] = np.arange(npts) + 8 + nspans

    return keypoint_index


def get_page_dims(corners, rough_dims, params):

    dst_br = corners[2].flatten()

    dims = np.array(rough_dims)

    def objective(dims):
        proj_br = project_xy(dims, params)
        return np.sum((dst_br - proj_br.flatten())**2)

    res = scipy.optimize.minimize(objective, dims, method='Powell')
    dims = res.x

    print('  got page dims', dims[0], 'x', dims[1])

    return dims


def remap_image(name, img, small, params):

    height = 1.0 * OUTPUT_ZOOM * img.shape[0]
    height = round_nearest_multiple(height, REMAP_DECIMATE)

    width = round_nearest_multiple(height * small.shape[1] / small.shape[0],
                                   REMAP_DECIMATE)

    print('  output will be {}x{}'.format(width, height))

    height_small = height / REMAP_DECIMATE
    width_small = width / REMAP_DECIMATE

    page_x_range = np.linspace(0, img.shape[1], width_small)
    page_y_range = np.linspace(0, img.shape[0], height_small)

    page_x_coords, page_y_coords = np.meshgrid(page_x_range, page_y_range)

    page_xy_coords = np.hstack((page_x_coords.flatten().reshape((-1, 1)),
                                page_y_coords.flatten().reshape((-1, 1))))

    page_xy_coords = page_xy_coords.astype(np.float32)

    page_xy_coords = pix2norm(img.shape, page_xy_coords)
    page_xy_coords = page_xy_coords.reshape((-1, 1, 2))
    image_points = []
    for i, p in enumerate(page_xy_coords):
        prj_xy = get_projected_xy(p, params)
        image_points.append(prj_xy)
    image_points = np.array(image_points, dtype=np.float32)
    image_points = image_points.reshape((-1, 1, 2))
    image_points = norm2pix(img.shape, image_points, False)

    image_x_coords = image_points[:, 0, 0].reshape(page_x_coords.shape)
    image_y_coords = image_points[:, 0, 1].reshape(page_y_coords.shape)

    image_x_coords = cv2.resize(image_x_coords, (width, height),
                                interpolation=cv2.INTER_LINEAR)

    image_y_coords = cv2.resize(image_y_coords, (width, height),
                                interpolation=cv2.INTER_LINEAR)

    remapped = cv2.remap(img, image_x_coords, image_y_coords,
                         cv2.INTER_CUBIC,
                         None, cv2.BORDER_REPLICATE)
    pil_image = Image.fromarray(remapped)
    pil_image.save(name + '_out.png', dpi=(OUTPUT_DPI, OUTPUT_DPI))


def main():

    if len(sys.argv) < 2:
        print('usage:', sys.argv[0], 'IMAGE1 [IMAGE2 ...]')
        sys.exit(0)

    if DEBUG_LEVEL > 0 and DEBUG_OUTPUT != 'file':
        cv2.namedWindow(WINDOW_NAME)

    outfiles = []
    line_img = cv2.imread(sys.argv[2], 0)
    imgfile = sys.argv[1]

    img = cv2.imread(imgfile)
    small = resize_to_screen(img)
    line_img = cv2.resize(line_img, (small.shape[1], small.shape[0]))
    basename = os.path.basename(imgfile)
    name, _ = os.path.splitext(basename)

    print('loaded', basename, 'with size', imgsize(img))
    print('and resized to', imgsize(small))

    if DEBUG_LEVEL >= 3:
        debug_show(name, 0.0, 'original', small)

    cinfo_list = get_contours(name, small, line_img)
    spans = [[c] for c in cinfo_list]

    span_points = sample_spans(small.shape, spans)

    print('  got', len(spans), 'spans',)
    print('with', sum([len(pts) for pts in span_points]), 'points.')
    '''
    corners, ycoords, xcoords = keypoints_from_samples(name, small,
                                                       pagemask,
                                                       page_outline,
                                                       span_points)
    '''

    params = get_default_params(small, span_points)

    params = optimize_params(params, span_points)
    print(params[:20])

    # page_dims = get_page_dims(corners, rough_dims, params)

    outfile = remap_image(name, img, small, params)


if __name__ == '__main__':
    main()
