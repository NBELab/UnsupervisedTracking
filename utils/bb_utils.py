import numpy as np
import cv2
import torch
import kornia
from os.path import join as fullfile


def cxy_wh_2_rect1(pos, sz):
    return np.array(
        [pos[0] - sz[0] / 2 + 1, pos[1] - sz[1] / 2 + 1, sz[0], sz[1]]
    )  # 1-index


def cxy_wh_2_rect1_torch(pos, sz):
    return torch.tensor(
        [pos[0] - sz[0] / 2 + 1, pos[1] - sz[1] / 2 + 1, sz[0], sz[1]]
    )  # 1-index


def bb_2_cxy_wh(bb):
    return np.array([bb[0], bb[1]]), np.array([bb[2] - bb[0], bb[3] - bb[1]])


def bbox_2_cxy_wh(bb):
    return np.array(
        [bb[0] + (bb[2] - bb[0]) / 2, bb[1] + (bb[3] - bb[1]) / 2]
    ), np.array([bb[2] - bb[0], bb[3] - bb[1]])


def rect1_2_cxy_wh(rect):
    return np.array([rect[0] + rect[2] / 2 - 1, rect[1] + rect[3] / 2 - 1]), np.array(
        [rect[2], rect[3]]
    )  # 0-index


def rect1_2_cxy_wh_test(rect):
    return np.array([rect[0] + rect[2], rect[1] + rect[3]]), np.array(
        [rect[2], rect[3]]
    )  # 0-index


def rect1_2_cxy_wh_torch(rect, device):
    return torch.tensor([rect[0] + rect[2] / 2 - 1, rect[1] + rect[3] / 2 - 1]).to(
        device
    ), torch.tensor([rect[2], rect[3]]).to(
        device
    )  # 0-index


def cxy_wh_2_bb(cxy, wh):
    return np.array([cxy[0], cxy[1], cxy[0] + wh[0], cxy[1] + wh[1]])  # 0-index


def cxy_wh_2_bbox(cxy, wh):
    return np.array(
        [cxy[0] - wh[0] / 2, cxy[1] - wh[1] / 2, cxy[0] + wh[0] / 2, cxy[1] + wh[1] / 2]
    )  # 0-index


def cxy_wh_2_bbox_torch(cxy, wh):
    return torch.tensor(
        [cxy[0] - wh[0] / 2, cxy[1] - wh[1] / 2, cxy[0] + wh[0] / 2, cxy[1] + wh[1] / 2]
    )  # 0-index


def gaussian_shaped_labels(sigma, sz):
    x, y = np.meshgrid(
        np.arange(1, sz[0] + 1) - np.floor(float(sz[0]) / 2),
        np.arange(1, sz[1] + 1) - np.floor(float(sz[1]) / 2),
    )
    d = x**2 + y**2
    g = np.exp(-0.5 / (sigma**2) * d)
    g = np.roll(g, int(-np.floor(float(sz[0]) / 2.0) + 1), axis=0)
    g = np.roll(g, int(-np.floor(float(sz[1]) / 2.0) + 1), axis=1)
    return g


def crop_chw(image, bbox, out_sz, padding=(0, 0, 0)):
    a = (out_sz - 1) / (bbox[2] - bbox[0])
    b = (out_sz - 1) / (bbox[3] - bbox[1])
    c = -a * bbox[0]
    d = -b * bbox[1]
    mapping = np.array([[a, 0, c], [0, b, d]]).astype(np.float)
    crop = cv2.warpAffine(
        image,
        mapping,
        (out_sz, out_sz),
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=padding,
    )
    return np.transpose(crop, (2, 0, 1))


def crop_chw_torch(image, bbox, out_sz, device, padding=(0, 0, 0)):
    a = (out_sz - 1) / (bbox[2] - bbox[0])
    b = (out_sz - 1) / (bbox[3] - bbox[1])
    c = -a * bbox[1]
    d = -b * bbox[0]
    # print(a,b,c,d)
    mapping = (
        torch.tensor([[a, 0, c], [0, b, d]]).type(torch.float32).unsqueeze(0).to(device)
    )
    image = image.unsqueeze(0)
    x = kornia.geometry.transform.warp_affine(
        image, mapping, (out_sz, out_sz), padding_mode="border", fill_value=padding
    )
    return x.squeeze()


def crop_chw_gray(image, bbox, out_sz, padding=(0, 0, 0)):
    a = (out_sz - 1) / (bbox[2] - bbox[0])
    b = (out_sz - 1) / (bbox[3] - bbox[1])
    c = -a * bbox[0]
    d = -b * bbox[1]
    mapping = np.array([[a, 0, c], [0, b, d]]).astype(np.float)
    crop = cv2.warpAffine(
        image,
        mapping,
        (out_sz, out_sz),
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=padding,
    )
    return crop


def crop_chw_mult_dim(image, bbox, out_sz, padding=(0, 0, 0)):
    a = (out_sz - 1) / (bbox[2] - bbox[0])
    b = (out_sz - 1) / (bbox[3] - bbox[1])
    c = -a * bbox[0]
    d = -b * bbox[1]
    mapping = np.array([[a, 0, c], [0, b, d]]).astype(np.float)

    crop = cv2.warpAffine(
        image,
        mapping,
        (out_sz, out_sz),
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=padding,
    )
    return np.transpose(crop, (2, 0, 1))


def overlap_ratio(rect1, rect2):
    """
    Compute overlap ratio between two rects
    - rect: 1d array of [x,y,w,h] or
            2d array of N x [x,y,w,h]
    """

    if rect1.ndim == 1:
        rect1 = rect1[None, :]
    if rect2.ndim == 1:
        rect2 = rect2[None, :]

    left = np.maximum(rect1[:, 0], rect2[:, 0])
    right = np.minimum(rect1[:, 0] + rect1[:, 2], rect2[:, 0] + rect2[:, 2])
    top = np.maximum(rect1[:, 1], rect2[:, 1])
    bottom = np.minimum(rect1[:, 1] + rect1[:, 3], rect2[:, 1] + rect2[:, 3])

    intersect = np.maximum(0, right - left) * np.maximum(0, bottom - top)
    union = rect1[:, 2] * rect1[:, 3] + rect2[:, 2] * rect2[:, 3] - intersect
    iou = np.clip(intersect / union, 0, 1)
    return iou


def compute_success_overlap(gt_bb, result_bb):
    thresholds_overlap = np.arange(0, 1.05, 0.05)
    n_frame = len(gt_bb)
    success = np.zeros(len(thresholds_overlap))
    iou = overlap_ratio(gt_bb, result_bb)
    for i in range(len(thresholds_overlap)):
        success[i] = sum(iou > thresholds_overlap[i]) / float(n_frame)
    return success


def compute_success_error(gt_center, result_center):
    thresholds_error = np.arange(0, 51, 1)
    n_frame = len(gt_center)
    success = np.zeros(len(thresholds_error))
    dist = np.sqrt(np.sum(np.power(gt_center - result_center, 2), axis=1))
    for i in range(len(thresholds_error)):
        success[i] = sum(dist <= thresholds_error[i]) / float(n_frame)
    return success


def get_result_bb(arch, seq):
    result_path = fullfile(arch, seq + ".txt")
    temp = np.loadtxt(result_path, delimiter=",").astype(np.float64)
    return np.array(temp)


def convert_bb_to_center(bboxes):
    return np.array(
        [
            (bboxes[:, 0] + (bboxes[:, 2] - 1) / 2),
            (bboxes[:, 1] + (bboxes[:, 3] - 1) / 2),
        ]
    ).T


def compute_center_dist(gt_center, result_center):
    return np.sqrt(np.sum(np.power(gt_center - result_center, 2), axis=1))


if __name__ == "__main__":
    a = gaussian_shaped_labels(10, [5, 5])
    print(a)
