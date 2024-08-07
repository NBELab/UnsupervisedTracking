import torch
import numpy as np
from tqdm import *
from os.path import join, isdir
from os import makedirs
import time as time
import glob
from os.path import join as fullfile
import argparse
import os
import sys

module_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "utils"))
if module_path not in sys.path:
    sys.path.append(module_path)
from dataset_utils import *
from events_contrast_maximization.utils.event_utils import (
    events_to_voxel_timesync_torch,
)
from events_contrast_maximization.tools.read_events import *
from bb_utils import *
from net_utils import *
from net import *

dataset = "indoor-real-event"


class TrackerConfig(object):
    feature_path = "param.pth"
    crop_sz = 128

    lambda0 = 1e-4
    padding = 2
    output_sigma_factor = 0.1
    interp_factor = 0.01
    num_scale = 3
    scale_step = 1.0275
    scale_factor = scale_step ** (np.arange(num_scale) - num_scale / 2)
    min_scale_factor = 0.2
    max_scale_factor = 5
    scale_penalty = 0.995
    scale_penalties = scale_penalty ** (np.abs((np.arange(num_scale) - num_scale / 2)))
    net_average_image = np.array([104, 117, 123]).reshape(-1, 1, 1).astype(np.float32)
    net_input_size = [crop_sz, crop_sz]
    output_sigma = crop_sz / (1 + padding) * output_sigma_factor
    y = gaussian_shaped_labels(output_sigma, net_input_size)
    yf = torch.fft.rfft2(torch.Tensor(y).view(1, 1, crop_sz, crop_sz).cuda())
    yf = torch.view_as_real(yf)
    cos_window = torch.Tensor(np.outer(np.hanning(crop_sz), np.hanning(crop_sz))).cuda()


config = TrackerConfig()


def test_events(net, files, data, bb_gt, num_bins, test_name):
    for video_id, video in tqdm(enumerate(files)):
        path = files[video]["aedat4"]
        size, events, frame_exposure_time, frame_interval_time, frames = get_items(path)
        frame_timestamp = frame_exposure_time
        n_images = len(frame_timestamp)
        xs = torch.tensor(events["x"])
        ys = torch.tensor(events["y"])
        ps = torch.tensor(events["polarity"])
        ts = torch.tensor(events["timestamp"], dtype=torch.int64)
        for k, indxs in enumerate(data[files[video]["name"]]["frames"]):
            try:
                bb = bb_gt[files[video]["name"] + f"__{k}"]
            except:
                continue
            # print(len(bb))
            bb_index = 0
            start_i = indxs[0]
            im = events_to_voxel_timesync_torch(
                xs,
                ys,
                ts,
                ps,
                num_bins,
                t0=frame_timestamp[start_i][0],
                t1=frame_timestamp[start_i][1],
                sensor_size=size,
                device="cpu",
            )
            im = im.permute(1, 2, 0)
            im2 = im.cpu().detach().numpy()

            target_pos, target_sz = bbox_2_cxy_wh(bb[bb_index])

            # confine results
            min_sz = np.maximum(config.min_scale_factor * target_sz, 4)
            max_sz = np.minimum(im.shape[:2], config.max_scale_factor * target_sz)

            # crop template
            window_sz = target_sz * (1 + config.padding)
            bbox = cxy_wh_2_bbox(target_pos, window_sz)

            patch = crop_chw_mult_dim(im2, bbox, config.crop_sz)
            target = patch

            net.update(torch.Tensor(np.expand_dims(target, axis=0)).cuda(), events=True)
            res = [cxy_wh_2_rect1(target_pos, target_sz)]  # save in .txt
            patch_crop = np.zeros(
                (config.num_scale, patch.shape[0], patch.shape[1], patch.shape[2]),
                np.float32,
            )
            # patch_crop_f = np.zeros((config.num_scale, target_f.shape[0], target_f.shape[1], target_f.shape[2]), np.float32)
            for i in range(start_i + 1, indxs[1]):  # track
                try:
                    im = events_to_voxel_timesync_torch(
                        xs,
                        ys,
                        ts,
                        ps,
                        num_bins,
                        t0=frame_timestamp[i][0],
                        t1=frame_timestamp[i][1],
                        sensor_size=size,
                        device="cpu",
                    )
                except Exception as e:
                    break
                im = im.permute(1, 2, 0)
                im2 = im.cpu().detach().numpy().astype("float32")
                for j in range(config.num_scale):  # crop multi-scale search region
                    window_sz = target_sz * (
                        config.scale_factor[j] * (1 + config.padding)
                    )
                    bbox = cxy_wh_2_bbox(target_pos, window_sz)
                    patch_crop[j, :] = crop_chw_mult_dim(im2, bbox, config.crop_sz)
                search = patch_crop  # - config.net_average_image

                response = net(torch.Tensor(search).cuda(), events=True)
                peak, idx = torch.max(response.view(config.num_scale, -1), 1)
                peak = peak.data.cpu().numpy() * config.scale_penalties
                best_scale = np.argmax(peak)
                r_max, c_max = np.unravel_index(
                    idx[best_scale].cpu().detach(), config.net_input_size
                )

                if r_max > config.net_input_size[0] / 2:
                    r_max = r_max - config.net_input_size[0]
                if c_max > config.net_input_size[1] / 2:
                    c_max = c_max - config.net_input_size[1]
                window_sz = target_sz * (
                    config.scale_factor[best_scale] * (1 + config.padding)
                )

                target_pos = (
                    target_pos
                    + np.array([c_max, r_max]) * window_sz / config.net_input_size
                )
                target_sz = np.minimum(
                    np.maximum(window_sz / (1 + config.padding), min_sz), max_sz
                )

                # model update
                window_sz = target_sz * (1 + config.padding)
                bbox = cxy_wh_2_bbox(target_pos, window_sz)

                patch = crop_chw_mult_dim(im2, bbox, config.crop_sz)
                target = patch

                net.update(
                    torch.Tensor(np.expand_dims(target, axis=0)).cuda(),
                    events=True,
                    lr=config.interp_factor,
                )
                res.append(cxy_wh_2_rect1(target_pos, target_sz))  # 1-index

            # save result
            test_path = join("track", "result", dataset, "events", test_name)
            if not isdir(test_path):
                makedirs(test_path)
            result_path = join(test_path, files[video]["name"] + f"__{k}.txt")
            with open(result_path, "w") as f:
                for x in res:
                    f.write(",".join(["{:.2f}".format(i) for i in x]) + "\n")


def test_add_events_frames(net, files, data, bb_gt, num_bins, test_name):
    for video_id, video in tqdm(enumerate(files)):  # run without resetting
        path = files[video]["aedat4"]
        size, events, frame_exposure_time, frame_interval_time, frames = get_items(path)
        frame_timestamp = frame_exposure_time
        n_images = len(frame_timestamp)
        xs = torch.tensor(events["x"])
        ys = torch.tensor(events["y"])
        ps = torch.tensor(events["polarity"])
        ts = torch.tensor(events["timestamp"], dtype=torch.int64)
        for k, indxs in enumerate(data[files[video]["name"]]["frames"]):
            try:
                bb = bb_gt[files[video]["name"] + f"__{k}"]
            except:
                continue
            # print(len(bb))
            start_i = indxs[0]
            im = events_to_voxel_timesync_torch(
                xs,
                ys,
                ts,
                ps,
                num_bins,
                t0=frame_timestamp[start_i][0],
                t1=frame_timestamp[start_i][1],
                sensor_size=size,
                device="cpu",
            )
            im = im.permute(1, 2, 0)
            im2 = im.cpu().detach().numpy()
            frame = frames[start_i].squeeze()
            target_pos, target_sz = bbox_2_cxy_wh(bb[0])  # OTB label is 1-indexed

            # confine results
            min_sz = np.maximum(config.min_scale_factor * target_sz, 4)
            max_sz = np.minimum(im.shape[:2], config.max_scale_factor * target_sz)

            # crop template
            window_sz = target_sz * (1 + config.padding)
            bbox = cxy_wh_2_bbox(target_pos, window_sz)

            patch = crop_chw_mult_dim(im2, bbox, config.crop_sz)
            target = patch

            patch_f = crop_chw_gray(frame, bbox, config.crop_sz)
            target_f = patch_f[np.newaxis, ...]  # - config.net_average_image

            net.update(
                torch.Tensor(np.expand_dims(target_f, axis=0)).cuda(),
                torch.Tensor(np.expand_dims(target, axis=0)).cuda(),
            )
            res = [cxy_wh_2_rect1(target_pos, target_sz)]  # save in .txt
            patch_crop = np.zeros(
                (config.num_scale, patch.shape[0], patch.shape[1], patch.shape[2]),
                np.float32,
            )
            patch_crop_f = np.zeros(
                (
                    config.num_scale,
                    target_f.shape[0],
                    target_f.shape[1],
                    target_f.shape[2],
                ),
                np.float32,
            )
            for i in range(start_i + 1, indxs[1]):  # track
                try:
                    im = events_to_voxel_timesync_torch(
                        xs,
                        ys,
                        ts,
                        ps,
                        num_bins,
                        t0=frame_timestamp[i][0],
                        t1=frame_timestamp[i][1],
                        sensor_size=size,
                        device="cpu",
                    )
                except:
                    break
                im = im.permute(1, 2, 0)
                im2 = im.cpu().detach().numpy()
                for j in range(config.num_scale):  # crop multi-scale search region
                    window_sz = target_sz * (
                        config.scale_factor[j] * (1 + config.padding)
                    )
                    bbox = cxy_wh_2_bbox(target_pos, window_sz)
                    patch_crop[j, :] = crop_chw_mult_dim(im2, bbox, config.crop_sz)
                search = patch_crop  # - config.net_average_image
                # print(search.shape)
                try:
                    frame = frames[i].squeeze()
                except:
                    break
                for j in range(config.num_scale):  # crop multi-scale search region
                    window_sz = target_sz * (
                        config.scale_factor[j] * (1 + config.padding)
                    )
                    bbox = cxy_wh_2_bbox(target_pos, window_sz)
                    patch_crop_f[j, :] = crop_chw_gray(frame, bbox, config.crop_sz)
                search_f = patch_crop_f

                response = net(
                    torch.Tensor(search_f).cuda(), torch.Tensor(search).cuda()
                )
                peak, idx = torch.max(response.view(config.num_scale, -1), 1)
                peak = peak.data.cpu().numpy() * config.scale_penalties
                best_scale = np.argmax(peak)
                r_max, c_max = np.unravel_index(
                    idx[best_scale].cpu().detach(), config.net_input_size
                )

                if r_max > config.net_input_size[0] / 2:
                    r_max = r_max - config.net_input_size[0]
                if c_max > config.net_input_size[1] / 2:
                    c_max = c_max - config.net_input_size[1]
                window_sz = target_sz * (
                    config.scale_factor[best_scale] * (1 + config.padding)
                )

                target_pos = (
                    target_pos
                    + np.array([c_max, r_max]) * window_sz / config.net_input_size
                )
                target_sz = np.minimum(
                    np.maximum(window_sz / (1 + config.padding), min_sz), max_sz
                )

                # model update
                window_sz = target_sz * (1 + config.padding)
                bbox = cxy_wh_2_bbox(target_pos, window_sz)

                patch = crop_chw_mult_dim(im2, bbox, config.crop_sz)
                target = patch  # - config.net_average_image

                patch_f = crop_chw_gray(frame, bbox, config.crop_sz)
                patch_f = patch_f[np.newaxis, ...]
                target_f = patch_f

                net.update(
                    torch.Tensor(np.expand_dims(target_f, axis=0)).cuda(),
                    torch.Tensor(np.expand_dims(target, axis=0)).cuda(),
                    lr=config.interp_factor,
                )

                res.append(cxy_wh_2_rect1(target_pos, target_sz))  # 1-index

            # save result
            test_path = join("track", "result", dataset, "event_frames", test_name)
            if not isdir(test_path):
                makedirs(test_path)
            result_path = join(test_path, files[video]["name"] + f"__{k}.txt")
            with open(result_path, "w") as f:
                for x in res:
                    f.write(",".join(["{:.2f}".format(i) for i in x]) + "\n")


def test_frames(net, files, data, bb_gt, test_name):
    for video_id, video in tqdm(enumerate(files)):  # run without resetting
        path = files[video]["aedat4"]
        size, events, frame_exposure_time, frame_interval_time, frames = get_items(path)
        frame_timestamp = frame_exposure_time
        n_images = len(frame_timestamp)
        for k, indxs in enumerate(data[files[video]["name"]]["frames"]):
            try:
                bb = bb_gt[files[video]["name"] + f"__{k}"]
            except:
                continue
            start_i = indxs[0]
            im = frames[start_i].squeeze()
            target_pos, target_sz = bbox_2_cxy_wh(bb[0])

            # confine results
            min_sz = np.maximum(config.min_scale_factor * target_sz, 4)
            max_sz = np.minimum(im.shape[:], config.max_scale_factor * target_sz)

            # crop template
            window_sz = target_sz * (1 + config.padding)
            bbox = cxy_wh_2_bbox(target_pos, window_sz)

            patch = crop_chw_gray(im, bbox, config.crop_sz)
            target = patch[np.newaxis, ...]
            net.update(
                torch.Tensor(np.expand_dims(target, axis=0)).cuda(), events=False
            )

            res = [cxy_wh_2_rect1(target_pos, target_sz)]  # save in .txt
            patch_crop = np.zeros(
                (config.num_scale, target.shape[0], target.shape[1], target.shape[2]),
                np.float32,
            )
            for i in range(start_i + 1, indxs[1]):  # track
                try:
                    im = frames[i].squeeze()
                except:
                    break
                # im = cv2.merge([gray,gray,gray])
                # im2 = im2.transpose((1,2,0))
                for j in range(config.num_scale):  # crop multi-scale search region
                    window_sz = target_sz * (
                        config.scale_factor[j] * (1 + config.padding)
                    )
                    bbox = cxy_wh_2_bbox(target_pos, window_sz)
                    patch_crop[j, :] = crop_chw_gray(im, bbox, config.crop_sz)

                search = patch_crop  #  - config.net_average_image
                # print(search.shape)
                response = net(torch.Tensor(search).cuda(), events=False)
                peak, idx = torch.max(response.view(config.num_scale, -1), 1)
                peak = peak.data.cpu().numpy() * config.scale_penalties
                best_scale = np.argmax(peak)
                r_max, c_max = np.unravel_index(
                    idx[best_scale].cpu().detach(), config.net_input_size
                )

                if r_max > config.net_input_size[0] / 2:
                    r_max = r_max - config.net_input_size[0]
                if c_max > config.net_input_size[1] / 2:
                    c_max = c_max - config.net_input_size[1]
                window_sz = target_sz * (
                    config.scale_factor[best_scale] * (1 + config.padding)
                )

                target_pos = (
                    target_pos
                    + np.array([c_max, r_max]) * window_sz / config.net_input_size
                )
                target_sz = np.minimum(
                    np.maximum(window_sz / (1 + config.padding), min_sz), max_sz
                )

                # model update
                window_sz = target_sz * (1 + config.padding)
                bbox = cxy_wh_2_bbox(target_pos, window_sz)

                patch = crop_chw_gray(im, bbox, config.crop_sz)
                target = patch[np.newaxis, ...]

                net.update(
                    torch.Tensor(np.expand_dims(target, axis=0)).cuda(),
                    events=False,
                    lr=config.interp_factor,
                )

                res.append(cxy_wh_2_rect1(target_pos, target_sz))  # 1-index

            # save result
            test_path = join("track", "result", dataset, "frames", test_name)
            if not isdir(test_path):
                makedirs(test_path)
            result_path = join(test_path, files[video]["name"] + f"__{k}.txt")
            with open(result_path, "w") as f:
                for x in res:
                    f.write(",".join(["{:.2f}".format(i) for i in x]) + "\n")


def eval_auc(files, data, bb_gt, tracker_reg="S*", start=0, end=1e6, ev=0):
    if ev == 0:
        trackers = glob.glob(
            fullfile("track", "result", dataset, "events", tracker_reg)
        )
    elif ev == 1:
        trackers = glob.glob(
            fullfile("track", "result", dataset, "frames", tracker_reg)
        )
    else:
        trackers = glob.glob(
            fullfile("track", "result", dataset, "event_frames", tracker_reg)
        )
    trackers = trackers[start : min(end, len(trackers))]

    seqs_arr = []
    for video in files:
        for k, indxs in enumerate(data[files[video]["name"]]["frames"]):
            seqs_arr.append(files[video]["name"] + f"__{k}")
    n_seq = len(seqs_arr)
    thresholds_overlap = np.arange(0, 1.05, 0.05)
    thresholds_error = np.arange(0, 51, 1)

    success_overlap = np.zeros((n_seq, len(trackers), len(thresholds_overlap)))
    success_error = np.zeros((n_seq, len(trackers), len(thresholds_error)))

    success_rate = np.zeros((n_seq, len(trackers)))
    for i in range(n_seq):
        seq = seqs_arr[i]
        try:
            bb_seq = bb_gt[seq]
        except:
            continue
        gt_rect = np.array([cxy_wh_2_rect1(*bbox_2_cxy_wh(x)) for x in bb_seq]).astype(
            np.float32
        )[:]
        gt_center = convert_bb_to_center(gt_rect)
        for j in range(len(trackers)):
            tracker = trackers[j]
            try:
                bb = get_result_bb(tracker, seq)
                m = min(len(bb), len(gt_rect))
                bb = bb[:m]
                gt_rect = gt_rect[:m]
                gt_center = gt_center[:m]
                center = convert_bb_to_center(bb)
                success_overlap[i][j] = compute_success_overlap(gt_rect, bb)
                success_error[i][j] = compute_success_error(gt_center, center)
                overlap = overlap_ratio(gt_rect, bb)
                success_rate[i][j] = len(overlap[overlap >= 0.5]) / len(overlap)
            except Exception as e:
                print(e)
                continue

    ids = []
    for i in range(n_seq):
        if seqs_arr[i] in seqs_arr:
            ids.append(i)
    for i in range(len(trackers)):
        auc = round(success_overlap[ids, i, :].mean(), 4)
        rate = round(success_rate[ids, i].mean(), 4)
        precision = round(success_error[ids, i, :].mean(), 4)
        print("auc       %s(%.4f)" % (trackers[i], auc))
        print("rate      %s(%.4f)" % (trackers[i], rate))
        print("precision %s(%.4f)" % (trackers[i], precision))


def parse_args():
    argparser = argparse.ArgumentParser(description="Run tracker on new dataset")
    argparser.add_argument("--name", required=True, help="Test name")
    argparser.add_argument(
        "--weights",
        default="./models/model.pth.tar",
        help="Path to model weights (default: ./models/model.pth.tar)",
    )
    argparser.add_argument(
        "--data-path", default="./data/", help="Path to test data (default: ./data/)"
    )
    argparser.add_argument(
        "--annotation-path",
        default="./data/annotations/",
        help="Path to annotation data (default: ./data/annotations/)",
    )
    argparser.add_argument(
        "--data-info-path",
        default="./data/",
        help="Path to data information (default: ./data)",
    )
    argparser.add_argument(
        "--num-bins",
        "-b",
        default=5,
        type=int,
        help="Number of temporal bins, must be the same as the network (default: 5)",
    )
    argparser.add_argument(
        "--track-events",
        "-e",
        default=True,
        action="store_true",
        help="Track on events (default: True)",
    )
    argparser.add_argument(
        "--no-track-events", dest="track_events", action="store_false"
    )
    argparser.add_argument(
        "--track-frames",
        "-f",
        default=True,
        action="store_true",
        help="Track on frames (default: True)",
    )
    argparser.add_argument(
        "--no-track-frames", dest="track_frames", action="store_false"
    )
    argparser.add_argument(
        "--track-combined",
        "-c",
        default=True,
        action="store_true",
        help="Track on events and frames combined (default: True)",
    )
    argparser.add_argument(
        "--no-track-combined", dest="track_combined", action="store_false"
    )
    argparser.add_argument(
        "--eval-only",
        default=False,
        action="store_true",
        help="Evaluate on previous results (default: False)",
    )
    argparser.add_argument(
        "-p",
        default=None,
        help="p (as in the paper), for reduction of frame net output (default:None)",
    )

    return argparser.parse_args()


def main(args):
    files = get_files(args.data_path)
    gt_bb = get_annotations(args.annotation_path)
    data = get_dataset_info(args.data_info_path)
    if args.track_events or args.track_frames:
        net = DCFNet(args.num_bins, config)
        net.load_param(args.weights)
        net.eval().cuda()
        net.config.scale_step = 1.010
        net.config.scale_penalty = 0.9800
        net.config.interp_factor = 0.014
        if args.track_events:
            if not args.eval_only:
                test_events(net, files, data, gt_bb, args.num_bins, args.name)
            eval_auc(files, data, gt_bb, args.name, ev=0)
        if args.track_frames:
            if not args.eval_only:
                test_frames(net, files, data, gt_bb, args.name)
            eval_auc(files, data, gt_bb, args.name, ev=1)
    if args.track_combined:
        net = DCFNet_add(args.num_bins, config, float(args.p))
        net.load_param(args.weights)
        net.eval().cuda()
        net.config.scale_step = 1.010
        net.config.scale_penalty = 0.9800
        net.config.interp_factor = 0.014
        if not args.eval_only:
            test_add_events_frames(net, files, data, gt_bb, args.num_bins, args.name)
        eval_auc(files, data, gt_bb, args.name, ev=2)


if __name__ == "__main__":
    main(parse_args())
