from curses import wrapper
from dis import dis
from turtle import width
import numpy as np
import torch.nn as nn
import os
import argparse
from utils.torch_utils import select_device, smart_inference_mode
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams, LoadScreenshots
from models.common import DetectMultiBackend
from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_boxes, strip_optimizer, xyxy2xywh)
import cv2
import torch
from utils.plots import Annotator, colors, save_one_box
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from pathlib import Path
import sys

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative


class Cropper:
    # IMAGE_SIZE = (640, 640)

    def __init__(self, config_path, weights, iou_thres=0.45, conf_thres=0.25, device='', dnn=False, half=False, imgsz=(640, 640), line_thickness=3, hide_labels=False, hide_conf=False) -> None:
        self.iou_thres = iou_thres
        self.conf_thres = conf_thres
        self.config_path = config_path
        self.weights = weights
        self.device = select_device(device)
        self.net = DetectMultiBackend(
            self.weights, device=self.device, dnn=dnn, data=self.config_path, fp16=half)

        self.imgsz = check_img_size(imgsz, s=self.net.stride)
        self.line_thickness = line_thickness
        self.hide_labels = hide_labels
        self.hide_conf = hide_conf

    # preprocess image by resize check size
    def preprocess_img(self, img_path):
        img = cv2.imread(img_path)
        return img

    # run inference from single image
    def infer(self, source):

        # resize img
        # height, width, _ = img.shape
        # # img_resize = letterbox(im0, self.img_size, stride=self.stride, auto=self.auto)[0]
        # img_resize = cv2.resize(
        #     img, tuple(self.imgsz), interpolation=cv2.INTER_AREA)

        # img_resize = img_resize.transpose((2, 0, 1))[::-1]  # HWC to CWH, BGR to RGB
        # img_resize = np.ascontiguousarray(img_resize)
        save_dir = increment_path(Path(ROOT / 'warped'), exist_ok=True)
        save_dir.mkdir(parents=True, exist_ok=True)

        dataset = LoadImages(source, img_size=self.imgsz, stride=self.net.stride, auto=self.net.pt, vid_stride=1)

        seen, windows, dt = 0, [], (Profile(), Profile(), Profile())
        for path, im, im0s, vid_cap, s in dataset:

            list_corners = {}

            with dt[0]:
                im = torch.from_numpy(im).to(self.net.device)
                im = im.half() if self.net.fp16 else im.float()  # uint8 to fp16/32
                im /= 255  # 0 - 255 to 0.0 - 1.0
                if len(im.shape) == 3:
                    im = im[None]  # expand for batch dim

            basename = os.path.basename(path)

            # Inference
            with dt[1]:
                pred = self.net(im)

            # NMS
            with dt[2]:
                pred = non_max_suppression(pred, self.conf_thres, self.iou_thres)

            # Second-stage classifier (optional)
            # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

            # Process predictions
            for i, det in enumerate(pred):  # per image
                seen += 1
                p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

                if len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

                    for *xyxy, conf, cls in reversed(det):
                        c = int(cls)  # integer class
                        label = None if self.hide_labels else (
                            self.net.names[c] if self.hide_conf else f'{self.net.names[c]} {conf:.2f}')
                        box = xyxy
                        p1, p2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))

                        middle_point = self.get_middle_point(p1, p2)

                        list_corners[c] = middle_point

                        # im0 = cv2.circle(im0, middle_point, radius=2, color=(0, 0, 255), thickness=3)

            # list_corners = sorted(list_corners.items())

            if len(list_corners) == 3:
                list_corners = self.append_missing_point(list_corners)

            list_corners = sorted(list_corners.items())

            if len(list_corners) < 4:
                continue

            list_points = np.array([p[1] for p in list_corners], dtype='float32')

            warped = self.four_point_transform(im0, list_points)

            cv2.imwrite(str(save_dir / basename), warped)

        return warped

    def find_missing_idx(self, list_idxs):
        for i in range(0, 4):
            if i not in list_idxs:
                return i

    def append_missing_point(self, list_corners):
        list_idxs = [corner[0] for corner in list_corners.items()]
        # list_points = [corner[0] for corner in list_corners]
        missing_idx = self.find_missing_idx(list_idxs)

        #  top_left: 0, top_right 1, bottom_left 2, bottom_right 3
        # missed corner is top_left
        if missing_idx == 0:
            #  (top_right + bottom_left) / 2
            midpoint = np.add(list_corners[1], list_corners[2]) // 2
            x = 2 * midpoint[0] - list_corners[3][0]
            y = 2 * midpoint[1] - list_corners[3][1]
        # missed corner is top_right
        elif missing_idx == 1:
            # (top_left+ bottom_right) /2
            midpoint = np.add(list_corners[0], list_corners[3]) // 2
            x = 2 * midpoint[0] - list_corners[2][0]
            y = 2 * midpoint[1] - list_corners[2][1]

        # missed corner is bottom_left
        elif missing_idx == 2:
            #  (top_left + bottom_right)/ 2
            midpoint = np.add(list_corners[0], list_corners[3]) // 2
            x = 2 * midpoint[0] - list_corners[1][0]
            y = 2 * midpoint[1] - list_corners[1][1]

        # missed corner is bottom right
        else:
            #  (top_right + bottom_left) / 2
            midpoint = np.add(list_corners[1], list_corners[2]) // 2
            x = 2 * midpoint[0] - list_corners[0][0]
            y = 2 * midpoint[1] - list_corners[0][1]

        list_corners[missing_idx] = (x, y)

        return list_corners

    def four_point_transform(self, im, rect):
        (tl, tr, bl, br) = rect
        width_a = np.sqrt((bl[0] - br[0])**2 + (bl[1] - br[1])**2)
        width_b = np.sqrt((tl[0] - tr[0])**2 + (tl[1] - tr[1])**2)
        max_width = max(int(width_a), int(width_b))
        height_a = np.sqrt((tl[0] - bl[0])**2 + (tl[1] - bl[1])**2)
        height_b = np.sqrt((tr[0] - br[0])**2 + (tr[1] - br[1])**2)
        max_height = max(int(height_a), int(height_b))
        dst = np.array([
            [0, 0],
            [max_width - 1, 0],
            [0, max_height - 1],
            [max_width - 1, max_height - 1]], dtype='float32'
        )
        M = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(im, M, (max_width, max_height), flags=cv2.INTER_NEAREST)
        return warped

    # get middle point of box

    def get_middle_point(self, p1, p2):
        x_middle, y_middle = (p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2
        return (x_middle, y_middle)

    # convert boxes to point
    def convert_bbox_to_points(self):
        pass

    def _check_points(self):
        pass

    #  crop screen and image alignment
    def crop_screen(self, img_path):
        self.infer(img_path)


if __name__ == '__main__':
    croper = Cropper(config_path='./data/4corners.yaml',
                     weights='./weights/best.pt')

    img_path = '/home/vietlq4/PaddleOCR/datasets/customdata/images/test/**/*.jpg'

    croper.crop_screen(img_path)
