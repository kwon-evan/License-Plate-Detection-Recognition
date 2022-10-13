import argparse
import os.path
import sys
import threading
import time
from collections import deque
from multiprocessing.context import Process
from pathlib import Path
from threading import Thread, Lock

import cv2
import torch
import torch.backends.cudnn as cudnn
from PIL import ImageDraw, ImageFont
from PIL.Image import Image
from enlighten import manager
from pytorch_lightning import Trainer
from numpy import random
import numpy as np
import enlighten
import warnings

warnings.filterwarnings("ignore")

sys.path.append('./yolov7')
from yolov7.models.experimental import attempt_load
from yolov7.utils.datasets import LoadStreams, LoadImages, letterbox
from yolov7.utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, \
    apply_classifier, scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from yolov7.utils.plots import plot_one_box, plot_one_box_PIL, plot_boxes_PIL
from yolov7.utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel

sys.path.append('./STLPRNet')
from STLPRNet.data.STLPRNDataModule import STLPRNDataModule
from STLPRNet.model.LPRNET import LPRNet, CHARS
from STLPRNet.model.STN import STNet
from STLPRNet.model.STLPRNet import STLPRNet, decode


class VideoReader(Thread):
    def __init__(self,
                 path):
        Thread.__init__(self)
        if os.path.isfile(path):
            self.cap = cv2.VideoCapture(path)
        else:
            assert "Video file not exist."
        self.frameWidth = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # 영상의 넓이(가로) 프레임
        self.frameHeight = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # 영상의 높이(세로) 프레임
        self.frame_size = (self.frameWidth, self.frameHeight)
        self.frame_counts = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print('frame_size={}, frames={}'.format(self.frame_size, self.frame_counts))

    def run(self):
        is_full = False
        while True:
            is_full = self.check_buffer()
            if not is_full:
                if self.cap.grab():
                    ret, frame = self.cap.retrieve()
                    buffer.append(frame)
                else:
                    if self.cap.isOpened():
                        self.cap.release()
                    print("Process VideoReader is dead.")
                    return
            time.sleep(0.01)

    def check_buffer(self):
        if len(buffer) > self.frame_counts * 0.02:
            return True
        elif len(buffer) < self.frame_counts * 0.005:
            return False


class LicensePlateDetector(Thread):
    def __init__(self,
                 device,
                 yolo_weights,
                 img_size,
                 augment=False,
                 conf_thres=0.7,
                 iou_thres=0.45,
                 agnostic_nms=False):
        Thread.__init__(self)
        self.buffer = buffer
        self.agnostic_nms = agnostic_nms
        self.classes = 0
        self.iou_thres = iou_thres
        self.conf_thres = conf_thres
        self.augment = augment
        self.device = select_device(device)
        self.half = self.device.type != 'cpu'
        # Load model
        self.yolo = attempt_load(yolo_weights, map_location=self.device)  # load FP32 model
        # self.lprn = STLPRNet().load_from_checkpoint(lprn_weights)
        self.stride = int(self.yolo.stride.max())  # model stride
        self.img_size = check_img_size(img_size, s=self.stride)

        if self.half:
            self.yolo.half()  # to FP16

    def run(self):
        # Run inference
        if self.device.type != 'cpu':
            self.yolo(torch.zeros(1, 3, self.img_size, self.img_size).to(self.device).type_as(
                next(self.yolo.parameters())))  # run once
        old_img_w = old_img_h = self.img_size
        old_img_b = 1

        while True:
            if buffer:
                self.detect(buffer.popleft())
            else:
                if not vr.cap.isOpened():
                    print("Process Detector is Dead.")
                    return
            time.sleep(0.01)

    def detect(self, im0):
        # Padded resize
        img = letterbox(im0, self.img_size, stride=self.stride)[0]

        # Convert
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)

        img = torch.from_numpy(img).to(self.device)
        img = img.half() if self.half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0

        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        pred = self.yolo(img, augment=self.augment)[0]

        # Apply NMS
        pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, classes=self.classes,
                                   agnostic=self.agnostic_nms)

        xyxys = []
        lp_preds = []
        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
                xyxys = [list(map(int, torch.tensor(xyxy).view(1, 4).view(-1).tolist()))
                         for *xyxy, conf, cls in reversed(det)]
        LPs.append((im0, xyxys))


class LicensePlateReader(Thread):
    def __init__(self, lprn_weights):
        Thread.__init__(self)
        self.lprn = STLPRNet().load_from_checkpoint(lprn_weights)
        self.colors = [random.randint(0, 255) for _ in range(3)]

    def run(self):
        while True:
            if LPs:
                self.detect(*LPs.popleft())
            else:
                if not buffer and not vr.cap.isOpened():
                    print("Process Reader is Dead.")
                    return

            time.sleep(0.001)

    def detect(self, img, xyxys):
        imgs = [img[y1:y2, x1:x2] for x1, y1, x2, y2 in xyxys]
        im0 = plot_boxes_PIL(xyxys,
                             img,
                             labels=self.lprn.detect(imgs) if xyxys else [],
                             color=self.colors,
                             line_thickness=1)
        outputs.append(im0)


def update_bars(bars):
    for bar, dq in zip(bars, [buffer, LPs, outputs]):
        if len(dq) / 500 > 0.8:
            _color = 'green'
        elif len(dq) / 500 > 0.2:
            _color = 'white'
        elif len(dq) / 500 > 0.1:
            _color = 'yellow'
        else:
            _color = 'red'
        bar.count = len(dq)
        bar.color = _color
        bar.update(0)


class VideoViewer(Thread):
    def __init__(self, fps, manager):
        Thread.__init__(self)
        self.fps = fps
        self.manager = manager
        print("Video Viewer is ready")

    def run(self):
        start = False
        while player_bar.count < vr.frame_counts:
            if len(outputs) > vr.frame_counts * 0.05:
                start = True
                status_bar.update(stage='Playing')

            t0 = time_synchronized()
            if outputs and start:
                img = outputs.popleft()
                cv2.imshow('result', img)
                cv2.waitKey(1)
                player_bar.update()
            time.sleep(1 / self.fps * 0.75)
            t1 = time_synchronized()

            update_bars([buffer_bar, lp_bar, output_bar])
            status_bar.update(fps=f'{1 / (t1 - t0):.2f}')

        status_bar.update(stage='Done.')
        print("Process VideoViewer Ended")
        cv2.destroyAllWindows()
        return


if __name__ == '__main__':
    buffer = deque()
    LPs = deque()
    outputs = deque()

    manager = enlighten.get_manager()
    status_format = '{program}{fill}Stage: {stage}{fill} FPS: {fps}'
    status_bar = manager.status_bar(color='bold_bright_black_on_white',
                                    status_format=status_format,
                                    program='License Plate Detection',
                                    stage='Loading',
                                    fps=f'{0.0:.2f}',
                                    position=7)
    bar_format = '{desc}{desc_pad}{percentage:3.0f}%|{bar}| {count:{len_total}d}{unit_pad}{unit}s'
    buffer_bar = manager.counter(total=500, desc='  buffer', unit='frame', bar_format=bar_format, position=5)
    lp_bar = manager.counter(total=500, desc='  lp    ', unit='lp   ', bar_format=bar_format, position=4)
    output_bar = manager.counter(total=500, desc='  output', unit='frame', bar_format=bar_format, position=3)

    vr = VideoReader(path='test/videoplayback (2).mp4')
    lpd = LicensePlateDetector(
        device='0',
        yolo_weights='yolov7/runs/train/yolov7-custom7/weights/best.pt',
        img_size=640
    )
    lpr = LicensePlateReader(
        lprn_weights='STLPRNet/saving_ckpt1/best.ckpt',
    )

    pb_format = '{desc}{desc_pad}{percentage:3.0f}%|{bar}| {count:{len_total}d}/{total:d} [{elapsed}<{eta}]'
    player_bar = manager.counter(total=vr.frame_counts, desc='Total', unit='frame', bar_format=pb_format,
                                 color='bold_bright_black_on_white', position=1)
    vr.start()
    lpd.start()
    lpr.start()

    status_bar.update(stage='Initializing')
    vv = VideoViewer(fps=30, manager=manager)
    vv.start()
