import os.path
import sys
import time
from collections import deque, Counter
from threading import Thread
import cv2
import torch
import torch.backends.cudnn
from numpy import random
import numpy as np
import enlighten
import warnings
from STLPRNet.model.STLPRNet import STLPRNet
from sort.sort import Sort
from draw.draw import plot_boxes_PIL

sys.path.append('./yolov7')
from models.experimental import attempt_load
from utils.datasets import letterbox
from utils.general import check_img_size, non_max_suppression, scale_coords
from utils.torch_utils import select_device

warnings.filterwarnings("ignore")


class VideoReader(Thread):
    def __init__(self,
                 path):
        Thread.__init__(self)
        if os.path.exists(path):
            self.cap = cv2.VideoCapture(path)
        else:
            assert "Video file not exist."
        self.frameWidth = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # 영상의 넓이(가로) 프레임
        self.frameHeight = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # 영상의 높이(세로) 프레임
        self.frame_size = (self.frameWidth, self.frameHeight)
        self.frame_counts = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def run(self):
        is_full = False
        while True:
            is_full = self.check_buffer()
            if not is_full:
                if self.cap.grab():
                    ret, frame = self.cap.retrieve()
                    origin.append(frame)
                else:
                    if self.cap.isOpened():
                        self.cap.release()
                    print("Process VideoReader is dead.")
                    return
            time.sleep(0.01)

    def check_buffer(self):
        if len(origin) > 100:
            return True
        elif len(origin) < 50:
            return False


class LicensePlateDetector(Thread):
    def __init__(self,
                 device,
                 yolo_weights,
                 img_size,
                 augment=False,
                 conf_thres=0.25,
                 iou_thres=0.45,
                 agnostic_nms=False):
        Thread.__init__(self)
        self.agnostic_nms = agnostic_nms
        self.classes = 0
        self.iou_thres = iou_thres
        self.conf_thres = conf_thres
        self.augment = augment
        self.device = select_device(device)
        self.half = self.device.type != 'cpu'
        self.tracker = Sort(max_age=30, min_hits=3)

        # Load model
        self.yolo = attempt_load(yolo_weights, map_location=self.device).eval()  # load FP32 model
        self.stride = int(self.yolo.stride.max())  # model stride
        self.img_size = check_img_size(img_size, s=self.stride)

        if self.half:
            self.yolo.half()  # to FP16

    def run(self):
        # Run inference
        if self.device.type != 'cpu':
            self.yolo(torch.zeros(1, 3, self.img_size, self.img_size).to(self.device).type_as(
                next(self.yolo.parameters())))  # run once

        while True:
            if origin:
                self.detect(origin.popleft())
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
        t0 = time.time()
        pred = self.yolo(img, augment=self.augment)[0]

        # Apply NMS
        pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, classes=self.classes,
                                   agnostic=self.agnostic_nms)
        t1 = time.time()
        bars['yolo'].update(time=f'{(t1 - t0) * 1000:2.2f}')

        # Process detections
        tracks = np.empty((0, 5))
        t2 = time.time()
        for i, det in enumerate(pred):  # detections per image
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
                # xyxys = [list(map(int, torch.tensor(xyxy).view(1, 4).view(-1).tolist()))
                #          for *xyxy, conf, cls in reversed(det)]
                xyxys_conf = np.array([list(map(int, torch.tensor(xyxy).view(1, 4).view(-1).tolist())) + [conf.tolist()]
                                       for *xyxy, conf, cls in reversed(det)])
                tracks = self.tracker.update(xyxys_conf).astype(int)
        t3 = time.time()

        bars['sort'].update(time=f'{(t3 - t2) * 1000:2.2f}')
        plates.append((im0, tracks[:, :4], tracks[:, 4]))


class LicensePlateReader(Thread):
    def __init__(self, stlprn_weights, device='0'):
        Thread.__init__(self)
        self.colors = np.random.randint(127, size=(255, 3))
        self.device = select_device(device)
        self.stlprn = STLPRNet().load_from_checkpoint(stlprn_weights).to(self.device).eval()
        self.half = self.device.type != 'cpu'
        if self.half:
            self.stlprn.half()

    def run(self):
        while True:
            if plates:
                self.detect(*plates.popleft())
            else:
                if not origin and plates:
                    print("Process Reader is Dead.")
                    return

            time.sleep(0.001)

    def detect(self, img, xyxys, ids):
        plate_imgs = [img[y1:y2, x1:x2].astype(np.uint8) for x1, y1, x2, y2 in xyxys]

        t0 = time.time()
        preds = self.stlprn.detect_imgs(plate_imgs, self.device, self.half)
        t1 = time.time()

        if plate_imgs:
            bars['stlprn'].update(time=f'{(t1 - t0) * 1000 / len(plate_imgs):2.2f}')

        for id, pred in zip(ids, preds):
            while len(id_list) - 1 < id:
                id_list.append(Counter())

            if self.stlprn.check(pred):
                id_list[id].update([pred])

        preds_by_id = [id_list[id].most_common()[0][0] if id_list[id] else "Unknown" for id in ids]
        im0 = plot_boxes_PIL(xyxys, img, labels=preds_by_id, ids=ids, colors=self.colors, line_thickness=2)
        buffer.append(im0)


def update_bars(bars):
    for bar, dq in zip(bars, [origin, plates, buffer]):
        if len(dq) / 100 > 0.8:
            _color = 'green'
        elif len(dq) / 100 > 0.2:
            _color = 'white'
        elif len(dq) / 100 > 0.1:
            _color = 'yellow'
        else:
            _color = 'red'
        bar.count = len(dq)
        bar.color = _color
        bar.update(0)


class VideoViewer(Thread):
    def __init__(self, fps):
        Thread.__init__(self)
        self.fps = fps
        print("Video Viewer is ready")

    def run(self):
        start = False
        while bars['player'].count < vr.frame_counts:
            if len(buffer) > 100:
                start = True
                bars['status'].update(stage='Playing')

            t0 = time.time()
            if buffer and start:
                img = buffer.popleft()
                cv2.imshow('result', img)
                cv2.waitKey(1)
                bars['player'].update()
            time.sleep(1 / self.fps * 0.75)
            t1 = time.time()

            update_bars([bars['origin'], bars['plates'], bars['buffer']])
            bars['status'].update(fps=f'{1 / (t1 - t0):2.2f}')

        bars['status'].update(stage='Done.')
        print("Process VideoViewer Ended")
        return


def init_bars(manager):
    bar_formats = {
        'status': '{program}{fill} Stage: {stage}{fill} FPS: {fps}',
        'file': '[ file ]   {file}, {width}x{height}, {length} frames{fill}',
        'counter': '{desc}{desc_pad} {percentage:3.0f}%|{bar}| {count:{len_total}d}{unit_pad}{unit}s',
        'inference': '[{name}]   {time}ms/{unit}{fill}',
        'player': '{desc}{desc_pad}{percentage:3.0f}%|{bar}| {count:{len_total}d}/{total:d} [{elapsed}<{eta}]'
    }
    bars = {
        'status': manager.status_bar(color='bold_bright_black_on_white', status_format=bar_formats['status'],
                                     program='License Plate Detection', stage='Loading', fps=f'{0.0:.2f}', position=11),
        'video': manager.status_bar(status_format=bar_formats['file'], file=os.path.split(VIDEO_PATH)[-1],
                                    width=vr.frameWidth, height=vr.frameHeight, length=vr.frame_counts, position=9),
        'origin': manager.counter(total=200, desc='[origin]', unit='frame', bar_format=bar_formats['counter'],
                                  position=8),
        'yolo': manager.status_bar(status_format=bar_formats['inference'], name='yolov7', time='0.0', unit='frame',
                                   position=7),
        'sort': manager.status_bar(status_format=bar_formats['inference'], name=' SORT ', time='0.0', unit='frame',
                                   position=6),
        'plates': manager.counter(total=200, desc='[plates]', unit='plate', bar_format=bar_formats['counter'],
                                  position=5),
        'stlprn': manager.status_bar(status_format=bar_formats['inference'], name='STLPRN', time='0.0', unit='plate',
                                     position=4),
        'buffer': manager.counter(total=200, desc='[buffer]', unit='frame', bar_format=bar_formats['counter'],
                                  position=3),
        'down': manager.status_bar(status_format='   ↓', position=2),
        'player': manager.counter(total=vr.frame_counts, desc='[  ▶   ]', unit='frame',
                                  bar_format=bar_formats['player'], color='white_on_black', position=1)
    }
    return bars


if __name__ == '__main__':
    VIDEO_PATH = 'test/video1.mp4'

    # Queues
    origin, plates, buffer, id_list = deque(), deque(), deque(), deque()

    # Threads
    vr = VideoReader(path=VIDEO_PATH)
    lpd = LicensePlateDetector(device='0', yolo_weights='weights/yolov7-best.pt', img_size=480)
    lpr = LicensePlateReader(stlprn_weights='weights/stlprn-best.pt')
    vv = VideoViewer(fps=30)
    vr.daemon, lpd.daemon, lpr.daemon, vv.daemon = True, True, True, True

    # display bars
    manager = enlighten.get_manager()
    bars = init_bars(manager)

    # Start Detect & Recognition
    bars['status'].update(stage='Initializing')
    vr.start(); lpd.start(); lpr.start(); vv.start()

    # End
    while vv.is_alive():
        time.sleep(5)
    cv2.destroyAllWindows()
