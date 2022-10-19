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
        if os.path.isfile(path):
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
                 conf_thres=0.5,
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
        # Load model
        self.yolo = attempt_load(yolo_weights, map_location=self.device).eval()  # load FP32 model
        self.stride = int(self.yolo.stride.max())  # model stride
        self.img_size = check_img_size(img_size, s=self.stride)

        if self.half:
            self.yolo.half()  # to FP16

        self.tracker = Sort(max_age=30, min_hits=3)

    def run(self):
        # Run inference
        if self.device.type != 'cpu':
            self.yolo(torch.zeros(1, 3, self.img_size, self.img_size).to(self.device).type_as(
                next(self.yolo.parameters())))  # run once
        old_img_w = old_img_h = self.img_size
        old_img_b = 1

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
        model_statuses[0].update(time=f'{(t1 - t0) * 1000:2.2f}')

        xyxys = []
        tracks = np.empty((0, 5))
        # Process detections
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
        model_statuses[1].update(time=f'{(t3 - t2) * 1000:2.2f}')
        LPs.append((im0, tracks[:, :4], tracks[:, 4]))


class LicensePlateReader(Thread):
    def __init__(self, lprn_weights, device='0'):
        Thread.__init__(self)
        self.colors = [random.randint(0, 255) for _ in range(3)]
        self.device = select_device(device)
        self.lprn = STLPRNet().load_from_checkpoint(lprn_weights).to(self.device).eval()
        self.half = self.device.type != 'cpu'
        if self.half:
            self.lprn.half()

    def run(self):
        while True:
            if LPs:
                self.detect(*LPs.popleft())
            else:
                if not origin and LPs:
                    print("Process Reader is Dead.")
                    return

            time.sleep(0.001)

    def detect(self, img, xyxys, ids):
        plates = [img[y1:y2, x1:x2].astype(np.uint8) for x1, y1, x2, y2 in xyxys]

        t0 = time.time()
        preds = self.lprn.detect_imgs(plates, self.device, self.half)
        t1 = time.time()
        if plates:
            model_statuses[2].update(time=f'{(t1 - t0) * 1000 / len(plates):2.2f}')

        for id, pred in zip(ids, preds):
            while len(id_list) - 1 < id:
                id_list.append(Counter())
            if self.lprn.check(pred):
                id_list[id].update([pred])

        preds_by_id = [id_list[id].most_common()[0][0] if id_list[id] else "Unknown" for id in ids]
        im0 = plot_boxes_PIL(xyxys,
                             img,
                             labels=preds_by_id,
                             color=self.colors,
                             line_thickness=1)
        buffer.append(im0)


def update_bars(bars):
    for bar, dq in zip(bars, [origin, LPs, buffer]):
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
    def __init__(self, fps, manager):
        Thread.__init__(self)
        self.fps = fps
        self.manager = manager
        print("Video Viewer is ready")

    def run(self):
        start = False
        while play_bar.count < vr.frame_counts:
            if len(buffer) > 100:
                start = True
                status_bar.update(stage='Playing')

            t0 = time.time()
            if buffer and start:
                img = buffer.popleft()
                cv2.imshow('result', img)
                cv2.waitKey(1)
                play_bar.update()
            time.sleep(1 / self.fps * 0.75)
            t1 = time.time()

            update_bars(bars)
            status_bar.update(fps=f'{1 / (t1 - t0):2.2f}')

        status_bar.update(stage='Done.')
        print("Process VideoViewer Ended")
        return


if __name__ == '__main__':
    VIDEO_PATH = 'test/video3.mp4'
    origin = deque()
    LPs = deque()
    buffer = deque()
    id_list = deque(Counter())

    # display bars
    manager = enlighten.get_manager()
    status_format = '{program}{fill} Stage: {stage}{fill} FPS: {fps}'
    file_format = '[ file ]   {file}, {width}x{height}, {length} frames{fill}'
    status_bar = manager.status_bar(color='bold_bright_black_on_white',
                                    status_format=status_format,
                                    program='License Plate Detection',
                                    stage='Loading',
                                    fps=f'{0.0:.2f}',
                                    position=10)
    vid_info_bar = manager.status_bar(status_format=file_format,
                                      file=os.path.split(VIDEO_PATH)[-1],
                                      width='0',
                                      height='0',
                                      length='0',
                                      position=9)
    bar_format = '{desc}{desc_pad} {percentage:3.0f}%|{bar}| {count:{len_total}d}{unit_pad}{unit}s'
    bars = [
        manager.counter(total=200, desc='[origin]', unit='frame', bar_format=bar_format, position=8),
        manager.counter(total=200, desc='[plates]', unit='plate', bar_format=bar_format, position=5),
        manager.counter(total=200, desc='[buffer]', unit='frame', bar_format=bar_format, position=3)
    ]
    model_format = '[{name}]   {time}ms/{unit}{fill}'
    model_statuses = [
        manager.status_bar(status_format=model_format, name='yolov7', time='0.0', unit='frame', position=7),
        manager.status_bar(status_format=model_format, name=' SORT ', time='0.0', unit='frame', position=6),
        manager.status_bar(status_format=model_format, name='STLPRN', time='0.0', unit='plate', position=4),
        manager.status_bar(status_format='   ↓', position=2)
    ]

    # Threads
    vr = VideoReader(path=VIDEO_PATH)
    lpd = LicensePlateDetector(
        device='0',
        yolo_weights='weights/yolov7-best.pt',
        img_size=640
    )
    lpr = LicensePlateReader(
        lprn_weights='weights/stlprn-best.pt',
    )

    pb_format = '{desc}{desc_pad}{percentage:3.0f}%|{bar}| {count:{len_total}d}/{total:d} [{elapsed}<{eta}]'
    play_bar = manager.counter(total=vr.frame_counts, desc='[  ▶   ]', unit='frame', bar_format=pb_format,
                               color='white_on_black', position=1)
    vid_info_bar.update(width=vr.frameWidth,
                        height=vr.frameHeight,
                        length=vr.frame_counts)

    vr.daemon = True
    lpd.daemon = True
    lpr.daemon = True

    vr.start()
    lpd.start()
    lpr.start()

    status_bar.update(stage='Initializing')
    vv = VideoViewer(fps=30, manager=manager)
    vv.daemon = True
    vv.start()

    while vv.is_alive():
        time.sleep(5)
    cv2.destroyAllWindows()
