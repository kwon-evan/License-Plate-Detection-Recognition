'''
@Author: kwon-evan
'''

import warnings
import time
from collections import deque, Counter
from threading import Thread
import torch
from cv2 import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import enlighten

from STLPRNet.model.STLPRNet import STLPRNet
from sort.sort import Sort
from yolo import Yolo

warnings.filterwarnings("ignore")

class VideoReader(Thread):
    def __init__(self, path: str):
        Thread.__init__(self)

        self.cap = cv2.VideoCapture(path)

        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.frame_size = (self.frame_width, self.frame_height)
        self.frame_total = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def run(self):
        while True:
            if not self.is_full(0.7, 200):
                if self.cap.grab():
                    _, frame = self.cap.retrieve()
                    origin.append(frame)
                else:
                    if self.cap.isOpened():
                        self.cap.release()
                    print("Process VideoReader is Done.")
                    return
            time.sleep(0.01)

    def is_full(self, thres, max_len):
        return True if len(origin) > max_len * thres else False


class PlateDetetor(Thread, Yolo):
    def __init__(self, root: str, batch_size:int =1, gpus:int =0):
        # Thread
        Thread.__init__(self)
        # Yolo
        Yolo.__init__(
                self,
                configPath=root + 'model.cfg',
                weightPath=root + 'model.weights',
                metaPath=root + 'model.data',
               batch_size=batch_size,
                gpus=gpus
        )
        # SORT
        self.tracker = Sort(max_age=30, min_hits=3)

    def run(self):
        while True:
            if origin:
                # Get Image from origin
                img = origin.popleft()

                # Yolo Inference
                t0 = time.time()
                pred = self.detect(img)
                pred = np.asarray(pred)
                t1 = time.time()
                bars['YOLO'].update(time=f'{(t1 - t0) * 1000:5.3f}')

                # SORT Algorithm
                t2 = time.time()
                if pred.any():
                    tracks = self.tracker.update(pred).astype(int)
                else:
                    tracks = np.empty((0, 5))
                t3 = time.time()
                bars['SORT'].update(time=f'{(t3 - t2) * 1000:5.3f}')

                # img, bbox, id
                plates.append((img, tracks[:, :4], tracks[:, 4]))

            else:
                if not vr.cap.isOpened():
                    print("Plate Detector is Done.")
                    return
            time.sleep(0.01)

class PlateReader(Thread, STLPRNet):
    def __init__(self):
        # THREAD
        Thread.__init__(self)

        # STLPRNet
        STLPRNet.__init__(self)

    def run(self):
        while True:
            t0 = time.time()
            if plates:
                img, xyxys, ids = plates.popleft()
                detect_cnt = 0

                if len(xyxys) > 0:
                    # extend id_list if len is less than current id
                    if len(id_list) < max(ids) + 1:
                        id_list.extend([Counter()] * (max(ids) + 1 - len(id_list)))

                    # update Counter of id list
                    for i, xyxy in zip(ids, xyxys):
                        x1, y1, x2, y2 = xyxy

                        if cv2.pointPolygonTest(polygon_coords, ((x1 + x2) // 2, (y1 + y2) // 2), measureDist=False) > 0:
                            pred = self.detect(img[y1:y2, x1:x2], device=DEVICE)
                            
                            if self.check(pred):
                                id_list[i].update([pred])

                            detect_cnt += 1

                buffer.append((img, xyxys, ids, [id_list[i].most_common()[0][0] if id_list[i] else "Unknown" for i in ids]))
                t1 = time.time()
                bars['LPDR'].update(time=f'{(t1 - t0) * 1000:5.3f}', plates=f'{detect_cnt}/{len(xyxys)} plates')
            else:
                if not origin and not plates and not pd.is_alive():
                    print("Plate Reader is Done.")
                    return
            time.sleep(0.001)

class BoxDrawer(Thread):
    def __init__(self):
        Thread.__init__(self)
        self.colors = np.random.randint(127, size=(255, 3))

    def run(self):
        while vr.is_alive() or buffer:
            if buffer:
                t0 = time.time()
                img, xyxys, ids, labels = buffer.popleft()

                # draw Contours
                img = cv2.drawContours(
                        img,
                        [polygon_coords],
                        -1,
                        color=(255, 0, 0),
                        thickness=2,
                        lineType=cv2.LINE_4
                )

                # draw boxs & labels
                for xyxy, i, label in zip(xyxys, ids, labels):
                    x1, y1, x2, y2 = xyxy

                    if cv2.pointPolygonTest(polygon_coords, ((x1 + x2) // 2, (y1 + y2) // 2), measureDist=False) > 0:
                        img = self.plot_box(img, xyxy, i, label)

                done.append(img)
                t1 = time.time()
                bars['DRAW'].update(time=f'{(t1 - t0) * 1000:5.3f}')

            time.sleep(0.01)

    def plot_box(self, img, box: list[float], i: int, label: str, line_thickness: float=2):
        img = Image.fromarray(img)
        draw = ImageDraw.Draw(img)
        line_thickness = line_thickness or max(int(min(img.size) / 200), 2)

        draw.rectangle(
                ((box[0], box[1]), (box[2], box[3])),
                width=line_thickness,
                outline=tuple(self.colors[i % 255])
        )

        fontsize = max(round(max(img.size) / 100), 12)
        font = ImageFont.truetype("./NotoSansKR-Medium.otf", fontsize)
        txt_width, txt_height = font.getsize(label)
        draw.rectangle(
                ((box[0], box[1] - txt_height + 4.), (box[0] + txt_width, box[1])),
                fill=tuple(self.colors[i % 255])
        )
        draw.text((box[0] + 1, box[1] - txt_height - 1), label, fill=(255, 255, 255), font=font)
        return np.asarray(img)


def init_bars(manager): 
    status_format = '[{name}]  {time}ms/{unit}{fill}'
    bars = {
            'FPS' : manager.status_bar(
                status_format=status_format, name=' FPS', time='0.0', unit='frame', position=5
            ),
            'YOLO' : manager.status_bar(
                status_format=status_format, name='YOLO', time='0.0', unit='frame', position=4
            ),
            'SORT' : manager.status_bar(
                status_format=status_format, name='SORT', time='0.0', unit='frame', position=3
            ),
            'LPDR' : manager.status_bar(
                status_format='[{name}]  {time}ms/{unit}, {plates}{fill}',
                name='LPDR', time='0.0', unit='frame', plates='0 plates', position=2
            ),
            'DRAW' : manager.status_bar(
                status_format=status_format, name='DRAW', time='0.0', unit='frame', position=1
            )
    }
    return bars


if __name__ == '__main__':
    VIDEO_PATH='test/test4.mp4'
    STLPRNet_WEIGHT='weights/stlprn-best.pt'

    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    FPS = 30

    # ROI Polygon Coords
    polygon_coords = np.ndarray((6, 1, 2), dtype=np.uint)

    polygon_coords[0] = np.array([[ 164, 382]])  # left top
    polygon_coords[1] = np.array([[1052, 392]]) # right top
    polygon_coords[2] = np.array([[1279, 589]]) # right bottom
    polygon_coords[3] = np.array([[1279, 729]]) # right bottom
    polygon_coords[4] = np.array([[   0, 729]]) # right bottom
    polygon_coords[5] = np.array([[   0, 593]])

    # QUEUE
    origin, plates, buffer, done, id_list = deque(), deque(), deque(), deque(), deque()

    # THREADS
    vr = VideoReader(path=VIDEO_PATH)
    pd = PlateDetetor(root='weights/darknet-plate/')
    pr = PlateReader().load_from_checkpoint(STLPRNet_WEIGHT).to(DEVICE).eval()
    bd = BoxDrawer()

    vr.daemon = True
    pd.daemon = True
    pr.daemon = True
    bd.daemon = True

    vr.start()
    pd.start()
    pr.start()
    bd.start()

    # display bars
    manager = enlighten.get_manager()
    bars = init_bars(manager)

    while bd.is_alive() or done:
        t0 = time.time()
        if len(done) > 0:
            cv2.imshow('frame', done.popleft())
            if cv2.waitKey(1) & 0xFF == ord('x'):
                break
        time.sleep(1 / FPS * 0.85)
        t1 = time.time()
        bars['FPS'].update(time=f'{1 / (t1 - t0):5.3f}')

    cv2.destroyAllWindows()

