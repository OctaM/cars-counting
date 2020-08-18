# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""A demo that runs object detection on camera frames using OpenCV.

TEST_DATA=../all_models

Run face detection model:
python3 detect.py \
  --model ${TEST_DATA}/mobilenet_ssd_v2_face_quant_postprocess_edgetpu.tflite

Run coco model:
python3 detect.py \
  --model ${TEST_DATA}/mobilenet_ssd_v2_coco_quant_postprocess_edgetpu.tflite \
  --labels ${TEST_DATA}/coco_labels.txt

"""
import argparse
import collections
import common
import cv2
import numpy as np
import os
import re
import datetime
import csv
import tflite_runtime.interpreter as tflite

from PIL import Image
from utils.centroidtracker import CentroidTracker
from utils import label_map_util_custom


PATH_TO_LABELS = os.path.join('../all_models', 'coco_labels.txt')

NUM_CLASSES = 90

# label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
# categories = label_map_util.convert_label_map_to_categories(label_map,
#                                                             max_num_classes=NUM_CLASSES, use_display_name=True)
# category_index = label_map_util.create_category_index(categories)
categories = label_map_util_custom.load_labelmap(PATH_TO_LABELS)
category_index = label_map_util_custom.create_category_index(categories)

Object = collections.namedtuple('Object', ['id', 'score', 'bbox'])


def load_labels(path):
    p = re.compile(r'\s*(\d+)(.+)')
    with open(path, 'r', encoding='utf-8') as f:
       lines = (p.match(line).groups() for line in f.readlines())
       return {int(num): text.strip() for num, text in lines}


class BBox(collections.namedtuple('BBox', ['xmin', 'ymin', 'xmax', 'ymax'])):
    """Bounding box.
    Represents a rectangle which sides are either vertical or horizontal, parallel
    to the x or y axis.
    """
    __slots__ = ()


def get_output(interpreter, score_threshold, top_k, image_scale=1.0):
    """Returns list of detected objects."""
    boxes = common.output_tensor(interpreter, 0)
    class_ids = common.output_tensor(interpreter, 1)
    scores = common.output_tensor(interpreter, 2)
    count = int(common.output_tensor(interpreter, 3))

    def make(i):
        ymin, xmin, ymax, xmax = boxes[i]
        return Object(
            id=int(class_ids[i]),
            score=scores[i],
            bbox=BBox(xmin=np.maximum(0.0, xmin),
                      ymin=np.maximum(0.0, ymin),
                      xmax=np.minimum(1.0, xmax),
                      ymax=np.minimum(1.0, ymax)))

    return [make(i) for i in range(top_k) if scores[i] >= score_threshold], boxes, class_ids, scores, count


def main():
    default_model_dir = '../all_models'
    default_model = 'mobilenet_ssd_v2_coco_quant_postprocess_edgetpu.tflite'
    default_labels = 'coco_labels.txt'
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help='.tflite model path',
                        default=os.path.join(default_model_dir,default_model))
    parser.add_argument('--labels', help='label file path',
                        default=os.path.join(default_model_dir, default_labels))
    parser.add_argument('--top_k', type=int, default=3,
                        help='number of categories with highest score to display')
    parser.add_argument('--camera_idx', type=int, help='Index of which video source to use. ', default = 0)
    parser.add_argument('--threshold', type=float, default=0.1,
                        help='classifier score threshold')
    args = parser.parse_args()

    print('Loading {} with {} labels.'.format(args.model, args.labels))
    interpreter = common.make_interpreter(args.model)
    interpreter.allocate_tensors()
    labels = load_labels(args.labels)
    detection_threshold = 0.5

    counted_cars_ids = []
    counted_cars = 0
    line_coords = 345
    frames_until_reset = 0
    csv_columns = ["Number", "Type", "Date"]
    csv_dict = []

    cap = cv2.VideoCapture(args.camera_idx)
    # fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    # out = cv2.VideoWriter('output.mp4', fourcc, 20.0, (640,352))

    ct = CentroidTracker()
    with open("output.csv", "w") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=csv_columns)
        writer.writeheader()
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            cv2_im = frame
            frames_until_reset += 1

            cv2_im_rgb = cv2.cvtColor(cv2_im, cv2.COLOR_BGR2RGB)
            pil_im = Image.fromarray(cv2_im_rgb)

            (h, w) = cv2_im.shape[:2]
            rects = []
            common.set_input(interpreter, pil_im)
            interpreter.invoke()
            objs, boxes, classes, scores, count = get_output(interpreter, score_threshold=args.threshold, top_k=args.top_k)
            boxes = np.squeeze(boxes)
            classes = np.squeeze(classes).astype(np.int32)
            scores = np.squeeze(scores)

            for ind in range(len(boxes)):
                if scores[ind] > detection_threshold and (classes[ind] == 2 or classes[ind] == 7):

                    box = boxes[ind] * np.array([h, w, h, w])
                    box = np.append(box, classes[ind])
                    rects.append(box.astype('int'))

                    (startY, startX, endY, endX, label) = box.astype("int")
                    cv2.rectangle(cv2_im, (startX, startY), (endX, endY),
                                  (0, 255, 0), 2)

            # cv2_im = append_objs_to_img(cv2_im, objs, labels)

            objects = ct.update(rects)

            for (objectID, centroid) in objects.items():
                # draw both the ID of the object and the centroid of the
                # object on the output frame

                if centroid[1] > line_coords:
                    if objectID not in counted_cars_ids and centroid[0] <= 570:
                        counted_cars += 1
                        object_type = labels[centroid[2]]
                        new_entry = {"Number": counted_cars, "Type": object_type, "Date": datetime.datetime.now()}
                        writer.writerow(new_entry)
                    counted_cars_ids.append(objectID)

                text = "ID {}".format(objectID)
                cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)

            # if frames_until_reset == 180:
            #     counted_cars_ids = []
            #     frames_until_reset = 0

            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(
                cv2_im,
                'Detected Vehicles: ' + str(counted_cars),
                (10, 35),
                font,
                0.8,
                (0, 0xFF, 0xFF),
                2,
                cv2.FONT_HERSHEY_SIMPLEX,
            )

            cv2.line(cv2_im, (0, line_coords), (w, line_coords), (0, 0, 0xFF), 5)
            cv2.line(cv2_im, (570, line_coords), (570, h), (0, 0, 0xFF), 5)

            # out.write(frame)
            cv2.imshow('frame', cv2_im)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        # out.release()
        cv2.destroyAllWindows()


def append_objs_to_img(cv2_im, objs, labels):
    height, width, channels = cv2_im.shape
    for obj in objs:
        x0, y0, x1, y1 = list(obj.bbox)
        x0, y0, x1, y1 = int(x0*width), int(y0*height), int(x1*width), int(y1*height)
        percent = int(100 * obj.score)
        label = '{}% {}'.format(percent, labels.get(obj.id, obj.id))

        cv2_im = cv2.rectangle(cv2_im, (x0, y0), (x1, y1), (0, 255, 0), 2)
        cv2_im = cv2.putText(cv2_im, label, (x0, y0+30),
                             cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 2)
    return cv2_im


if __name__ == '__main__':
    main()