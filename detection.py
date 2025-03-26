import os
import cv2
import numpy as np
import onnxruntime
from typing import Tuple, List

def distance2bbox(points, distance):
    """Convert distance to bounding box."""
    x1 = points[:, 0] - distance[:, 0]
    y1 = points[:, 1] - distance[:, 1]
    x2 = points[:, 0] + distance[:, 2]
    y2 = points[:, 1] + distance[:, 3]
    return np.stack([x1, y1, x2, y2], axis=-1)

def distance2kps(points, distance):
    """Convert distance to keypoints (eyes, nose, mouth corners)."""
    preds = []
    for i in range(5):  # 5 Keypoints
        px = points[:, 0] + distance[:, i * 2]
        py = points[:, 1] + distance[:, i * 2 + 1]
        preds.append(px)
        preds.append(py)
    return np.stack(preds, axis=-1)

class SCRFD:
    def __init__(self, model_path: str, input_size: Tuple[int, int] = (640, 640), conf_thres: float = 0.3, iou_thres: float = 0.4) -> None:
        self.input_size = input_size
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres

        self.fmc = 3
        self._feat_stride_fpn = [8, 16, 32]
        self._num_anchors = 2
        self.use_kps = True

        self.mean = 127.5
        self.std = 128.0

        self.center_cache = {}
        self._initialize_model(model_path=model_path)

    def _initialize_model(self, model_path: str):
        try:
            self.session = onnxruntime.InferenceSession(
                model_path, providers=["CPUExecutionProvider"]
            )
            self.output_names = [x.name for x in self.session.get_outputs()]
            self.input_names = [x.name for x in self.session.get_inputs()]
            print("Model loaded successfully!")
        except Exception as e:
            print(f"Failed to load the model: {e}")
            raise

    def forward(self, image: np.ndarray, threshold: float):
        scores_list, bboxes_list, kpss_list = [], [], []

        # Ensure correct shape for blob
        input_blob = cv2.dnn.blobFromImage(
            image, 1.0 / self.std, self.input_size, (self.mean, self.mean, self.mean), swapRB=True
        )
        input_blob = np.expand_dims(input_blob[0], axis=0)  # Ensure batch dimension

        outputs = self.session.run(self.output_names, {self.input_names[0]: input_blob})

        input_height, input_width = self.input_size

        for idx, stride in enumerate(self._feat_stride_fpn):
            scores = outputs[idx]
            bbox_preds = outputs[idx + self.fmc] * stride
            kps_preds = outputs[idx + self.fmc * 2] * stride if self.use_kps else None

            height, width = input_height // stride, input_width // stride
            key = (height, width, stride)

            if key in self.center_cache:
                anchor_centers = self.center_cache[key]
            else:
                anchor_centers = np.stack(np.mgrid[:height, :width][::-1], axis=-1).astype(np.float32)
                anchor_centers = (anchor_centers * stride).reshape((-1, 2))
                if self._num_anchors > 1:
                    anchor_centers = np.stack([anchor_centers] * self._num_anchors, axis=1).reshape((-1, 2))
                if len(self.center_cache) < 100:
                    self.center_cache[key] = anchor_centers

            pos_inds = np.where(scores >= threshold)[0]
            if pos_inds.size == 0:
                continue

            bboxes = distance2bbox(anchor_centers, bbox_preds)
            pos_scores = scores[pos_inds]
            pos_bboxes = bboxes[pos_inds]
            scores_list.append(pos_scores)
            bboxes_list.append(pos_bboxes)

            if self.use_kps:
                kpss = distance2kps(anchor_centers, kps_preds)
                pos_kpss = kpss[pos_inds]
                kpss_list.append(pos_kpss)

        return scores_list, bboxes_list, kpss_list

    def detect(self, image: np.ndarray):
        scores_list, bboxes_list, kpss_list = self.forward(image, self.conf_thres)

        if not bboxes_list:
            return [], []

        scores = np.vstack(scores_list).ravel()
        order = scores.argsort()[::-1]

        bboxes = np.vstack(bboxes_list)
        kpss = np.vstack(kpss_list) if self.use_kps else None

        pre_det = np.hstack((bboxes, scores.reshape(-1, 1))).astype(np.float32, copy=False)
        pre_det = pre_det[order, :]
        keep = self.nms(pre_det, iou_thres=self.iou_thres)
        det = pre_det[keep, :]

        if kpss is not None:
            kpss = kpss[order, :]
            kpss = kpss[keep, :]

        return det, kpss

    def nms(self, dets, iou_thres):
        x1, y1, x2, y2, scores = dets[:, 0], dets[:, 1], dets[:, 2], dets[:, 3], dets[:, 4]
        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h
            ovr = inter / (areas[i] + areas[order[1:]] - inter)

            order = order[np.where(ovr <= iou_thres)[0] + 1]

        return keep

# RTSP Stream URL
RTSP_URL = "rtsp://admin:123456@192.168.1.205:554/avstream/channel=1/stream=0.sdp"

if __name__ == "__main__":
    detector = SCRFD(model_path="det_10g.onnx")
    cap = cv2.VideoCapture(RTSP_URL)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("RTSP stream lost, retrying...")
            break

        frame = cv2.resize(frame, (640, 640))
        boxes_list, kpss_list = detector.detect(frame)

import os

save_dir = "detected_faces"
os.makedirs(save_dir, exist_ok=True)  # Create directory if not exists

frame_count = 0  # To track frames

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("RTSP stream lost, retrying...")
        break

    frame = cv2.resize(frame, (640, 640))
    boxes_list, kpss_list = detector.detect(frame)

    for idx, box in enumerate(boxes_list):
        x1, y1, x2, y2, score = map(int, box)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"{score:.2f}", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            # Crop the face region
        cropped_face = frame[y1:y2, x1:x2]
            
            # Resize for embedding extraction (optional, depending on your embedding model)
        cropped_face = cv2.resize(cropped_face, (112, 112))

            # Save face with unique name
        face_filename = os.path.join(save_dir, f"face_{frame_count}_{idx}.jpg")
        cv2.imwrite(face_filename, cropped_face)

        if kpss_list is not None:
            for kps in kpss_list:
                for i in range(0, len(kps), 2):  # Iterate through keypoints (x, y)
                    x, y = int(kps[i]), int(kps[i + 1])
                    cv2.circle(frame, (x, y), 2, (0, 0, 255), -1)  # Red dots for keypoints

        cv2.imshow("Face Detection", frame)

        frame_count += 1  # Increment frame count

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

    #     for box in boxes_list:
    #         x1, y1, x2, y2, score = map(int, box)
    #         cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    #         cv2.putText(frame, f"{score:.2f}", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    #     if kpss_list is not None:
    #         for kps in kpss_list:
    #             for i in range(0, len(kps), 2):  # Iterate through keypoints (x, y)
    #                 x, y = int(kps[i]), int(kps[i + 1])
    #                 cv2.circle(frame, (x, y), 2, (0, 0, 255), -1)  # Red dots for keypoints

    #     cv2.imshow("Face Detection", frame)

    #     if cv2.waitKey(1) & 0xFF == ord("q"):
    #         break

    # cap.release()
    # cv2.destroyAllWindows()
