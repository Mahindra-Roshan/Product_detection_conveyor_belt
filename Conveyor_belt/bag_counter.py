"""
Smart Bag Counting & Classification PoC.

Pipeline: video in -> classical CV detection (Otsu-style threshold + ROI mask +
connected components) -> centroid tracking -> virtual tripwire line-crossing
counting -> size-bucket + product classification -> annotated frames out.

Local-file, single-script PoC. No server, no DB, no live camera ingestion.
Can be run standalone (writes output_annotated.mp4) or imported and driven
frame-by-frame (see app.py) for a live UI.
"""

import time

import cv2
import numpy as np

VIDEO_IN = "make_with_different_bag_sizes.mp4"
VIDEO_OUT = "output_annotated.mp4"

# ROI margin, as a fraction of frame size, applied along the axis PERPENDICULAR
# to belt travel (e.g. for vertical flow this trims left/right rails; for
# horizontal flow it trims top/bottom). Trims background/rails outside the belt.
ROI_MARGIN_CROSS_AXIS = 0.18

# Tripwire position as a fraction of the travel axis (mid-frame by default).
TRIPWIRE_FRAC = 0.5

MIN_CONTOUR_AREA_FRAC = 0.008  # min blob area as a fraction of ROI area, filters noise
THRESH_VALUE = 140  # binary threshold between dark belt and bright bag gray levels

# Size buckets by bounding-box area as a fraction of ROI area (placeholder,
# pending real-world camera calibration / homography).
SIZE_BUCKETS_FRAC = [
    (0.0, 0.10, "10kg"),
    (0.10, 0.16, "20kg"),
    (0.16, 1.0, "50kg"),
]

# Product classification placeholder: no color/texture model to distinguish
# flour vs semolina sacks (they look alike), so we use bag size as a stand-in
# signal per user direction: >20kg -> Flour, else -> Semolina.
def classify_product(size_label):
    return "Flour" if size_label == "50kg" else "Semolina"


# Detection runs on a downscaled copy of each frame (bags are large, blocky
# shapes, so this loses no signal that matters for tripwire counting or size
# classification) — measured ~6.6x faster than full-res with identical counts
# on the sample clips. Overlays are still drawn on the original full-res frame.
DETECTION_SCALE = 0.5


MAX_DISAPPEARED = 8  # frames a track can go undetected before being dropped
MAX_MATCH_DIST_FRAC = 0.06  # nearest-neighbor association threshold, fraction of frame diagonal


class Track:
    def __init__(self, track_id, detection):
        self.id = track_id
        self.centroid = detection["centroid"]
        self.prev_centroid = detection["centroid"]
        self.bbox = detection["bbox"]
        self.area = detection["area"]
        self.disappeared = 0
        self.counted = False

    def update(self, detection):
        self.prev_centroid = self.centroid
        self.centroid = detection["centroid"]
        self.bbox = detection["bbox"]
        self.area = detection["area"]
        self.disappeared = 0


class CentroidTracker:
    def __init__(self, max_match_dist):
        self.tracks = {}
        self._next_id = 1
        self.max_match_dist = max_match_dist

    def _new_track(self, detection):
        t = Track(self._next_id, detection)
        self._next_id += 1
        self.tracks[t.id] = t
        return t

    def update(self, detections):
        if not self.tracks:
            for det in detections:
                self._new_track(det)
            return self.tracks

        track_ids = list(self.tracks.keys())
        track_centroids = np.array([self.tracks[tid].centroid for tid in track_ids])

        if detections:
            det_centroids = np.array([d["centroid"] for d in detections])
            dist_matrix = np.linalg.norm(
                track_centroids[:, None, :] - det_centroids[None, :, :], axis=2
            )

            matched_tracks = set()
            matched_dets = set()
            flat = [
                (dist_matrix[i, j], i, j)
                for i in range(dist_matrix.shape[0])
                for j in range(dist_matrix.shape[1])
            ]
            flat.sort(key=lambda x: x[0])
            pairs = []
            for dist, i, j in flat:
                if i in matched_tracks or j in matched_dets:
                    continue
                if dist > self.max_match_dist:
                    continue
                pairs.append((i, j))
                matched_tracks.add(i)
                matched_dets.add(j)

            for i, j in pairs:
                self.tracks[track_ids[i]].update(detections[j])

            for j, det in enumerate(detections):
                if j not in matched_dets:
                    self._new_track(det)

            for i, tid in enumerate(track_ids):
                if i not in matched_tracks:
                    self.tracks[tid].disappeared += 1
        else:
            for tid in track_ids:
                self.tracks[tid].disappeared += 1

        self.tracks = {
            tid: t for tid, t in self.tracks.items() if t.disappeared <= MAX_DISAPPEARED
        }
        return self.tracks


def crossed_tripwire(prev_pos, curr_pos, line_pos):
    return prev_pos < line_pos <= curr_pos or prev_pos > line_pos >= curr_pos


class BagCounterPipeline:
    """Stateful pipeline: construct once per video, call process_frame() per frame.

    direction: "vertical" (bags travel top<->bottom, tripwire is horizontal)
               or "horizontal" (bags travel left<->right, tripwire is vertical).
    """

    def __init__(self, frame_width, frame_height, direction="vertical"):
        if direction not in ("vertical", "horizontal"):
            raise ValueError(f"Unsupported direction: {direction}")
        self.direction = direction
        self.w = frame_width
        self.h = frame_height

        if direction == "vertical":
            # Travel axis is Y; cross-axis (trimmed) is X.
            x0 = int(frame_width * ROI_MARGIN_CROSS_AXIS)
            x1 = int(frame_width * (1 - ROI_MARGIN_CROSS_AXIS))
            y0, y1 = 0, frame_height
            self.tripwire_pos = int(frame_height * TRIPWIRE_FRAC)
            self.axis_index = 1  # centroid[1] = y
        else:
            # Travel axis is X; cross-axis (trimmed) is Y.
            x0, x1 = 0, frame_width
            y0 = int(frame_height * ROI_MARGIN_CROSS_AXIS)
            y1 = int(frame_height * (1 - ROI_MARGIN_CROSS_AXIS))
            self.tripwire_pos = int(frame_width * TRIPWIRE_FRAC)
            self.axis_index = 0  # centroid[0] = x

        self.roi_poly = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=np.int32)
        roi_area = (x1 - x0) * (y1 - y0)

        # Detection runs on a downscaled frame (see DETECTION_SCALE); build a
        # matching downscaled ROI mask so masking happens in that space.
        self.detect_w = max(1, int(frame_width * DETECTION_SCALE))
        self.detect_h = max(1, int(frame_height * DETECTION_SCALE))
        small_roi_poly = np.array(
            [[int(px * DETECTION_SCALE), int(py * DETECTION_SCALE)] for px, py in self.roi_poly],
            dtype=np.int32,
        )
        self.roi_mask = np.zeros((self.detect_h, self.detect_w), dtype=np.uint8)
        cv2.fillPoly(self.roi_mask, [small_roi_poly], 255)

        self.min_contour_area = roi_area * MIN_CONTOUR_AREA_FRAC
        self.size_buckets = [
            (lo * roi_area, hi * roi_area, label) for lo, hi, label in SIZE_BUCKETS_FRAC
        ]

        diag = (frame_width ** 2 + frame_height ** 2) ** 0.5
        self.tracker = CentroidTracker(max_match_dist=diag * MAX_MATCH_DIST_FRAC)

        self.total_count = 0
        self.bucket_counts = {label: 0 for _, _, label in SIZE_BUCKETS_FRAC}
        self.product_counts = {"Flour": 0, "Semolina": 0}
        self.events = []  # [{frame, time, track_id, size_label, product}]
        self.frame_idx = 0
        self.start_time = time.time()

    def classify_size(self, area):
        for lo, hi, label in self.size_buckets:
            if lo <= area < hi:
                return label
        return "unknown"

    def _detect(self, frame):
        small = cv2.resize(frame, (self.detect_w, self.detect_h), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        masked = cv2.bitwise_and(gray, gray, mask=self.roi_mask)

        _, thresh = cv2.threshold(masked, THRESH_VALUE, 255, cv2.THRESH_BINARY)

        kernel = np.ones((9, 9), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)

        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(thresh)

        # Scale detections back up to full-frame coordinates for drawing/counting.
        inv_scale = 1.0 / DETECTION_SCALE
        detections = []
        for i in range(1, num_labels):  # skip background label 0
            area = stats[i, cv2.CC_STAT_AREA]
            if area < self.min_contour_area:
                continue
            x = int(stats[i, cv2.CC_STAT_LEFT] * inv_scale)
            y = int(stats[i, cv2.CC_STAT_TOP] * inv_scale)
            w = int(stats[i, cv2.CC_STAT_WIDTH] * inv_scale)
            h = int(stats[i, cv2.CC_STAT_HEIGHT] * inv_scale)
            cx, cy = centroids[i]
            cx, cy = cx * inv_scale, cy * inv_scale
            detections.append({"bbox": (x, y, w, h), "centroid": (cx, cy), "area": area * inv_scale * inv_scale})
        return detections

    def process_frame(self, frame):
        """Runs detection+tracking+counting on one frame, draws overlays in place,
        and returns a list of newly-crossed events (empty if none this frame)."""
        self.frame_idx += 1
        detections = self._detect(frame)
        tracks = self.tracker.update(detections)

        cv2.polylines(frame, [self.roi_poly], True, (255, 200, 0), 1)
        if self.direction == "vertical":
            cv2.line(frame, (0, self.tripwire_pos), (self.w, self.tripwire_pos), (0, 0, 255), 2)
        else:
            cv2.line(frame, (self.tripwire_pos, 0), (self.tripwire_pos, self.h), (0, 0, 255), 2)

        newly_counted = []
        for tid, t in tracks.items():
            if t.disappeared > 0:
                continue

            x, y, bw, bh = t.bbox
            size_label = self.classify_size(t.area)
            product = classify_product(size_label)
            cv2.rectangle(frame, (x, y), (x + bw, y + bh), (0, 255, 0), 2)
            cv2.putText(
                frame, f"ID {t.id} {product} {size_label}", (x, max(0, y - 8)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2,
            )

            prev_pos = t.prev_centroid[self.axis_index]
            curr_pos = t.centroid[self.axis_index]
            if not t.counted and crossed_tripwire(prev_pos, curr_pos, self.tripwire_pos):
                t.counted = True
                self.total_count += 1
                self.bucket_counts[size_label] = self.bucket_counts.get(size_label, 0) + 1
                self.product_counts[product] = self.product_counts.get(product, 0) + 1
                event = {
                    "frame": self.frame_idx,
                    "elapsed_sec": round(time.time() - self.start_time, 1),
                    "track_id": t.id,
                    "size_label": size_label,
                    "product": product,
                }
                self.events.append(event)
                newly_counted.append(event)

        cv2.putText(
            frame, f"Count: {self.total_count}", (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 255), 3,
        )
        y_off = 80
        for label, cnt in self.bucket_counts.items():
            cv2.putText(
                frame, f"{label}: {cnt}", (20, y_off),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2,
            )
            y_off += 30

        return newly_counted


def main():
    cap = cv2.VideoCapture(VIDEO_IN)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open {VIDEO_IN}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 24
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(VIDEO_OUT, fourcc, fps, (w, h))

    pipeline = BagCounterPipeline(w, h, direction="vertical")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        pipeline.process_frame(frame)
        writer.write(frame)

    cap.release()
    writer.release()

    print(f"Processed {pipeline.frame_idx} frames")
    print(f"Total bag count: {pipeline.total_count}")
    for label, cnt in pipeline.bucket_counts.items():
        print(f"  {label}: {cnt}")
    print(f"Product mix: {pipeline.product_counts}")
    print(f"Annotated video written to {VIDEO_OUT}")


if __name__ == "__main__":
    main()
