"""
track_deepsort.py  -  Human tracking with occlusion handling
Uses your fine-tuned YOLOv11-seg model + DeepSORT
"""

import cv2
import numpy as np
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_PATH   = "runs/segment/train2/weights/best.pt"
VIDEO_IN     = "train_video.mp4"
#VIDEO_IN    = 0  # use webcam
VIDEO_OUT    = "tracked_deepsort_output.mp4"
#VIDEO_OUT  = "deepsort_output_webcam.mp4"
SHOW_LIVE    = True

PERSON_CLS   = 0
OCCLUDER_CLS = []

CONF_THRESH      = 0.40
OCC_IOU_THRESH   = 0.20
OCC_AREA_RATIO   = 0.35

MAX_AGE      = 50
N_INIT       = 2
MAX_IOU_DIST = 0.7
MAX_COS_DIST = 0.3
NN_BUDGET    = 100

PALETTE = [
    (255,  56,  56), (255, 157, 151), (255, 112,  31), (255, 178,  29),
    ( 72, 249,  10), (146, 204,  23), ( 26, 147,  52), (  0, 194, 255),
    ( 52,  69, 147), (100, 115, 255), (132,  56, 255), (203,  56, 255),
]

def color(tid):
    return PALETTE[int(tid) % len(PALETTE)]

def box_iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0
    ua = (ax2-ax1)*(ay2-ay1) + (bx2-bx1)*(by2-by1) - inter
    return inter / (ua + 1e-6)

def mask_overlap_ratio(person_mask, occluder_mask):
    if person_mask is None or occluder_mask is None:
        return 0.0
    person_area = person_mask.sum()
    if person_area == 0:
        return 0.0
    overlap = np.logical_and(person_mask, occluder_mask).sum()
    return float(overlap) / float(person_area)

def draw_dashed_rect(img, pt1, pt2, clr, thickness=1, dash=10):
    x1, y1 = pt1
    x2, y2 = pt2
    for sx, sy, ex, ey in [(x1,y1,x2,y1),(x2,y1,x2,y2),
                            (x2,y2,x1,y2),(x1,y2,x1,y1)]:
        dist = int(np.hypot(ex - sx, ey - sy))
        for d in range(0, dist, dash * 2):
            t0 = d / max(dist, 1)
            t1 = min((d + dash) / max(dist, 1), 1.0)
            p0 = (int(sx + t0*(ex-sx)), int(sy + t0*(ey-sy)))
            p1 = (int(sx + t1*(ex-sx)), int(sy + t1*(ey-sy)))
            cv2.line(img, p0, p1, clr, thickness)

def put_label(img, text, x, y, clr):
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), _ = cv2.getTextSize(text, font, 0.55, 1)
    cv2.rectangle(img, (x, y - th - 6), (x + tw + 4, y), clr, -1)
    cv2.putText(img, text, (x + 2, y - 3), font, 0.55, (255, 255, 255), 1)

class GhostMemory:
    def __init__(self):
        self.boxes     = {}
        self.last_seen = {}

    def update(self, tid, box, frame_idx, is_occluded):
        self.last_seen[tid] = frame_idx
        if not is_occluded:
            self.boxes[tid] = list(map(int, box))

    def active_ghosts(self, active_ids, frame_idx):
        return {
            tid: box for tid, box in self.boxes.items()
            if tid not in active_ids
            and (frame_idx - self.last_seen.get(tid, 0)) <= MAX_AGE
        }

def main():
    model = YOLO(MODEL_PATH)
    print("[INFO] Model classes:", model.names)

    tracker = DeepSort(
        max_age=MAX_AGE,
        n_init=N_INIT,
        max_iou_distance=MAX_IOU_DIST,
        max_cosine_distance=MAX_COS_DIST,
        nn_budget=NN_BUDGET,
    )
    memory = GhostMemory()

    cap = cv2.VideoCapture(VIDEO_IN)
    assert cap.isOpened(), f"Cannot open {VIDEO_IN}"
    #cap = cv2.VideoCapture(VIDEO_IN)
    #assert cap.isOpened(), f"Cannot open camera {VIDEO_IN}"
    #print(f"[INFO] Camera resolution: {int(cap.get(3))} x {int(cap.get(4))}")

    W       = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    FPS_SRC = cap.get(cv2.CAP_PROP_FPS) or 5.0
    #FPS_SRC = cap.get(cv2.CAP_PROP_FPS) or 30.0   # webcams default to 30fps

    writer = cv2.VideoWriter(
        VIDEO_OUT,
        cv2.VideoWriter_fourcc(*'mp4v'),
        FPS_SRC,
        (W, H)
    )

    frame_idx = 0
    prev_time = cv2.getTickCount()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, verbose=False)[0]

        person_dets   = []
        person_boxes  = []
        person_masks  = []
        occluder_info = []

        has_masks = results.masks is not None

        for i, box in enumerate(results.boxes):
            cls_id = int(box.cls[0])
            conf   = float(box.conf[0])
            if conf < CONF_THRESH:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(W, x2), min(H, y2)

            mask = None
            if has_masks and i < len(results.masks.data):
                raw  = results.masks.data[i].cpu().numpy()
                mask = cv2.resize(raw, (W, H)) > 0.5

            cls_name = model.names.get(cls_id, str(cls_id))

            if cls_id == PERSON_CLS:
                w  = x2 - x1
                h  = y2 - y1
                cx = x1 + w // 2
                cy = y1 + h // 2
                person_dets.append(([cx, cy, w, h], conf, "person"))
                person_boxes.append([x1, y1, x2, y2])
                person_masks.append(mask)
            else:
                if not OCCLUDER_CLS or cls_id in OCCLUDER_CLS:
                    occluder_info.append(([x1, y1, x2, y2], mask, cls_name))

        tracks = tracker.update_tracks(person_dets, frame=frame)

        def is_occluded(tbox):
            for obox, omask, oname in occluder_info:
                if box_iou(tbox, obox) > OCC_IOU_THRESH:
                    return True, oname
                if omask is not None:
                    best_pmask, best_iou = None, 0
                    for pbox, pmask in zip(person_boxes, person_masks):
                        piou = box_iou(tbox, pbox)
                        if piou > best_iou:
                            best_iou   = piou
                            best_pmask = pmask
                    if mask_overlap_ratio(best_pmask, omask) > OCC_AREA_RATIO:
                        return True, oname
            return False, None

        annotated = frame.copy()

        overlay = annotated.copy()
        for obox, omask, oname in occluder_info:
            ox1, oy1, ox2, oy2 = obox
            if omask is not None:
                overlay[omask] = (
                    overlay[omask] * 0.5 + np.array([0, 140, 255]) * 0.5
                ).astype(np.uint8)
            else:
                cv2.rectangle(overlay, (ox1, oy1), (ox2, oy2), (0, 140, 255), -1)
        cv2.addWeighted(overlay, 0.25, annotated, 0.75, 0, annotated)

        for obox, omask, oname in occluder_info:
            ox1, oy1, ox2, oy2 = obox
            cv2.rectangle(annotated, (ox1, oy1), (ox2, oy2), (0, 140, 255), 2)
            put_label(annotated, oname, ox1, oy1, (0, 100, 200))

        active_ids = set()
        for track in tracks:
            if not track.is_confirmed():
                continue

            tid  = track.track_id
            active_ids.add(tid)
            l, t, r, b = map(int, track.to_ltrb())
            tbox = [l, t, r, b]
            c    = color(tid)

            occ, occ_name = is_occluded(tbox)
            memory.update(tid, tbox, frame_idx, occ)

            if occ:
                draw_dashed_rect(annotated, (l, t), (r, b), c, 2)
                label = f"ID:{tid} OCCLUDED"
                if occ_name:
                    label += f" by {occ_name}"
            else:
                cv2.rectangle(annotated, (l, t), (r, b), c, 2)
                label = f"ID:{tid}"

            put_label(annotated, label, l, t, c)

        for tid, gbox in memory.active_ghosts(active_ids, frame_idx).items():
            c = color(tid)
            gx1, gy1, gx2, gy2 = gbox
            draw_dashed_rect(annotated, (gx1, gy1), (gx2, gy2), c, 1)
            put_label(annotated, f"ID:{tid} HIDDEN", gx1, gy1, (80, 80, 80))

        now  = cv2.getTickCount()
        fps  = cv2.getTickFrequency() / (now - prev_time + 1)
        prev_time = now

        cv2.rectangle(annotated, (0, 0), (280, 60), (20, 20, 20), -1)
        cv2.putText(annotated, f"FPS: {fps:.1f}",
                    (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 180), 1)
        cv2.putText(annotated, f"Tracks: {len(active_ids)}  Frame: {frame_idx}",
                    (8, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

        writer.write(annotated)
        frame_idx += 1

        if SHOW_LIVE:
            cv2.imshow("DeepSORT | Occlusion Tracking", annotated)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    writer.release()
    cv2.destroyAllWindows()
    print(f"\n[DONE] {frame_idx} frames processed -> {VIDEO_OUT}")

if __name__ == "__main__":
    main()