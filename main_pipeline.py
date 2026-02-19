from ultralytics import YOLO
import cv2
import torch
from collections import defaultdict, deque
import os

# ============================================
# LOAD MODEL
# ============================================
device = "cuda" if torch.cuda.is_available() else "cpu"
model = YOLO("helmet best.pt")
model.to(device)

# ============================================
# MAIN FUNCTION
# ============================================
def process_video(video_path):

    cap = cv2.VideoCapture(video_path)

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    output_path = "processed.mp4"

    out = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (w, h)
    )

    # 🧠 memory for stable prediction
    history = defaultdict(lambda: deque(maxlen=15))
    final_label = {}
    violation_table = []

    frame_no = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_no += 1

        results = model.track(
            frame,
            persist=True,
            conf=0.35,
            device=device
        )

        annotated = frame.copy()

        if results[0].boxes is not None:
            boxes = results[0].boxes
            ids = boxes.id if boxes.id is not None else [None]*len(boxes)

            for box, track_id in zip(boxes, ids):

                if track_id is None:
                    continue

                track_id = int(track_id)

                cls = int(box.cls[0])
                label = model.names[cls]

                # ignore non helmet classes
                if label in ["bike","driver","passenger"]:
                    continue

                # convert labels
                if "without" in label:
                    current = "No Helmet"
                else:
                    current = "Helmet"

                history[track_id].append(current)

                # majority voting after 10 frames
                if track_id not in final_label and len(history[track_id]) >= 10:
                    helmet_votes = history[track_id].count("Helmet")
                    nohelmet_votes = history[track_id].count("No Helmet")

                    if nohelmet_votes > helmet_votes:
                        final_label[track_id] = "No Helmet"
                        violation_table.append([track_id, "Helmet Violation"])
                    else:
                        final_label[track_id] = "Helmet"

                if track_id not in final_label:
                    continue

                x1,y1,x2,y2 = map(int, box.xyxy[0])

                decision = final_label[track_id]

                if decision == "No Helmet":
                    color = (0,0,255)
                    text = f"ID {track_id} : NO HELMET"
                else:
                    color = (0,255,0)
                    text = f"ID {track_id} : HELMET"

                cv2.rectangle(annotated,(x1,y1),(x2,y2),color,3)
                cv2.putText(annotated,text,(x1,y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX,0.8,color,2)

        out.write(annotated)

    cap.release()
    out.release()

    # 🎬 Convert to web playable video (IMPORTANT)
    os.system("ffmpeg -y -i processed.mp4 -vcodec libx264 output_web.mp4")

    return "output_web.mp4", violation_table
