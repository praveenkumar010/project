from ultralytics import YOLO
import cv2
import torch
from collections import defaultdict, deque

def process_video(video_path):

    device = 0 if torch.cuda.is_available() else "cpu"

    model = YOLO("model/helmet best.pt")

    cap = cv2.VideoCapture(video_path)

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    output_path = "output.mp4"

    out = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (w,h)
    )

    history = defaultdict(lambda: deque(maxlen=15))
    final_label = {}
    violation_table = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model.track(frame, persist=True, conf=0.4, device=device)
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
                x1,y1,x2,y2 = map(int, box.xyxy[0])

                # ignore non helmet classes
                if label in ["bike","driver","passenger"]:
                    continue

                if "without" in label:
                    current = "NO_HELMET"
                else:
                    current = "HELMET"

                history[track_id].append(current)

                if track_id not in final_label and len(history[track_id]) >= 10:
                    if history[track_id].count("NO_HELMET") > history[track_id].count("HELMET"):
                        final_label[track_id] = "NO_HELMET"
                        violation_table.append({
                            "Vehicle ID": track_id,
                            "Violation": "No Helmet"
                        })
                    else:
                        final_label[track_id] = "HELMET"

                if track_id not in final_label:
                    continue

                decision = final_label[track_id]

                if decision == "NO_HELMET":
                    color = (0,0,255)
                    text = f"VIOLATION ID {track_id}"
                else:
                    color = (0,255,0)
                    text = f"HELMET ID {track_id}"

                cv2.rectangle(annotated,(x1,y1),(x2,y2),color,3)
                cv2.putText(annotated,text,(x1,y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX,0.8,color,2)

        out.write(annotated)

    cap.release()
    out.release()

    return output_path, violation_table
