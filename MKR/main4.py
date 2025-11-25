from ultralytics import YOLO
import cv2
from pathlib import Path
import shutil

MODEL_FILE = "/kaggle/working/runs/detect/train/weights/best.pt"
INPUT_VIDEO_PATH = "/kaggle/input/mkr-video-nike/PlayWithSkill_PhantomGT_NikeFootball.mp4"
OUTPUT_FOLDER = Path("/kaggle/working/processed-videos")
OUTPUT_FOLDER.mkdir(exist_ok=True)

OUTPUT_PATH_1 = OUTPUT_FOLDER / "auto_processed.mp4"
OUTPUT_PATH_2 = OUTPUT_FOLDER / "manual_processed.mp4"
OUTPUT_PATH_3 = OUTPUT_FOLDER / "stats_processed.mp4"

CONFIDENCE_THRESH = 0.25
IOU_THRESH = 0.45
LINE_THICKNESS = 2
FONT_SCALE_FACTOR = 0.5
DISPLAY_LABELS = True
DISPLAY_CONFIDENCE = True


def run_auto_processing():
    print("=" * 50)
    print("AUTO PROCESSING METHOD")
    print("=" * 50)

    detector = YOLO(MODEL_FILE)
    preds = detector.predict(
        source=INPUT_VIDEO_PATH,
        save=True,
        conf=CONFIDENCE_THRESH,
        iou=IOU_THRESH,
        show_labels=DISPLAY_LABELS,
        show_conf=DISPLAY_CONFIDENCE,
        line_width=LINE_THICKNESS,
        project=str(OUTPUT_FOLDER),
        name="temp_auto",
        verbose=True
    )

    temp_vid = OUTPUT_FOLDER / "temp_auto" / Path(INPUT_VIDEO_PATH).name
    if temp_vid.exists():
        shutil.move(str(temp_vid), str(OUTPUT_PATH_1))
        shutil.rmtree(OUTPUT_FOLDER / "temp_auto")

    print(f"Saved auto-processed video at: {OUTPUT_PATH_1}\n")
    return str(OUTPUT_PATH_1)


def run_manual_processing():
    print("=" * 50)
    print("MANUAL FRAME-BY-FRAME PROCESSING")
    print("=" * 50)

    detector = YOLO(MODEL_FILE)
    video_capture = cv2.VideoCapture(INPUT_VIDEO_PATH)

    if not video_capture.isOpened():
        raise RuntimeError(f"Unable to open video file: {INPUT_VIDEO_PATH}")

    fps_val = int(video_capture.get(cv2.CAP_PROP_FPS))
    frame_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_total = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))

    video_codec = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(str(OUTPUT_PATH_2), video_codec, fps_val, (frame_width, frame_height))

    processed_frames = 0
    detection_frames = 0

    print(f"Video info: {frame_width}x{frame_height}, {fps_val} FPS, total frames: {frame_total}\n")

    while video_capture.isOpened():
        ret_val, frame = video_capture.read()
        if not ret_val:
            break

        processed_frames += 1

        predictions = detector.predict(frame, conf=CONFIDENCE_THRESH, iou=IOU_THRESH, verbose=False)

        frame_annotated = predictions[0].plot(
            line_width=LINE_THICKNESS,
            font_size=FONT_SCALE_FACTOR,
            labels=DISPLAY_LABELS,
            conf=DISPLAY_CONFIDENCE
        )

        if len(predictions[0].boxes) > 0:
            detection_frames += 1

        video_writer.write(frame_annotated)

        if processed_frames % 30 == 0:
            progress_pct = (processed_frames / frame_total) * 100
            print(f"  Processed frames: {processed_frames}/{frame_total} ({progress_pct:.1f}%)")

    video_capture.release()
    video_writer.release()

    print("\n" + "=" * 50)
    print("MANUAL PROCESSING DONE")
    print("=" * 50)
    print(f"Total frames processed: {processed_frames}")
    print(f"Frames with detections: {detection_frames}")
    print(f"Saved manual processed video to: {OUTPUT_PATH_2}\n")

    return str(OUTPUT_PATH_2)


def run_statistical_processing():
    print("=" * 50)
    print("PROCESSING WITH STATISTICS")
    print("=" * 50)

    detector = YOLO(MODEL_FILE)
    cap = cv2.VideoCapture(INPUT_VIDEO_PATH)

    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {INPUT_VIDEO_PATH}")

    fps_val = int(cap.get(cv2.CAP_PROP_FPS))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc_code = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(str(OUTPUT_PATH_3), fourcc_code, fps_val, (frame_width, frame_height))

    class_detection_counts = {}
    frame_index = 0

    print(f"Starting stats processing on video: {frame_width}x{frame_height} at {fps_val} FPS\n")

    while cap.isOpened():
        ret_flag, frame = cap.read()
        if not ret_flag:
            break

        frame_index += 1

        prediction = detector.predict(frame, conf=CONFIDENCE_THRESH, verbose=False)
        annotated = prediction[0].plot(line_width=LINE_THICKNESS)

        for box in prediction[0].boxes:
            cls_id = int(box.cls[0])
            cls_name = detector.names[cls_id]
            class_detection_counts[cls_name] = class_detection_counts.get(cls_name, 0) + 1

        writer.write(annotated)

        if frame_index % 50 == 0:
            print(f"  Processed frame {frame_index}/{total_frames}")

    cap.release()
    writer.release()

    print("\n" + "=" * 50)
    print("DETECTION SUMMARY:")
    print("=" * 50)
    for cls_name, cnt in sorted(class_detection_counts.items(), key=lambda item: item[1], reverse=True):
        print(f"  {cls_name}: {cnt} detections")

    print(f"\nSaved stats processed video at: {OUTPUT_PATH_3}\n")

    return str(OUTPUT_PATH_3)


if __name__ == "__main__":
    if not Path(MODEL_FILE).exists():
        print(f"Model file missing: {MODEL_FILE}")
        exit(1)

    if not Path(INPUT_VIDEO_PATH).exists():
        print(f"Input video missing: {INPUT_VIDEO_PATH}")
        exit(1)

    print("Starting all video processing methods...\n")

    video_out_1 = run_auto_processing()
    video_out_2 = run_manual_processing()
    video_out_3 = run_statistical_processing()

    print("\n" + "=" * 50)
    print("ALL VIDEO PROCESSING COMPLETE!")
    print("=" * 50)
    print(f"\nVideos saved in directory: {OUTPUT_FOLDER}")
    print(f"1. {video_out_1}")
    print(f"2. {video_out_2}")
    print(f"3. {video_out_3}")