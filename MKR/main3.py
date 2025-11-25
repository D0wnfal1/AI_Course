from ultralytics import YOLO
import gc
import matplotlib.pyplot as plt

gc.collect()

model_instance = YOLO("yolo11n.pt")

model_instance.train(
    data="/kaggle/working/dataset-augmentation/data.yaml",
    epochs=20,
    imgsz=288,
    batch=2,
    workers=1,
    pretrained=True,
    optimizer="AdamW"
)

validation_output = model_instance.val()

precision_recall = validation_output.curves_results[0]
pr_recall = precision_recall[0].ravel()
pr_precision = precision_recall[1].ravel()
plt.figure(figsize=(6,6))
plt.plot(pr_recall, pr_precision, label="Precision-Recall")
plt.xlabel("True Positive Rate")
plt.ylabel("Positive Predictive Value")
plt.title("Curve: Precision vs Recall")
plt.grid(True)
plt.legend()
plt.savefig("precision_recall_curve.png")
plt.show()

f1_scores = validation_output.curves_results[1]
f1_confidence = f1_scores[0].ravel()
f1_values = f1_scores[1].ravel()
plt.figure(figsize=(6,4))
plt.plot(f1_confidence, f1_values, label="F1-score over Confidence")
plt.xlabel("Detection Confidence")
plt.ylabel("F1 Metric")
plt.title("F1 Metric by Confidence Threshold")
plt.grid(True)
plt.legend()
plt.savefig("f1Curve.png")
plt.show()

map50_scores = validation_output.curves_results[2]
map50_confidence = map50_scores[0].ravel()
map50_values = map50_scores[1].ravel()
plt.figure(figsize=(6,4))
plt.plot(map50_confidence, map50_values, label="mAP@50 vs Conf.")
plt.xlabel("Confidence Level")
plt.ylabel("Mean Average Precision (50%)")
plt.title("mAP@50 Across Confidence Levels")
plt.grid(True)
plt.legend()
plt.savefig("map50Curve.png")
plt.show()