import os

import cv2
from inference_models import AutoModel
from kalman_filter import KalmanFilter

bb_color = (0, 0, 255)
bb_thickness = 2
point_color = (0, 255, 0)
point_thickness = 5

def load_model():
    model = AutoModel.from_pretrained(
        "./weights/models-cache/v2-shells-workspace-5rddh-player-detection-rf-detr--c5e2cdfdfd20e4517f060ad947ef8c91/5362b72bfb9f01d2e0b8cba2048d932c")
    return model

def predict_on_video(input_video: str, output_video: str) -> tuple[list[list[tuple[int, int]]], list[list[tuple[int, int, int, int]]]]:
    cap = cv2.VideoCapture(input_video)
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    if original_fps == 0:
        raise ValueError(f"Could not open video file")

    #width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    #height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    #fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    #out = cv2.VideoWriter(output_video, fourcc, original_fps, (width, height))

    prediction_results_centers: list[list[tuple[int, int]]] = []
    prediction_results: list[list[tuple[int, int, int, int]]] = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        prediction = model.infer(frame)
        try:
            class_ids = prediction[0].class_id.tolist()
            indices = [i for i, val in enumerate(class_ids) if val == 0]

            prediction_results_centers.append([])
            prediction_results.append([])
            for index in indices:
                x1, y1, x2, y2 = prediction[0].xyxy.tolist()[index]
                #cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
                prediction_results_centers[-1].append(((x1 + x2) // 2, (y1 + y2)  // 2))
                prediction_results[-1].append((x1, y1, x2, y2))

            #out.write(frame)


        except Exception as e:
            print(f"Caught an exception: {e}")
            print(f"Exception type: {type(e).__name__}")

    cap.release()
    #out.release()
    return prediction_results_centers, prediction_results
if __name__ == "__main__":
    model = AutoModel.from_pretrained(
        "./weights/models-cache/v2-shells-workspace-5rddh-player-detection-rf-detr--c5e2cdfdfd20e4517f060ad947ef8c91/5362b72bfb9f01d2e0b8cba2048d932c")

    for folder in os.listdir("Videos"):
        if folder.startswith("."):
            continue
        for file in os.listdir("Videos/" + folder):
            if file.endswith(".mp4"):
                input_video = "Videos/" + folder + "/" + file
                output_video = "results/" + folder + "/" + file
                print(f'Predicting on {input_video} in {folder}')
                center_predictions, bb_predictions = predict_on_video(input_video, output_video)
                print("Done!")
