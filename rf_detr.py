from inference_models import AutoModel


class RfDetr:
    def __init__(self, model_path):
        self.model = self.__load_model(model_path)
        # one inner list for each frame
        self.results: list[list[tuple[int, int, int, int]]] = []
        self.result_centers: list[list[tuple[int, int]]] = []


    def __load_model(self, model_path: str):
        return AutoModel.from_pretrained(model_path)
        # "./weights/models-cache/v2-shells-workspace-5rddh-player-detection-rf-detr--c5e2cdfdfd20e4517f060ad947ef8c91/5362b72bfb9f01d2e0b8cba2048d932c"

    # there can be [0, infinite] predictions for each frame
    # each prediction is stored as a tuple of bounding box coordinates (x1, y1, x2, y2)
    def predict(self, image) -> None:
        try:
            prediction = self.model.infer(image)
        except Exception as e:
            print(f'Prediction error: {e}')
            return

        class_ids = prediction[0].class_id.tolist()
        indices = [i for i, val in enumerate(class_ids) if val == 0]

        self.results.append([])
        for index in indices:
            x1, y1, x2, y2 = prediction[0].xyxy.tolist()[index]
            self.results[-1].append((x1, y1, x2, y2))

    def create_result_centers(self) -> None:
        for result in self.results:
            self.result_centers.append([])
            for prediction in result:
                x1, y1, x2, y2 = prediction
                self.result_centers[-1].append(((x1 + x2) // 2, (y1 + y2)  // 2))



