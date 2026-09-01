from models.base import BaseClassifier

class SentimentClassifier(BaseClassifier):
    DEFAULT_MODEL = "nlptown/bert-base-multilingual-uncased-sentiment"

    def __init__(self, model_name: str = None, device: str = "cpu"):
        selected_model = model_name or self.DEFAULT_MODEL
        super().__init__(model_name=selected_model, device=device)