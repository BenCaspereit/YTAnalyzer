from models.base import BaseClassifier

class EmotionClassifier(BaseClassifier):
    DEFAULT_MODEL = "bhadresh-savani/distilbert-base-uncased-emotion"

    def __init__(self, model_name: str = None, device: str = "cpu"):
        selected_model = model_name or self.DEFAULT_MODEL
        super().__init__(model_name=selected_model, device=device)