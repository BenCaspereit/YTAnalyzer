from models.base import BaseClassifier

class IntentionClassifier(BaseClassifier):
    DEFAULT_MODEL = "conceptnetUk/intent-classifier"

    def __init__(self, model_name: str = None, device: str = "cpu"):
        selected_model = model_name or self.DEFAULT_MODEL
        super().__init__(model_name=selected_model, device=device)