import torch
from models.sentiment import SentimentClassifier
from models.emotion import EmotionClassifier
from models.intention import IntentionClassifier
from models.theme import ThemeClassifier
from preprocessing import preprocess_batch

class CommentAnalyzer:
    def __init__(self, tasks: list[str] = None, device: str = None):
        # automatic device selection
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Analyzer initialized on device: {self.device}")

        self.tasks = tasks or ["sentiment", "emotion", "intention", "theme"]
        self.models = {}

        if "sentiment" in self.tasks:
            print("Loading Sentiment Model...")
            self.models["sentiment"] = SentimentClassifier(device=self.device)
            
        if "emotion" in self.tasks:
            print("Loading Emotion Model...")
            self.models["emotion"] = EmotionClassifier(device=self.device)
            
        if "intention" in self.tasks:
            print("Loading Intention Model...")
            self.models["intention"] = IntentionClassifier(device=self.device)
            
        if "theme" in self.tasks:
            print("Loading Theme Model...")
            self.models["theme"] = ThemeClassifier(device=self.device)


    def analyze_batch(self, comments: list[str]) -> list[dict]:
        if not comments:
            return []

        # 1. Preprocessing 
        cleaned_comments = preprocess_batch(comments)

        # 2. Inferenz result 
        results_by_task = {}
        for task_name, model in self.models.items():
            results_by_task[task_name] = model.predict(cleaned_comments)

        # 3. structure output
        output = []
        for i, raw_comment in enumerate(comments):
            res = {"comment": raw_comment}
            for task_name in self.tasks:
                res[task_name] = results_by_task[task_name][i]
            output.append(res)

        return output