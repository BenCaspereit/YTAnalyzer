import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

class ThemeClassifier:
    DEFAULT_MODEL = "valhalla/distilbart-mnli-12-1"
    DEFAULT_LABELS = [
        "Gaming", "Music", "Movies", "TV Shows", "Education", "Technology", "DIY", "Art",
        "Sports", "Comedy", "Vlogs", "Food", "Beauty", "Health", "News", "Animals",
        "Science", "Motivation", "Travel", "Unboxing"
    ]

    def __init__(self, model_name: str = None, candidate_labels: list[str] = None, device: str = "cpu"):
        self.device = device
        self.model_name = model_name or self.DEFAULT_MODEL
        self.candidate_labels = candidate_labels or self.DEFAULT_LABELS

        # Fast Tokenizer ist hier deutlich schneller
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name, 
            use_safetensors=True
        ).to(self.device)
        self.model.eval()

    def predict(self, batch_texts: list[str]) -> list[str]:
        """Echter Batch-Durchlauf für Zero-Shot Theme Classification."""
        if not batch_texts:
            return []

        results = []
        
        # Für jeden Kommentar im Batch vergleichen wir gegen alle candidate_labels
        for text in batch_texts:
            # 1. Wir erstellen Paare: [(text, label_1), (text, label_2), ...]
            pairs = [(text, label) for label in self.candidate_labels]
            
            # 2. Tokenisieren aller 20 Paare gleichzeitig
            inputs = self.tokenizer(
                pairs, 
                return_tensors = "pt", 
                padding = True, 
                truncation = True, 
                max_length = 256
            ).to(self.device)

            # 3. Inferenz für alle 20 Paare in einem einzigen Modell-Aufruf
            with torch.inference_mode(), torch.amp.autocast(device_type=self.device, enabled=(self.device == "cuda")):
                logits = self.model(**inputs).logits
                
                # Bei MNLI ist Index 2 (oder Index 0 je nach Modell) meist Entailment.
                # Bei Bart-MNLI steht Logit-Index 2 für 'entailment' (Zustimmung).
                # Wir nehmen die Logits der Entailment-Klasse für alle 20 Labels:
                entailment_logits = logits[:, 2] 
                
                best_idx = torch.argmax(entailment_logits).item()
                results.append(self.candidate_labels[best_idx])

        return results