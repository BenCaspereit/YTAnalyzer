import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

class BaseClassifier:
    def __init__(self, model_name: str, device: str = "cpu"):
        self.device = device
        self.model_name = model_name
        
        # Tokenizer und Modell laden
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name, 
            use_safetensors=True
        ).to(self.device)
        
        # Performance-Optimierung 1: Modell in den Evaluierungsmodus versetzen
        self.model.eval()

    def predict(self, batch_texts: list[str]) -> list[str]:
        """Führt Batch-Inferenz für eine Liste von Texten aus."""
        if not batch_texts:
            return []

        inputs = self.tokenizer(
            batch_texts, 
            return_tensors="pt", 
            padding=True, 
            truncation=True,
            return_token_type_ids=False  # Verhindert den TypeError bei DistilBERT
        ).to(self.device)

        # Performance-Optimierung 2: inference_mode statt no_grad
        with torch.inference_mode(), torch.amp.autocast(device_type=self.device, enabled=(self.device == "cuda")):
            logits = self.model(**inputs).logits
            
            # Performance-Optimierung 3: Kein Softmax nötig!
            # Der höchste Logit-Wert hat auch immer die höchste Wahrscheinlichkeit.
            labels = torch.argmax(logits, dim=-1)

        # Labels in Text-Klassen umwandeln
        id2label = self.model.config.id2label
        return [id2label[l.item()] for l in labels]