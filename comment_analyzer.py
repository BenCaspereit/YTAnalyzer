import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import json
from pathlib import Path
import emoji

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", DEVICE)

COMMENTS_FILE = Path("comments.json")
RESULTS_FILE = Path("results.json")

# Model Names
MODEL_SENTIMENT = "nlptown/bert-base-multilingual-uncased-sentiment"
MODEL_EMOTION   = "bhadresh-savani/bert-base-uncased-emotion"
MODEL_INTENTION = "mindpadi/intent_classifier"
MODEL_THEME     = "microsoft/deberta-v3-large"



candidate_labels = ["Gaming","Music","Movies","TV Shows","Education","Technology","DIY","Art",
                    "Sports","Comedy","Vlogs","Food","Beauty","Health","News","Animals",
                    "Science","Motivation","Travel","Unboxing"]

# Load tokenizers and models
tokenizer_sentiment = AutoTokenizer.from_pretrained(MODEL_SENTIMENT)
model_sentiment = AutoModelForSequenceClassification.from_pretrained(MODEL_SENTIMENT, use_safetensors=True).to(DEVICE)

tokenizer_emotion = AutoTokenizer.from_pretrained(MODEL_EMOTION)
model_emotion = AutoModelForSequenceClassification.from_pretrained(MODEL_EMOTION, use_safetensors=True).to(DEVICE)

tokenizer_intention = AutoTokenizer.from_pretrained(MODEL_INTENTION)
model_intention = AutoModelForSequenceClassification.from_pretrained(MODEL_INTENTION, use_safetensors=True).to(DEVICE)

tokenizer_theme = AutoTokenizer.from_pretrained(MODEL_THEME, use_fast=False)
model_theme = AutoModelForSequenceClassification.from_pretrained(MODEL_THEME, use_safetensors=True).to(DEVICE)

# Load comments
def load_comments():
    if COMMENTS_FILE.exists():
        try:
            with open(COMMENTS_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                return [entry["comment"] for entry in data if "comment" in entry]
        except json.JSONDecodeError:
            print(f"Warning: {COMMENTS_FILE} is empty or invalid. Start with empty list.")
            return []
    return []

# Load previous results
def load_results():
    if RESULTS_FILE.exists():
        try:
            with open(RESULTS_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except json.JSONDecodeError:
            return []
    return []

# Save results
def save_results(results):
    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"{len(results)} results saved in '{RESULTS_FILE}'")

# Predict sentiment
def predict_sentiment(batch_texts):
    inputs = tokenizer_sentiment(batch_texts, return_tensors="pt", padding=True, truncation=True).to(DEVICE)
    with torch.no_grad(), torch.amp.autocast(device_type='cuda', enabled=(DEVICE=="cuda")):
        logits = model_sentiment(**inputs).logits
        probs = torch.nn.functional.softmax(logits, dim=-1)
        labels = torch.argmax(probs, dim=1)
    return [model_sentiment.config.id2label[l.item()] for l in labels]

# Predict emotion
def predict_emotion(batch_texts):
    inputs = tokenizer_emotion(batch_texts, return_tensors="pt", padding=True, truncation=True).to(DEVICE)
    with torch.no_grad(), torch.amp.autocast(device_type='cuda', enabled=(DEVICE=="cuda")):
        logits = model_emotion(**inputs).logits
        probs = torch.nn.functional.softmax(logits, dim=-1)
        labels = torch.argmax(probs, dim=1)
    return [model_emotion.config.id2label[l.item()] for l in labels]

# Predict intention
def predict_intention(batch_texts):
    inputs = tokenizer_intention(batch_texts, return_tensors="pt", padding=True, truncation=True).to(DEVICE)
    with torch.no_grad(), torch.amp.autocast(device_type='cuda', enabled=(DEVICE=="cuda")):
        logits = model_intention(**inputs).logits
        probs = torch.nn.functional.softmax(logits, dim=-1)
        labels = torch.argmax(probs, dim=1)
    return [model_intention.config.id2label[l.item()] for l in labels]

# Predict theme (per comment)
def predict_theme(text):
    texts = [text]*len(candidate_labels)
    labels = candidate_labels
    inputs = tokenizer_theme(texts, labels, return_tensors="pt", padding=True, truncation=True, max_length=128).to(DEVICE)
    with torch.no_grad(), torch.amp.autocast(device_type='cuda', enabled=(DEVICE=="cuda")):
        logits = model_theme(**inputs).logits
        probs = torch.nn.functional.softmax(logits, dim=1)[:,0]  # entailment
    best_idx = torch.argmax(probs).item()
    return candidate_labels[best_idx]

# Main processing
def main():
    batch_size = 8  # moderat, damit GPU/RAM nicht überlastet
    comments = load_comments()
    results = load_results()
    if not comments:
        print("No comments found")
        return

    print(f"Processing {len(comments)} comments...")

    for i in range(0, len(comments), batch_size):
        batch = [emoji.demojize(c) for c in comments[i:i+batch_size]]

        sentiments = predict_sentiment(batch)
        emotions = predict_emotion(batch)
        intentions = predict_intention(batch)
        themes = [predict_theme(text) for text in batch]

        for j, text in enumerate(batch):
            results.append({
                "comment": text,
                "sentiment": sentiments[j],
                "emotion": emotions[j],
                "intention": intentions[j],
                "theme": themes[j]
            })

        save_results(results)

    print("Processing complete.")

if __name__ == "__main__":
    main()
