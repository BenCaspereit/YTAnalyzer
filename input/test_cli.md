# Test CLI

## This md just contains some commands for the cli to test the functionality

### Reads input/comments.json, runs all 4 AI models (sentiment, emotion, intention, theme), and saves the results in /results.

```
python cli.py analyze-file --file comments.json --output results/comments_analyzed.parquet
```

---

### Loads only the Sentiment and Emotion models via lazy loading for a faster run.

```
python cli.py analyze-file --file comments.json -t sentiment -t emotion
```

---

### Increases the batch size to 64 for higher GPU utilization and faster speed.

```
python cli.py analyze-file --file comments.json --batch-size 64
```

---

### Fetches up to 500 English comments directly from a YouTube video and runs the AI pipeline on them.

```
python cli.py analyze-video --video-id dQw4w9WgXcQ --max-comments 500
```

---

### Searches YouTube for the top 5 videos on a topic, fetches up to 100 comments per video, and analyzes the stream.

```
python cli.py analyze-topic --topic "Gaming Setup Review" --max-videos 5 --max-comments 100
```

---

### Displays the CLI overview and all available parameters directly in the terminal.

```
python cli.py --help
```
