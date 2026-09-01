import typer
import json
import time
from typing import List, Optional
from pathlib import Path
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from scraper import YouTubeScraper
from analyzer import CommentAnalyzer

app = typer.Typer(
    name="ytanalyzer",
    help="🤖 YouTube AI Comment Sentiment, Emotion & Topic Pipeline",
    add_completion=False
)

INPUT_DIR = Path("input")


def process_in_batches(generator, batch_size: int):
    # Batch processing generator    
    batch = []
    for item in generator:
        batch.append(item)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def load_comments_from_file(file_path: Path):
    # reads comments from JSON, CSV, Parquet or TXT
    if not file_path.exists():
        # if just the filename is given, try to find it in the input folder
        fallback_path = INPUT_DIR / file_path
        if fallback_path.exists():
            file_path = fallback_path
        else:
            typer.secho(f"❌ File not found: {file_path}", fg=typer.colors.RED)
            raise typer.Exit(code=1)

    typer.echo(f"📂 Loading comments from: {file_path}")

    ext = file_path.suffix.lower()

    if ext == ".json":
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            for entry in data:
                if isinstance(entry, dict):
                    comment_text = entry.get("comment") or entry.get("text")
                    if comment_text:
                        yield {**entry, "comment": comment_text}
                elif isinstance(entry, str):
                    yield {"comment": entry}

    elif ext == ".csv":
        df = pd.read_csv(file_path)
        col = "comment" if "comment" in df.columns else df.columns[0]
        for row in df.to_dict(orient="records"):
            yield {**row, "comment": str(row[col])}

    elif ext == ".parquet":
        df = pd.read_parquet(file_path)
        col = "comment" if "comment" in df.columns else df.columns[0]
        for row in df.to_dict(orient="records"):
            yield {**row, "comment": str(row[col])}

    elif ext == ".txt":
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    yield {"comment": line}

    else:
        typer.secho(f"❌ Unsupported format: {ext}. Allowed: .json, .csv, .parquet, .txt", fg=typer.colors.RED)
        raise typer.Exit(code=1)



@app.command()
def analyze_video(
    video_id: str = typer.Option(..., "--video-id", "-v", help="YouTube Video ID"),
    output: Path = typer.Option(Path("results/results.parquet"), "--output", "-o", help="Target path for the Parquet file"),
    max_comments: int = typer.Option(1000, "--max-comments", "-m", help="Max. comments per Video"),
    batch_size: int = typer.Option(32, "--batch-size", "-b", help="KI Batch Size"),
    tasks: List[str] = typer.Option(["sentiment", "emotion", "intention", "theme"], "--tasks", "-t", help="Analyse-Tasks")
):
    # Analyze comments for a specific YouTube video ID
    typer.echo(f"🎬 Get comments for the video: {video_id}")
    
    scraper = YouTubeScraper()
    analyzer = CommentAnalyzer(tasks=tasks)
    
    comment_stream = scraper.fetch_comments_for_video(video_id, max_comments=max_comments)
    _run_pipeline(comment_stream, analyzer, output, batch_size)

@app.command()
def analyze_topic(
    topic: str = typer.Option(..., "--topic", "-q", help="Search term / topic"),
    max_videos: int = typer.Option(10, "--max-videos", help="Number of videos to search for"),
    max_comments_per_video: int = typer.Option(200, "--max-comments", help="Max. comments per Video"),
    output: Path = typer.Option(Path("results/topic_results.parquet"), "--output", "-o", help="Target path for the Parquet file"),
    batch_size: int = typer.Option(32, "--batch-size", "-b", help="KI Batch Size"),
    tasks: List[str] = typer.Option(["sentiment", "emotion", "intention", "theme"], "--tasks", "-t")
):
    """Searches for videos on a topic and analyzes their comments."""
    typer.echo(f"🔍 Searching for videos on the topic: '{topic}'...")
    
    scraper = YouTubeScraper()
    analyzer = CommentAnalyzer(tasks=tasks)
    
    video_ids = scraper.get_video_ids_for_topic(topic, max_videos=max_videos)
    typer.echo(f"📌 {len(video_ids)} Videos found. Starting comment download & AI analysis...")

    def multi_video_stream():
        for vid in video_ids:
            yield from scraper.fetch_comments_for_video(vid, max_comments=max_comments_per_video)

    _run_pipeline(multi_video_stream(), analyzer, output, batch_size)


@app.command()
def analyze_file(
    file: Path = typer.Option(..., "--file", "-f", help="Pfad oder Dateiname im 'input/'-Ordner (JSON, CSV, Parquet, TXT)"),
    output: Path = typer.Option(Path("results/file_results.parquet"), "--output", "-o", help="Zielpfad"),
    batch_size: int = typer.Option(32, "--batch-size", "-b", help="KI Batch Size"),
    tasks: List[str] = typer.Option(["sentiment", "emotion", "intention", "theme"], "--tasks", "-t")
):
    """Analysiert Kommentare aus einer LOKALEN Datei (z.B. aus dem input/ Ordner)."""
    analyzer = CommentAnalyzer(tasks=tasks)
    comment_stream = load_comments_from_file(file)
    _run_pipeline(comment_stream, analyzer, output, batch_size)


    
def _run_pipeline(comment_stream, analyzer: CommentAnalyzer, output_path: Path, batch_size: int):
    start_time = time.perf_counter()
    # Helper function: Streams data through the analyzer and saves to Parquet.
    writer = None
    total_count = 0

    for batch_dicts in process_in_batches(comment_stream, batch_size):
        if not batch_dicts:
            continue

        # Extract raw comment texts for AI analysis
        raw_texts = [d["comment"] for d in batch_dicts]
        ai_results = analyzer.analyze_batch(raw_texts)

        # AI results are combined with the original metadata for each comment
        combined_data = []
        for meta, ai_res in zip(batch_dicts, ai_results):
            combined_entry = {**meta, **ai_res}
            combined_data.append(combined_entry)

        df = pd.DataFrame(combined_data)
        table = pa.Table.from_pandas(df)

        if writer is None:
            writer = pq.ParquetWriter(output_path, table.schema, compression="snappy")
        
        writer.write_table(table)
        total_count += len(combined_data)
        typer.echo(f"✓ {total_count} comments processed...")

    elapsed_time = time.perf_counter() - start_time 
    if writer:
        writer.close()
        typer.secho(f"✨ Finished! {total_count} results saved to '{output_path}' in {elapsed_time:.2f} seconds.", fg=typer.colors.GREEN, bold=True)
    else:
        typer.secho("⚠️ No matching comments found.", fg=typer.colors.YELLOW)

if __name__ == "__main__":
    app()