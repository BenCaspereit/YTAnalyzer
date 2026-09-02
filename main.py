import argparse
from pathlib import Path
import json
import time
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from analyzer import CommentAnalyzer

BASE_DIR = Path(__file__).resolve().parent

def stream_comments_from_json(file_path: Path):
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        for entry in data:
            if isinstance(entry, dict) and "comment" in entry:
                yield entry["comment"]
            elif isinstance(entry, str):
                yield entry

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

def run_pipeline(
    input_file: Path,
    output_file: Path,
    tasks: list[str],
    batch_size: int = 32,
    device: str = None
):
    print(f"🚀 Starte Analyse für '{input_file}'...")
    print(f"📦 Batch Size: {batch_size} | Tasks: {tasks}")

    
    # 1. initialize analyzer
    analyzer = CommentAnalyzer(tasks=tasks, device=device)

    # 2. set up Parquet Writer 
    writer = None
    total_processed = 0
    start_time = time.time()

    comment_stream = stream_comments_from_json(input_file)
    batch_generator = process_in_batches(comment_stream, batch_size)

    try:
        for batch_idx, comment_batch in enumerate(batch_generator):
    
            results = analyzer.analyze_batch(comment_batch)

            df = pd.DataFrame(results)
            table = pa.Table.from_pandas(df)

            if writer is None:
                writer = pq.ParquetWriter(output_file, table.schema, compression='snappy')
            writer.write_table(table)

            total_processed += len(results)
            if (batch_idx + 1) % 10 == 0 or len(comment_batch) < batch_size:
                elapsed = time.time() - start_time
                speed = total_processed / elapsed if elapsed > 0 else 0
                print(f"Progress: {total_processed} Kommentare verarbeitet ({speed:.1f} Comm/s)")

    finally:
        if writer:
            writer.close()
            print(f"✅ Fertig! {total_processed} Ergebnisse gespeichert in '{output_file}'")

if __name__ == "__main__":
    # Einfache CLI-Schnittstelle
    parser = argparse.ArgumentParser(description="YouTube Comment Sentiment & Topic Analyzer")
    parser.add_argument("--input", type=str, default="comments.json", help="Pfad zur Eingabe-JSON")
    parser.add_argument("--output", type=str, default="results.parquet", help="Pfad zur Ausgabe-Parquet")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch Size für Inferenz")
    parser.add_argument("--tasks", nargs="+", default=["sentiment", "emotion", "intention", "theme"], 
                        help="Tasks z.B.: --tasks sentiment emotion")
    parser.add_argument("--device", type=str, default=None, help="cuda oder cpu")

    args = parser.parse_args()

    run_pipeline(
        input_file=BASE_DIR / args.input,
        output_file=BASE_DIR / args.output,
        tasks=args.tasks,
        batch_size=args.batch_size,
        device=args.device
    )