import pandas as pd

# Path to your parquet file
file_path = "comments_analyzed.parquet"

try:
    # Read the parquet file
    df = pd.read_parquet(file_path)

    # Check available columns (helps if 'Themes' is spelled differently)
    print(f"Spalten in der Datei: {list(df.columns)}\n")

    # Column name for themes (adjust if necessary, e.g. 'theme' or 'Themes')
    column_name = "theme"

    if column_name in df.columns:
        # 1. Absolute frequency of themes
        print("=== Absolute Verteilung der Themes ===")
        print(df[column_name].value_counts(dropna=False))

        # 2. Relative frequency (percentage)
        print("\n=== Relative Verteilung (Prozent) ===")
        print(df[column_name].value_counts(normalize=True, dropna=False) * 100)

    else:
        print(
            f"Spalte '{column_name}' nicht gefunden. Bitte wähle eine der vorhandenen Spalten."
        )

except Exception as e:
    print(f"Fehler beim Lesen der Parquet-Datei: {e}")