import pandas as pd

def _read_dataset(path, required_columns):
    df = pd.read_json(path, lines=True)
    missing = set(required_columns) - set(df.columns)
    if missing:
        raise ValueError(
            f"Columnas faltantes en {path}: {sorted(missing)}. "
            f"Columnas disponibles: {df.columns.tolist()}"
        )
    return df[required_columns]

def corpus_test():
    df = _read_dataset(
        "hf://datasets/mteb/trec-covid/qrels/test.jsonl",
        ["query-id", "corpus-id", "score"],
    )
    return df.to_dict(orient="records")

def corpus():
    df = _read_dataset(
        "hf://datasets/mteb/trec-covid/corpus.jsonl",
        ["_id", "title","text"],
    )
    return df.to_dict(orient="records")

def corpus_queries():
    df = _read_dataset(
        "hf://datasets/mteb/trec-covid/queries.jsonl",
        ["_id", "text"],
    )
    return df.to_dict(orient="records")

print("Corpus test:", corpus_test()[:2])
print("Corpus:", corpus()[:2])
print("Corpus queries:", corpus_queries()[:2])