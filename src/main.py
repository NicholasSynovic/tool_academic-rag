import pickle  # nosec
from pathlib import Path

import click
from modin import pandas as pd
from modin.pandas import DataFrame, Series
from numpy import ndarray
from sentence_transformers import SentenceTransformer
from sqlalchemy import Engine, create_engine


def embed(modelID: str, content: Series) -> ndarray:
    model: SentenceTransformer = SentenceTransformer(
        model_name_or_path=modelID,
        device="cuda",
    )

    return model.encode(
        sentences=content.to_list(),
        show_progress_bar=True,
        batch_size=100,
    )


def readDB(dbEngine: Engine) -> DataFrame:
    return pd.read_sql_table(table_name="documents", con=dbEngine)


@click.command()
@click.option(
    "-i",
    "--input",
    "inputPath",
    help="Path to arXiv SQLite3 database",
    type=click.Path(
        exists=True,
        file_okay=True,
        readable=True,
        resolve_path=True,
        path_type=Path,
    ),
    required=True,
)
@click.option(
    "-m",
    "--model",
    "modelName",
    help="Name of the embedding model",
    type=str,
    required=False,
    default="all-mpnet-base-v2",
    show_default=True,
)
def main(inputPath: Path, modelName: str) -> None:
    dbEngine: Engine = create_engine(url=f"sqlite:///{inputPath}")

    df: DataFrame = readDB(dbEngine=dbEngine)

    abstracts: Series = df["abstract"][0:1000]

    embeddings: ndarray = embed(modelID=modelName, content=abstracts)

    with open("embeddings.ndarray.pickle", "wb") as pf:
        pickle.dump(obj=embeddings, file=pf)
        pf.close()


if __name__ == "__main__":
    main()
