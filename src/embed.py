from pathlib import Path
from warnings import filterwarnings

import click
from chromadb import PersistentClient
from chromadb.api.models.Collection import Collection
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


def toChromaDB(
    ids: Series,
    documents: Series,
    embeddings: ndarray,
    dbPath: Path,
) -> None:
    client: PersistentClient = PersistentClient(path=dbPath.__str__())
    collection: Collection = client.create_collection(
        name="arXiv",
        get_or_create=True,
    )
    collection.add(
        ids=ids.to_list(),
        embeddings=embeddings.tolist(),
        documents=documents.to_list(),
    )


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
    "-o",
    "--output",
    "outputPath",
    help="Path to store Chroma DB",
    type=click.Path(
        exists=False,
        dir_okay=True,
        writable=True,
        resolve_path=True,
        path_type=Path,
    ),
    required=False,
    default="../data/chroma",
    show_default=True,
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
def main(inputPath: Path, outputPath: Path, modelName: str) -> None:
    dbEngine: Engine = create_engine(url=f"sqlite:///{inputPath}")

    print(f"Reading {inputPath}...")
    df: DataFrame = readDB(dbEngine=dbEngine)

    print("Getting abstracts...")
    abstracts: Series = df["abstract"]

    embeddings: ndarray = embed(modelID=modelName, content=abstracts)

    toChromaDB(
        embeddings=embeddings[0:1000],
        dbPath=outputPath,
        ids=df["id"][0:1000],
        documents=abstracts[0:1000],
    )


if __name__ == "__main__":
    filterwarnings(action="ignore")
    main()
