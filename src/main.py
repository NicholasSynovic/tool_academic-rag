from pathlib import Path

import click
from modin import pandas as pd
from modin.pandas import DataFrame
from sqlalchemy import Engine, create_engine


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
def main(inputPath: Path) -> None:
    dbEngine: Engine = create_engine(url=f"sqlite:///{inputPath}")

    df: DataFrame = readDB(dbEngine=dbEngine)

    print(df)


if __name__ == "__main__":
    main()
