"""Files of the SeisBench datasets on the remote server, and comparison of two metadata tables.

Used by notebook 09 to check the ESM25GoodMP dataset (manually processed waveforms of the good
records) before downloading it: size of the waveform files, metadata, and how its records and
columns relate to those of ESM25GoodCV.
"""

from __future__ import annotations

import re
import urllib.request
from collections.abc import Iterable, Sequence
from pathlib import Path

import pandas as pd
from tqdm.auto import tqdm

REPOSITORY_PLACEHOLDER = "{repository_name}"
CONTENT_LENGTH_HEADER = "Content-Length"
DEFAULT_TIMEOUT_S = 60.0
DOWNLOAD_CHUNK_BYTES = 1 << 20
PARTIAL_SUFFIX = ".part"


def remote_dataset_url(template: str, repository_name: str, file_name: str) -> str:
    """URL of a file of a remote SeisBench dataset.

    Args:
        template: ``waveforms.remote_url_template`` (URL of the waveform file, with ``{repository_name}``).
        repository_name: Dataset folder on the server (e.g. ``esm25goodmp``).
        file_name: File inside the dataset folder (e.g. ``metadata.csv``).

    Returns:
        URL of the file.

    Raises:
        ValueError: If the template has no ``{repository_name}`` placeholder.
    """
    if REPOSITORY_PLACEHOLDER not in template:
        raise ValueError(f"The URL template must contain '{REPOSITORY_PLACEHOLDER}'.")
    folder = template.format(repository_name=repository_name).rsplit("/", 1)[0]
    return f"{folder}/{file_name}"


def remote_file_size(url: str, timeout_s: float = DEFAULT_TIMEOUT_S) -> int:
    """Size of a remote file from the ``Content-Length`` of a HEAD request.

    Args:
        url: File URL.
        timeout_s: Timeout of the request.

    Returns:
        Size in bytes.

    Raises:
        RuntimeError: If the server does not report the size.
        urllib.error.URLError: If the server cannot be reached.
    """
    request = urllib.request.Request(url, method="HEAD")
    with urllib.request.urlopen(request, timeout=timeout_s) as response:
        length = response.headers.get(CONTENT_LENGTH_HEADER)
    if length is None:
        raise RuntimeError(f"The server does not report the size of {url}.")
    return int(length)


def download_file(url: str, path: Path, timeout_s: float = DEFAULT_TIMEOUT_S) -> Path:
    """Download a file, written under a temporary name and renamed when complete.

    Args:
        url: File URL.
        path: Destination (its folder is created if needed).
        timeout_s: Timeout of the connection.

    Returns:
        The destination path.

    Raises:
        urllib.error.URLError: If the server cannot be reached.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + PARTIAL_SUFFIX)
    with urllib.request.urlopen(url, timeout=timeout_s) as response, partial.open("wb") as output:
        total = response.headers.get(CONTENT_LENGTH_HEADER)
        with tqdm(total=int(total) if total else None, unit="B", unit_scale=True, desc=path.name) as progress:
            while chunk := response.read(DOWNLOAD_CHUNK_BYTES):
                output.write(chunk)
                progress.update(len(chunk))
    partial.replace(path)
    return path


def key_overlap(first: pd.Series, second: pd.Series, names: tuple[str, str]) -> pd.Series:
    """Counts of the identifiers of two tables: in each, in both, in one only, duplicated.

    Args:
        first: Identifiers of the first table.
        second: Identifiers of the second table.
        names: Names of the two tables, used in the labels.

    Returns:
        Series of counts.
    """
    first_keys, second_keys = set(first.dropna()), set(second.dropna())
    return pd.Series({
        f"records in {names[0]}": len(first),
        f"records in {names[1]}": len(second),
        "identifiers in both": len(first_keys & second_keys),
        f"only in {names[0]}": len(first_keys - second_keys),
        f"only in {names[1]}": len(second_keys - first_keys),
        f"duplicated in {names[0]}": int(first.duplicated().sum()),
        f"duplicated in {names[1]}": int(second.duplicated().sum()),
    })


def columns_only_in(table: pd.DataFrame, other: pd.DataFrame) -> list[str]:
    """Columns of a table that the other table does not have, in their order.

    Args:
        table: Table whose columns are listed.
        other: Reference table.

    Returns:
        Column names.
    """
    return [column for column in table.columns if column not in other.columns]


def matching_columns(columns: Iterable[str], patterns: Sequence[str]) -> list[str]:
    """Columns whose name contains any of the patterns (regular expressions, case ignored).

    Args:
        columns: Column names.
        patterns: Regular expressions.

    Returns:
        Matching column names, in their order.
    """
    expression = re.compile("|".join(f"(?:{pattern})" for pattern in patterns), flags=re.IGNORECASE)
    return [column for column in columns if expression.search(column)]


def shared_column_agreement(first: pd.DataFrame, second: pd.DataFrame, key: str, columns: Sequence[str]) -> pd.DataFrame:
    """Share of the records in both tables whose values agree, for columns present in both.

    Records are matched on ``key``; two missing values count as agreeing.

    Args:
        first: First table.
        second: Second table.
        key: Identifier column present in both.
        columns: Columns to compare (present in both).

    Returns:
        Table with one row per column: matched records and share of agreeing values.
    """
    merged = first[[key, *columns]].merge(second[[key, *columns]], on=key, suffixes=("_first", "_second"))
    rows = {}
    for column in columns:
        left, right = merged[f"{column}_first"], merged[f"{column}_second"]
        agree = (left == right) | (left.isna() & right.isna())
        rows[column] = {"matched_records": len(merged), "share_equal": float(agree.mean()) if len(merged) else float("nan")}
    return pd.DataFrame.from_dict(rows, orient="index")
