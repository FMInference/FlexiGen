# The source code in this file is mainly adapted from
# https://github.com/HazyResearch/fm_data_tasks/blob/main/fm_data_tasks/utils/data_utils.py
# which is under Apache License Version 2.0.

"""Data utils."""
import logging
from functools import partial
from pathlib import Path
from typing import Dict, List

import pandas as pd

from flexgen.apps.data_wrangle.utils import constants

logger = logging.getLogger(__name__)


def sample_train_data(train: pd.DataFrame, n_rows: int):
    """
    Sample train data.

    Used when random sampling points for prompt.
    """
    res = train.sample(n_rows)
    return res


def serialize_row(
    row: pd.core.series.Series,
    column_map: Dict[str, str],
    sep_tok: str,
    nan_tok: str,
) -> str:
    """Turn structured row into string."""
    res = []
    for c_og, c_map in column_map.items():
        if str(row[c_og]) == "nan":
            row[c_og] = nan_tok
        else:
            row[c_og] = f"{str(row[c_og]).strip()}"
        res.append(f"{c_map}: {row[c_og]}".lstrip())
    if len(sep_tok) > 0 and sep_tok != ".":
        sep_tok = f" {sep_tok}"
    return f"{sep_tok} ".join(res)


def serialize_match_pair(
    row: pd.core.series.Series,
    column_mapA: Dict[str, str],
    column_mapB: Dict[str, str],
    add_instruction: bool,
    instruction: str,
    suffix: str,
    prod_name: str,
    sep_tok: str,
    nan_tok: str,
) -> str:
    """Turn structured pair of entities into string for matching."""
    res = (
        f"{prod_name} A is {serialize_row(row, column_mapA, sep_tok, nan_tok)}."
        f" {prod_name} B is {serialize_row(row, column_mapB, sep_tok, nan_tok)}."
        f"{suffix} "
    )
    if add_instruction:
        res = f"{instruction} {res}"
    return res


def serialize_imputation(
    row: pd.core.series.Series,
    column_map: Dict[str, str],
    impute_col: str,
    add_instruction: bool,
    instruction: str,
    suffix: str,
    sep_tok: str,
    nan_tok: str,
) -> str:
    """Turn single entity into string for imputation."""
    assert impute_col not in column_map, f"{impute_col} cannot be in column map"
    # Rename to avoid passing white spaced sep token to serialize_row
    sep_tok_ws = sep_tok
    if len(sep_tok) > 0 and sep_tok != ".":
        sep_tok_ws = f" {sep_tok}"
    res = f"{serialize_row(row, column_map, sep_tok, nan_tok)}{sep_tok_ws}{suffix} "
    if add_instruction:
        res = f"{instruction} {res}"
    return res


def serialize_error_detection_spelling(
    row: pd.core.series.Series,
    add_instruction: bool,
    instruction: str,
    suffix: str,
    sep_tok: str,
    nan_tok: str,
) -> str:
    """Turn single cell into string for error detection."""
    column_map = {row["col_name"]: row["col_name"]}
    res = f"Is there a x spelling error in {serialize_row(row, column_map, sep_tok, nan_tok)}{suffix} "
    if add_instruction:
        res = f"{instruction} {res}"
    return res


def serialize_error_detection(
    row: pd.core.series.Series,
    add_prefix: bool,
    instruction: str,
    suffix: str,
    sep_tok: str,
    nan_tok: str,
) -> str:
    """Turn single cell into string for error detection."""
    column_map = {
        c: c
        for c in row.index
        if str(c) not in ["Unnamed: 0", "text", "col_name", "label_str", "is_clean"]
    }
    entire_row = serialize_row(row, column_map, sep_tok, nan_tok)
    column_map = {row["col_name"]: row["col_name"]}
    res = f"{entire_row}\n\nIs there an error in {serialize_row(row, column_map, sep_tok, nan_tok)}{suffix} "
    if add_prefix:
        res = f"{instruction} {res}"
    return res


def serialize_schema_match(
    row: pd.core.series.Series,
    add_prefix: bool,
    instruction: str,
    suffix: str,
    sep_tok: str,
    nan_tok: str,
) -> str:
    """Turn single cell into string for schema matching."""
    res = f"A is {row['left']}. B is {row['right']}. {suffix} "
    if add_prefix:
        res = f"{instruction}\n\n{res}"
    return res


def read_blocked_pairs(
    split_path: str,
    tableA: pd.DataFrame,
    tableB: pd.DataFrame,
    cols_to_drop: List[str],
    col_renaming: Dict[str, str],
    add_instruction: bool,
    instruction: str,
    suffix: str,
    prod_name: str,
    sep_tok: str,
    nan_tok: str,
) -> pd.DataFrame:
    """Read in pre-blocked pairs with T/F match labels."""
    for c in cols_to_drop:
        tableA = tableA.drop(c, axis=1, inplace=False)
        tableB = tableB.drop(c, axis=1, inplace=False)
    if len(col_renaming) > 0:
        tableA = tableA.rename(columns=col_renaming, inplace=False)
        tableB = tableB.rename(columns=col_renaming, inplace=False)

    column_mapA = {f"{c}_A": c for c in tableA.columns if c != "id"}
    column_mapB = {f"{c}_B": c for c in tableB.columns if c != "id"}

    labels = pd.read_csv(split_path)

    mergedA = pd.merge(labels, tableA, right_on="id", left_on="ltable_id")
    merged = pd.merge(
        mergedA,
        tableB,
        right_on="id",
        left_on="rtable_id",
        suffixes=("_A", "_B"),
    )

    merged["text"] = merged.apply(
        lambda row: serialize_match_pair(
            row,
            column_mapA,
            column_mapB,
            add_instruction,
            instruction,
            suffix,
            prod_name,
            sep_tok,
            nan_tok,
        ),
        axis=1,
    )
    merged["label_str"] = merged.apply(
        lambda row: "Yes\n" if row["label"] == 1 else "No\n", axis=1
    )
    return merged


def read_imputation_single(
    split_path: str,
    impute_col: str,
    cols_to_drop: List[str],
    col_renaming: Dict[str, str],
    add_instruction: bool,
    instruction: str,
    suffix: str,
    sep_tok: str,
    nan_tok: str,
) -> pd.DataFrame:
    """Read in table and create label impute col."""
    table = pd.read_csv(split_path)
    for c in cols_to_drop:
        table = table.drop(c, axis=1, inplace=False)
    if len(col_renaming) > 0:
        table = table.rename(columns=col_renaming, inplace=False)
    column_map = {c: c for c in table.columns if c != "id" and c != impute_col}

    table["text"] = table.apply(
        lambda row: serialize_imputation(
            row,
            column_map,
            impute_col,
            add_instruction,
            instruction,
            suffix,
            sep_tok,
            nan_tok,
        ),
        axis=1,
    )
    table["label_str"] = table[impute_col].apply(lambda x: f"{x}\n")
    return table


def read_error_detection_single(
    split_path: str,
    table: pd.DataFrame,
    cols_to_drop: List[str],
    col_renaming: Dict[str, str],
    add_instruction: bool,
    instruction: str,
    suffix: str,
    sep_tok: str,
    nan_tok: str,
    spelling: bool,
) -> pd.DataFrame:
    """Read in table and create label impute col."""
    for c in cols_to_drop:
        table = table.drop(c, axis=1, inplace=False)
    if len(col_renaming) > 0:
        table = table.rename(columns=col_renaming, inplace=False)
    # row_id, col_name, is_clean
    labels = pd.read_csv(split_path)

    if spelling:
        merged = pd.merge(labels, table, left_on="row_id", right_index=True)
        merged["text"] = merged.apply(
            lambda row: serialize_error_detection_spelling(
                row,
                add_instruction,
                instruction,
                suffix,
                sep_tok,
                nan_tok,
            ),
            axis=1,
        )
    else:
        merged = table
        merged["text"] = merged.apply(
            lambda row: serialize_error_detection(
                row,
                add_instruction,
                instruction,
                suffix,
                sep_tok,
                nan_tok,
            ),
            axis=1,
        )

    merged["label_str"] = merged.apply(
        lambda row: "No\n" if row["is_clean"] == 1 else "Yes\n", axis=1
    )
    return merged


def read_schema_match_single(
    split_path: str,
    table: pd.DataFrame,
    cols_to_drop: List[str],
    col_renaming: Dict[str, str],
    add_instruction: bool,
    instruction: str,
    suffix: str,
    sep_tok: str,
    nan_tok: str,
) -> pd.DataFrame:
    """Read in table and create label impute col."""
    file = pd.read_csv(split_path)
    for c in cols_to_drop:
        file = file.drop(c, axis=1, inplace=False)
    if len(col_renaming) > 0:
        file = file.rename(columns=col_renaming, inplace=False)
    # row_id, col_name, is_clean
    # labels = pd.read_csv(split_path)
    # merged = pd.merge(labels, table, left_on="Unnamed: 0", right_index=True)
    file["text"] = file.apply(
        lambda row: serialize_schema_match(
            row,
            add_instruction,
            instruction,
            suffix,
            sep_tok,
            nan_tok,
        ),
        axis=1,
    )
    file["label_str"] = file.apply(
        lambda row: "No\n" if row["label"] == 0 else "Yes\n", axis=1
    )
    return file


def read_raw_data(
    data_dir: str,
    add_instruction: bool = False,
    task_instruction_idx: int = 0,
    sep_tok: str = ".",
    nan_tok: str = "nan",
):
    """Read in data where each directory is unique for a task."""
    data_files_sep = {"test": {}, "train": {}, "validation": {}}
    logger.info(f"Processing {data_dir}")
    if data_dir not in constants.DATA2TASK:
        raise ValueError(
            f"{data_dir} not one of {constants.DATA2TASK.keys()}. Make sure to set DATASET_PATH."
        )
    task = constants.DATA2TASK[data_dir]
    instruction = constants.DATA2INSTRUCT[data_dir]
    suffix = constants.DATA2SUFFIX[data_dir]
    cols_to_drop = constants.DATA2DROPCOLS[data_dir]
    col_renaming = constants.DATA2COLREMAP[data_dir]
    data_dir_p = Path(data_dir)
    if task == "entity_matching":
        train_file = data_dir_p / "train.csv"
        valid_file = data_dir_p / "valid.csv"
        test_file = data_dir_p / "test.csv"
        tableA_file = data_dir_p / "tableA.csv"
        tableB_file = data_dir_p / "tableB.csv"

        tableA = pd.read_csv(tableA_file)
        tableB = pd.read_csv(tableB_file)

        label_col = "label"
        read_data_func = partial(
            read_blocked_pairs,
            tableA=tableA,
            tableB=tableB,
            cols_to_drop=cols_to_drop,
            col_renaming=col_renaming,
            add_instruction=add_instruction,
            instruction=instruction,
            suffix=suffix,
            prod_name=constants.MATCH_PROD_NAME[data_dir],
            sep_tok=sep_tok,
            nan_tok=nan_tok,
        )
    elif task == "data_imputation":
        train_file = data_dir_p / "train.csv"
        valid_file = data_dir_p / "valid.csv"
        test_file = data_dir_p / "test.csv"
        label_col = constants.IMPUTE_COLS[data_dir]
        read_data_func = partial(
            read_imputation_single,
            impute_col=label_col,
            cols_to_drop=cols_to_drop,
            col_renaming=col_renaming,
            add_instruction=add_instruction,
            instruction=instruction,
            suffix=suffix,
            sep_tok=sep_tok,
            nan_tok=nan_tok,
        )
    elif task == "error_detection_spelling":
        train_file = data_dir_p / "train.csv"
        valid_file = data_dir_p / "valid.csv"
        test_file = data_dir_p / "test.csv"
        table_file = data_dir_p / "table.csv"

        table = pd.read_csv(table_file)
        label_col = "is_clean"
        read_data_func = partial(
            read_error_detection_single,
            table=table,
            cols_to_drop=cols_to_drop,
            col_renaming=col_renaming,
            add_instruction=add_instruction,
            instruction=instruction,
            suffix=suffix,
            sep_tok=sep_tok,
            nan_tok=nan_tok,
            spelling=True,
        )
    elif task == "error_detection":
        train_file = data_dir_p / "train.csv"
        valid_file = data_dir_p / "valid.csv"
        test_file = data_dir_p / "test.csv"
        table_file = data_dir_p / "table.csv"

        table = pd.read_csv(table_file)
        label_col = "is_clean"
        read_data_func = partial(
            read_error_detection_single,
            table=table,
            cols_to_drop=cols_to_drop,
            col_renaming=col_renaming,
            add_instruction=add_instruction,
            instruction=instruction,
            suffix=suffix,
            sep_tok=sep_tok,
            nan_tok=nan_tok,
            spelling=False,
        )
    elif task == "schema_matching":
        train_file = data_dir_p / "train.csv"
        valid_file = data_dir_p / "valid.csv"
        test_file = data_dir_p / "test.csv"
        table_file = data_dir_p / "table.csv"
        label_col = "label"
        table = pd.read_csv(table_file)
        read_data_func = partial(
            read_schema_match_single,
            table=table,
            cols_to_drop=cols_to_drop,
            col_renaming=col_renaming,
            add_instruction=add_instruction,
            instruction=instruction,
            suffix=suffix,
            sep_tok=sep_tok,
            nan_tok=nan_tok,
        )
    else:
        raise ValueError(f"Task {task} not recognized.")

    data_files_sep["train"] = read_data_func(train_file)
    # Read validation
    if valid_file.exists():
        data_files_sep["validation"] = read_data_func(valid_file)
    # Read test
    if test_file.exists():
        data_files_sep["test"] = read_data_func(test_file)
    return data_files_sep, label_col


def read_data(
    data_dir: str,
    class_balanced: bool = False,
    add_instruction: bool = False,
    task_instruction_idx: int = 0,
    max_train_samples: int = -1,
    max_train_percent: float = -1,
    sep_tok: str = ".",
    nan_tok: str = "nan",
):
    """Read in data where each directory is unique for a task."""
    data_files_sep, label_col = read_raw_data(
        data_dir=data_dir,
        add_instruction=add_instruction,
        task_instruction_idx=task_instruction_idx,
        sep_tok=sep_tok,
        nan_tok=nan_tok,
    )
    task = constants.DATA2TASK[data_dir]
    # Don't class balance on open ended classificiation tasks
    if class_balanced and task != "data_imputation":
        # Class balance sample the train data
        label_cnts = data_files_sep["train"].groupby(label_col).count()
        sample_per_class = label_cnts.min()["text"]
        logger.info(f"Class balanced: train sample per class: {sample_per_class}")
        data_files_sep["train"] = (
            data_files_sep["train"]
            .groupby(label_col, group_keys=False)
            .apply(lambda x: x.sample(sample_per_class, random_state=42))
        )
    # Shuffle train data
    data_files_sep["train"] = (
        data_files_sep["train"].sample(frac=1, random_state=42).reset_index(drop=True)
    )
    if max_train_samples > 0:
        orig_train_len = len(data_files_sep["train"])
        if max_train_samples > 1.0:
            raise ValueError("max_train_samples must be between 0 and 1")
        max_examples = int(max_train_samples * orig_train_len)
        data_files_sep["train"] = data_files_sep["train"].iloc[:max_examples]
        logger.info(
            f"Length of {data_dir} train is "
            f"{data_files_sep['train'].shape[0]} from {orig_train_len}"
        )
    return data_files_sep