# The source code in this file is mainly adapted from
# https://github.com/HazyResearch/fm_data_tasks/blob/main/fm_data_tasks/utils/prompt_utils.py
# which is under Apache License Version 2.0.

"""Prompt utils."""
import logging
from typing import Any, Tuple

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from flexgen.apps.data_wrangle.utils import constants
from flexgen.apps.data_wrangle.utils.data_utils import sample_train_data

logger = logging.getLogger(__name__)


def get_manual_prompt(data_dir: str, example: pd.Series) -> str:
    """Get manual prompt for data name."""
    if data_dir not in constants.DATA2TASK.keys():
        raise ValueError(f"{data_dir} not recognized for prompts")
    subkey_attr = constants.DATA2EXAMPLE_SUBKEY_ATTR[data_dir]
    if subkey_attr is None:
        if not isinstance(constants.PREFIXES[data_dir], str):
            raise ValueError(f"Prefix was not a string for {data_dir}")
        return constants.PREFIXES[data_dir]
    else:
        if not isinstance(constants.PREFIXES[data_dir], dict):
            raise ValueError(
                f"Prefix was not a dict with {subkey_attr} subkeys for {data_dir}"
            )
        return constants.PREFIXES[data_dir][str(example[subkey_attr])]


def get_random_prompt(train_data: pd.DataFrame, num_examples: int = 10) -> str:
    """Get random examples for prompt from trian data."""
    prefix_exs_rows = sample_train_data(train_data, num_examples)
    serialized_prefixes = [
        (txt + label).strip()
        for txt, label in zip(prefix_exs_rows["text"], prefix_exs_rows["label_str"])
    ]
    prefix_exs = "\n\n".join(serialized_prefixes) + "\n"
    return prefix_exs


def get_validation_prompt(
    validation_path: str, num_examples: int = 10, task: str = "entity_matching"
) -> str:
    """Get prompt from validation errors."""
    return get_validation_embs_prompts(
        df=pd.read_feather(validation_path),
        num_exs=num_examples,
        model_name="sentence-transformers/sentence-t5-base",
        task=task,
    )


def setup_st_pipeline(model_name: str) -> SentenceTransformer:
    """Get Sentence Transformer pipeline."""
    logger.info("Loading SentenceTransfomer pipeline")
    pipeline = SentenceTransformer(model_name)
    return pipeline


def extract_st_features(
    errors: pd.DataFrame, model: SentenceTransformer, text_col: str = "text"
) -> np.ndarray:
    """Extract ST features."""
    logger.info("Extracting SentenceTransfomer features")
    feats = model.encode(errors[text_col].tolist())
    return feats


def get_hard_samples(
    df: pd.DataFrame, errors_index: pd.Index, embs: np.ndarray, num_clusters: int = 10
) -> Tuple[pd.DataFrame, Any]:
    """
    Choose samples that are nearby eachother in embedding space but different labels.

    One sample must be an error.
    """
    from sklearn.metrics.pairwise import cosine_similarity

    # Upper triangle of cosine similarity matrix
    sim = np.triu(cosine_similarity(embs))

    top_indexes = []
    for row_idx in range(sim.shape[0]):
        sorted_idxs = np.argsort(sim[row_idx], axis=0)[::-1]
        for idx in sorted_idxs[: num_clusters + 1]:
            # Skip rows similarity to itself
            if idx == row_idx:
                continue
            top_indexes.append([[row_idx, idx], sim[row_idx, idx]])
    # Get most similar pairs
    top_indexes = sorted(top_indexes, key=lambda x: x[1], reverse=True)
    df_indexes = []
    for i in range(len(top_indexes)):
        row_idx, col_idx = top_indexes[i][0]
        if (row_idx in errors_index or col_idx in errors_index) and (
            df.iloc[row_idx]["label_str"] != df.iloc[col_idx]["label_str"]
        ):
            if row_idx not in df_indexes:
                df_indexes.append(row_idx)
            if col_idx not in df_indexes:
                df_indexes.append(col_idx)
    return df.iloc[df_indexes[:num_clusters]], None


def get_validation_embs_prompts(
    df: pd.DataFrame,
    num_exs: int = 10,
    model_name: str = "sentence-transformers/sentence-t5-base",
    task: str = "entity_matching",
) -> str:
    """
    Generate prompt from cluster of data errors.

    We use sentence embeddings to cluster each error example.
    We then select `num_exs` examples from each cluster.

    If all examples are of one class, we randomly swap one
    example for an instance from the validation data with the missing
    class.
    """
    errors = df[
        df["label_str"].str.lower().str.strip() != df["preds"].str.lower().str.strip()
    ]
    pipeline = setup_st_pipeline(model_name)
    embs = extract_st_features(df, pipeline)
    samples, _ = get_hard_samples(df, errors.index, embs, num_clusters=num_exs)
    if task in {"entity_matching", "error_detection"}:
        # Add missing class if all examples are of one Yes/No class
        # (We do not do this for imputation, only Yes/No)
        missing_class = None
        if len(samples[samples["label_str"].str.strip() == "Yes\n"]) == len(samples):
            missing_class = "No\n"
        elif len(samples[samples["label_str"].str.strip() == "No\n"]) == len(samples):
            missing_class = "Yes\n"
        if missing_class is not None:
            pre_len = len(samples)
            drop_indices = np.random.choice(samples.index, 1, replace=False)
            samples = samples.drop(drop_indices)
            samples = samples.append(
                df[df["label_str"].str.strip() == missing_class].sample(1)
            )
            assert len(samples) == pre_len

    logger.info(f"Number of samples: {len(samples)}")
    new_prompt = (
        "\n\n".join(
            [
                (txt + label).strip()
                for txt, label in zip(samples["text"], samples["label_str"])
            ]
        )
        + "\n"
    )
    return new_prompt