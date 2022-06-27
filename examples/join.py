import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import logging

import torch
import torch_mlir
import iree_torch

from typing import Sequence

log = logging.getLogger(__name__)

# TODO(bsarden): Publish .parquet files for downstream consumers.
DATASET_ROOT = Path("/work/bsarden/data/sf_1/parq")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a `join` operation using input parquet data from the TPC-H benchmark.",
    )
    argparse.ArgumentParser()
    parser.add_argument(
        "--orders-parquet-path",
        type=Path,
        default=DATASET_ROOT / "orders.parquet",
        help="Path to the .parquet Table file to load for left-hand-side of join.",
    )
    parser.add_argument(
        "--lineitem-parquet-path",
        type=Path,
        default=DATASET_ROOT / "lineitem.parquet",
        help="Path to the .parquet Table file to load for left-hand-side of join.",
    )
    parser.add_argument(
        "-n",
        "--num-rows",
        type=int,
        default=20,
        help="Number of rows to do a `Join` on.",
    )
    return parser.parse_args()


class DataFrameJoin:
    """Runs a join operator using `pandas` API's."""

    def forward(
        self,
        left_df,
        right_df,
        left_cols: Sequence[str],
        right_cols: Sequence[str],
        left_on: str,
        right_on: str,
    ):
        """Execute a join operation using the `Pandas` API."""
        return pd.merge(
            left_df[left_cols],
            right_df[right_cols],
            left_on=left_on,
            right_on=right_on,
            # Provides a _merge column with extra info on source of each merge
            # https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.merge.html#pandas.DataFrame.merge
            # indicator=True,  # useful if you want to see how the columns were merged
        )


class SimpleJoinModel(torch.nn.Module):
    """Runs a join operator using `join` API's."""

    def forward(
        self,
        left: torch.Tensor,
        right: torch.Tensor,
        left_on: int,
        right_on: int,
        val_can_be_null: bool = False,
    ):
        """Performs a join_operator
        Args:
            left: left table to join on
            right: right table to join on
            left_on: ordinal value indicating the columns to join with on the left table
            right_on: ordinal value indicating the columns to join with on the right table
        Returns:
            A joined tensor of size `tensor<max(len(right_on, left_on), ?>`)
        """
        # TODO: Improve impl from nested-join -> merge / hash join.
        # Can we do the appending without using a python list?
        result = []
        for outer_row in right:
            for inner_row in left:
                if inner_row[left_on] == outer_row[right_on]:
                    result.append(torch.cat((inner_row, outer_row), dim=0))
        return torch.stack(result)


def _find_col_indices(df: pd.DataFrame, col_names: Sequence[str]):
    """Helper function to return ordinal representation(s) of `pd.DataFrame` column(s)."""
    indices = []
    for cname in col_names:
        for idx, col in enumerate(df.columns):
            if cname == col:
                indices.append(idx)
    return indices


def main(args):
    log.debug(f"{args=}")
    orders_df = pd.read_parquet(DATASET_ROOT / "orders.parquet")
    lineitem_df = pd.read_parquet(DATASET_ROOT / "lineitem.parquet")

    if args.num_rows > orders_df.shape[0]:
        raise ValueError(
            f"--num-rows is out of bounds. {args.num_rows=}, {orders_df.shape[0]=}"
        )

    orders_df_small = orders_df.head(args.num_rows)
    lineitem_df_small = lineitem_df.head(args.num_rows)

    log.info(f"[Pandas] running `join` on {args.num_rows} rows.")
    df_join_model = DataFrameJoin()
    expected_df = df_join_model.forward(
        orders_df_small,
        lineitem_df_small,
        left_cols=["o_orderkey"],
        right_cols=["l_orderkey", "l_quantity"],
        left_on="o_orderkey",
        right_on="l_orderkey",
    )
    log.info("[Pandas] `join` complete.")

    o_orderkey_tensor = torch.tensor(orders_df_small["o_orderkey"].values)
    l_orderkey_tensor = torch.tensor(lineitem_df_small["l_orderkey"].values)
    l_quantity_tensor = torch.tensor(lineitem_df_small["l_quantity"].values)

    o_cols = o_orderkey_tensor.reshape(args.num_rows, 1)
    l_cols = torch.stack((l_orderkey_tensor, l_quantity_tensor), dim=1)

    log.info("Checking to make sure df inputs equal torch inputs")
    df_o_cols_np = orders_df_small["o_orderkey"].to_numpy()
    df_l_cols_np = lineitem_df_small[["l_orderkey", "l_quantity"]].to_numpy()
    torch_o_cols_np = o_cols.numpy()
    torch_l_cols_np = l_cols.numpy()
    assert np.allclose(
        df_o_cols_np.flatten(), torch_o_cols_np.flatten()
    ), "Orders table input col mismatch!"
    assert np.allclose(
        df_l_cols_np.flatten(), torch_l_cols_np.flatten()
    ), "Lineitem table input col mismatch!"

    log.info("Performing pytorch join operator.")
    join_model = SimpleJoinModel()
    actual = join_model(
        left=o_cols,
        right=l_cols,
        left_on=_find_col_indices(orders_df, ["o_orderkey"])[0],
        right_on=_find_col_indices(lineitem_df, ["l_orderkey"])[0],
        val_can_be_null=False,
    )
    log.info(f"{actual.shape=}")
    log.info(f"{expected_df.shape=}")

    log.info("Comparing result vs actual")
    actual_np = actual.numpy()
    expected_np = expected_df.to_numpy()

    if not np.all(np.isclose(actual_np, expected_np)):
        raise ValueError(
            f"Join Result Mismatch! \n"
            f"actual_np =\n{actual_np}\n"
            f"expected_np =\n{expected_np}\n"
        )
    log.info("SUCCESS!")


if __name__ == "__main__":
    logging.basicConfig()
    log.setLevel(logging.DEBUG)
    main(parse_arguments())
