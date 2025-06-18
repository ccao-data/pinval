#!/usr/bin/env python3
"""
List township codes for a specific triad and model run ID.

Examples
--------
Print an array of township codes in the North tri for a model run :
    $ python3 list_townships.py \
        --run-id 2025-02-11-charming-eric \
        --triad north

Write a matrix of township codes to the GITHUB_OUTPUT env var, for use in a
GitHub workflow:
    $ python3 list_townships.py \
        --run-id 2025-02-11-charming-eric \
        --triad north \
        --write-github-output
"""

import argparse
import json
import os

from pyathena import connect

from constants import RUN_ID_MAP, TRIAD_CHOICES


def parse_args() -> argparse.Namespace:
    """Parse command‑line arguments and perform basic validation"""

    parser = argparse.ArgumentParser(
        description="List township codes for a triad and model run ID",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--run-id",
        required=True,
        choices=list(
            RUN_ID_MAP.keys()
        ),  # Temporarily limits run_ids to those in the map
        help="Model run ID (e.g. 2025-02-11-charming-eric)",
    )

    parser.add_argument(
        "--triad",
        required=False,
        choices=TRIAD_CHOICES,
        help="List townships in this triad that were reassessed by the model run",
    )

    parser.add_argument(
        "--write-github-output",
        action=argparse.BooleanOptionalAction,
        help="Write output to the GITHUB_OUTPUT env var, for use in workflows",
    )

    args = parser.parse_args()

    return args


def get_township_codes(run_id: str, triad: str | None) -> list[str]:
    """
    Given a model run ID and a triad, return all the township codes for that
    triad that were assessed in the model run.

    Returns an empty list if no towns in the triad were assessed in the run.
    """
    codes = [""]

    if triad:
        sql = f"""
        SELECT DISTINCT meta_township_code
        FROM pinval.vw_assessment_card
        WHERE run_id = '{run_id}'
            AND assessment_triad = '{triad}'
        ORDER BY meta_township_code
        """
        codes = [
            row[0] for row in connect(region_name="us-east-1").cursor().execute(sql)
        ]

    return codes


def main() -> None:
    """Main entrypoint for the script"""
    args = parse_args()
    codes = get_township_codes(args.run_id, args.triad)

    if args.write_github_output:
        matrix = json.dumps({"township": codes})
        count = len(codes)
        print(f"matrix={matrix}")

        with open(os.environ["GITHUB_OUTPUT"], "a") as fh:
            fh.write(f"matrix={matrix}\n")
            fh.write(f"count={count}\n")
    else:
        print(codes)


if __name__ == "__main__":
    main()
