#!/usr/bin/env python3
"""
Resolve model metadata based on a run ID or assessment year.

If the --write-github-output flag is present, the script will write metadata to a
GITHUB_OUTPUT environment variable such that metadata can be used in subsequent
steps of a GitHub workflow. Otherwise, the script will print values to the
console.
"""

import argparse
import json
import os

from pyathena import connect
from pyathena.cursor import DictCursor

import constants


def parse_args() -> argparse.Namespace:
    """Parse command‑line arguments and perform basic validation"""

    parser = argparse.ArgumentParser(
        description=("Resolve model metadata for a model run by ID or assessment year"),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--run-id",
        required=False,
        default="",
        help="Model run ID, mutually exclusive with --year",
    )

    parser.add_argument(
        "--year",
        required=False,
        default="",
        help=(
            "Assessment year to use to derive a final model run, mutually "
            "exclusive with --run-id"
        ),
    )

    parser.add_argument(
        "--pin",
        nargs="*",
        metavar="PIN",
        help="One or more PINs to use for filtering reports",
    )

    parser.add_argument(
        "--township",
        nargs="*",
        help=(
            "Restrict all-PIN mode to one or more County township codes "
            "(two-digit string, e.g. 01, 23)"
        ),
    )

    parser.add_argument(
        "--write-github-output",
        action=argparse.BooleanOptionalAction,
        required=False,
        default=False,
        help="Write output to the GITHUB_OUTPUT env var, for use in workflows",
    )

    args = parser.parse_args()

    # Remove empty strings from list args
    args.pin = [] if args.pin == [""] else args.pin
    args.township = [] if args.township == [""] else args.township

    if bool(args.run_id) == bool(args.year):
        parser.error("Exactly one of --year or --run-id is required")

    return args


def get_township_codes(
    run_id: str, townships: list[str], pins: list[str], write_github_output: bool
) -> list[str]:
    """
    Given a model run ID, return all the township codes for that run.

    If `townships` is non-empty, return it as-is and skip querying altogether.

    If `pins` is non-empty, the function will return one of two outputs depending
    on the value of `write_github_output`:

        - If `True`, returns a list with one element, an empty string
            - This is useful for generating a workflow job matrix with one
              consolidated job
        - If `False`, returns an empty list
    """
    codes = [""] if write_github_output else []

    if townships:
        codes = townships

    elif not pins:
        codes = [
            row[0]
            for row in connect(region_name="us-east-1")
            .cursor()
            .execute(
                f"""
                    SELECT DISTINCT meta_township_code
                    FROM {constants.PINVAL_ASSESSMENT_CARD_TABLE}
                    WHERE run_id = %(run_id)s
                    ORDER BY meta_township_code
                """,
                {"run_id": run_id},
            )
        ]

        if not codes:
            raise ValueError(f"No township codes found for model run ID '{run_id}'")

    return codes


def get_run_id(run_id: str, year: str) -> str:
    """Return a model run ID derived from two optional arguments, a run ID and
    an assessment year.

    If the run ID is non-empty, return it directly. Otherwise, query the
    final model for the given assessment year, and return that model's
    run ID."""
    if not run_id:
        run_ids = [
            row[0]
            for row in connect(region_name="us-east-1")
            .cursor()
            .execute(
                """
                    SELECT run_id
                    FROM model.final_model
                    WHERE year = %(year)s
                        AND is_final
                        AND type = 'res'
                """,
                {"year": year},
            )
        ]

        if not run_ids:
            raise ValueError(f"No final model run found for year '{year}'")

        run_id = run_ids[0]

    return run_id


def get_model_metadata(run_id: str) -> dict:
    """Return a dict of model metadata for a model based on its run ID."""
    metadata = [
        row
        for row in connect(region_name="us-east-1")
        .cursor(DictCursor)
        .execute(
            """
                SELECT run_id, assessment_year, assessment_triad as triad
                FROM model.metadata
                WHERE run_id = %(run_id)s
            """,
            {"run_id": run_id},
        )
    ]

    if not metadata:
        raise ValueError(f"No model metadata found for model run ID {run_id}")

    return metadata[0]


def main() -> None:
    """Main entrypoint for the script"""
    args = parse_args()

    run_id = get_run_id(args.run_id, args.year)
    metadata = get_model_metadata(run_id)
    township_codes = get_township_codes(
        run_id, args.township, args.pin, args.write_github_output
    )

    matrix = json.dumps({"township": township_codes})
    count = len(township_codes)

    output_vars = [
        f"matrix={matrix}",
        f"count={count}",
        f"assessment-year={metadata['assessment_year']}",
        f"run-id={metadata['run_id']}",
        f"triad={metadata['triad']}",
    ]

    # Log all outputs to console, for debugging purposes
    for var in output_vars:
        print(var)

    if args.write_github_output:
        # Write all outputs to the reserved env var that controls GitHub
        # workflow job output
        with open(os.environ["GITHUB_OUTPUT"], "a") as fh:
            for var in output_vars:
                fh.write(f"{var}\n")


if __name__ == "__main__":
    main()
