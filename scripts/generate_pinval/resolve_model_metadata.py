#!/usr/bin/env python3
"""
Resolve model metadata based on a comps run ID or assessment year.

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
        description=(
            "Resolve model metadata for a model run by comps run ID or assessment year"
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--run-id",
        required=False,
        default="",
        help="Comps model run ID, mutually exclusive with --year",
    )

    parser.add_argument(
        "--year",
        required=False,
        default="",
        help=(
            "Assessment year to use to derive a comps model run, mutually "
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
    year: str, townships: list[str], pins: list[str], write_github_output: bool
) -> list[str]:
    """
    Given an assessment year, return all the township codes for that year.
    We use assessment year instead of run ID because an assessment year can
    have different final models for different townships.

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
                    WHERE assessment_year = %(year)s
                    ORDER BY meta_township_code
                """,
                {"year": year},
            )
        ]

        if not codes:
            raise ValueError(f"No township codes found for assessment year '{year}'")

    return codes


def get_run_id(run_id: str, year: str) -> str:
    """Return a comps model run ID derived from two optional arguments, a run
    ID and an assessment year.

    If the run ID is non-empty, check to make sure it's valid and return it
    directly if so. Otherwise, query the comps view for the given assessment
    year, and return that year's comps run ID."""
    if run_id:
        run_id_is_valid = bool(
            [
                row[0]
                for row in connect(region_name="us-east-1")
                .cursor()
                .execute(
                    f"""
                    SELECT 1
                    FROM {constants.PINVAL_COMP_TABLE}
                    WHERE run_id = %(run_id)s
                    LIMIT 1
                """,
                    {"run_id": run_id},
                )
            ]
        )
        if not run_id_is_valid:
            raise ValueError(
                f"Run ID {run_id} not found in view "
                f"{constants.PINVAL_COMP_TABLE}. "
                "Double-check to make sure this is a comps run ID."
            )
    else:
        # Get the latest run ID for the assessment year in the comps
        # view. This works because currently there is only ever one
        # comps run per year in that view, but if that ever changes
        # we'll need to adjust this code
        run_ids = [
            row[0]
            for row in connect(region_name="us-east-1")
            .cursor()
            .execute(
                f"""
                    SELECT MAX(run_id)
                    FROM {constants.PINVAL_COMP_TABLE}
                    WHERE assessment_year = %(year)s
                """,
                {"year": year},
            )
        ]

        if not run_ids:
            raise ValueError(f"No comps model run found for year '{year}'")

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
        metadata["assessment_year"], args.township, args.pin, args.write_github_output
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
