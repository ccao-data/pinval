#!/usr/bin/env python3
"""
Generate PINVAL report markdown (and optionally HTML) files for a given comps
model run ID. A user may ask for either **one or more explicit PINs** *or* for **all
PINs that belong to a triad** (city, north, south). Exactly one of the two must
be supplied. If the user passes an empty string for either of the --pin or --triad
arguments, the script will ignore that argument.

Examples
--------
Generate every PIN:
    $ python3 generate_pinval.py \
          --run-id 2025-04-25-fancy-free-billy \

Generate two specific PINs:
    $ python3 generate_pinval.py \
          --run-id 2025-04-25-fancy-free-billy \
          --pin 01011000040000 10112040080000

Generate every PIN in towns 10 and 11:
    $ python3 generate_pinval.py \
          --run-id 2025-04-25-fancy-free-billy \
          --township 10 11
"""

from __future__ import annotations

import argparse
import os
import subprocess as sp
import gc
import time
import typing
from pathlib import Path

import ccao
import numpy as np
import pandas as pd
import orjson
from pyathena import connect
from pyathena.pandas.cursor import PandasCursor

import constants


# Argparse interface
# ─────────────────────────
# ───────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    """Parse command‑line arguments and perform basic validation."""

    parser = argparse.ArgumentParser(
        description="Generate PINVAL report markdown (and optionally HTML) files",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--run-id",
        required=True,
        help="Comps run ID used by the pinval.vw_comps view (e.g. 2025-04-25-fancy-free-billy)",
    )

    parser.add_argument(
        "--pin",
        nargs="*",
        metavar="PIN",
        help=(
            "One or more Cook County PINs to generate reports for. When empty, "
            "generates reports for all PINs in the assessment year"
        ),
    )

    parser.add_argument(
        "--skip-html",
        action="store_true",
        help="Generate only frontmatter files; skip running the Hugo build step",
    )

    parser.add_argument(
        "--township",
        help=(
            "Restrict all-PIN mode to a single Cook County township code "
            "(two-digit string, e.g. 01, 23). Ignored if --pin is used."
        ),
    )

    parser.add_argument(
        "--environment",
        choices=["dev", "prod"],
        default="dev",
        help="Deployment target",
    )

    args = parser.parse_args()

    if args.pin == [""]:
        # Remove empty string
        args.pin = []

    return args


# Declare functions
# ────────────────────────────────────────────────────────────────────────────
def pin_pretty(raw_pin: str) -> str:
    """Format 14‑digit Cook County PIN for display.

    If the PIN ends in '0000' we truncate it and return a 10-digit PIN."""

    truncated_pin = f"{raw_pin[:2]}-{raw_pin[2:4]}-{raw_pin[4:7]}-{raw_pin[7:10]}"
    if raw_pin[10:] == "0000":
        return truncated_pin
    else:
        return f"{truncated_pin}-{raw_pin[10:]}"


def _clean_predictors(raw: np.ndarray | list | str) -> list[str]:
    """
    Return a *clean* list of raw predictor column names.
    """

    # Parse numpy arrays and Arrow lists into plain Python lists
    if isinstance(raw, np.ndarray):
        raw = raw.tolist()

    # Clean up existing lists
    if isinstance(raw, list):
        preds_cleaned = [str(x).strip() for x in raw if str(x).strip()]
    else:
        # Fix list parsing
        txt = str(raw).strip()
        if txt.startswith("[") and txt.endswith("]"):
            txt = txt[1:-1]
        preds_cleaned = [p.strip() for p in txt.split(",") if p.strip()]

    # Add top chars to the front of the list
    top_chars = [
        "meta_nbhd_code",
        "char_class",
        "char_yrblt",
        "char_bldg_sf",
        "char_land_sf",
        "char_beds",
        "char_fbath",
        "char_hbath",
    ]
    preds_cleaned = [c for c in top_chars if c in preds_cleaned] + [
        p for p in preds_cleaned if p not in top_chars
    ]

    return preds_cleaned


def build_front_matter(
    df_target_pin: pd.DataFrame,
    df_comps: pd.DataFrame,
    pretty_fn: typing.Callable[[str], str],
    environment: str,
) -> dict:
    """
    Assemble the front-matter dict for **one PIN**.

    Parameters
    ----------
    df_target_pin : DataFrame
        All assessment rows for this PIN (one per card).
    df_comps : DataFrame
        All comp rows for this PIN (across cards).
    pretty_fn : Callable[[str], str]
        Function that converts a raw model column name → human-readable label.
    """
    special_multi = bool(df_target_pin["is_parcel_small_multicard"].iloc[0])

    # Header
    tp = df_target_pin.iloc[0]  # all cards share the same PIN-level chars
    preds_cleaned: list[str] = _clean_predictors(tp["model_predictor_all_name"])

    # swap out the original sqft column for the combined version
    if special_multi:
        preds_cleaned = [
            "combined_bldg_sf" if p == "char_bldg_sf" else p for p in preds_cleaned
        ]

    front: dict = {
        "layout": "report",
        "title": "Report: How Did the Assessor's Model Estimate My Home Value?",
        "assessment_year": tp["assessment_year"],
        "final_model_run_date": pd.to_datetime(tp["final_model_run_date"]).strftime(
            "%B %d, %Y"
        ),
        "pin": tp["meta_pin"],
        "pin_pretty": pin_pretty(tp["meta_pin"]),
        "pred_pin_final_fmv_round": tp["pred_pin_final_fmv_round"],
        "meta_pin_num_cards": tp["meta_pin_num_cards"],
        "cards": [],
        "var_labels": {k: pretty_fn(k) for k in preds_cleaned},
        "special_case_multi_card": special_multi,
        "environment": environment,
    }

    # Exit early if this PIN is ineligible for a report, in which case we
    # will pass the doc some info to help explain why the parcel is ineligible
    if not tp["is_report_eligible"]:
        front["layout"] = "ineligible"
        for attr in [
            "reason_report_ineligible",
            "assessment_triad_name",
            "char_class",
            "char_class_desc",
            "meta_triad_name",
        ]:
            front[attr] = tp[attr]
        return front

    # Per card
    for card_num, card_df in df_target_pin.groupby("meta_card_num"):
        card_df = card_df.iloc[0]

        comps_df = (
            df_comps[df_comps["card"] == card_num]
            .sort_values("comp_num")
            .reset_index(drop=True)
        )

        # Add all of the feature columns to the card
        subject_chars = {
            pred: card_df[pred] for pred in preds_cleaned if pred in card_df
        }
        # Keep the original building-SF for the subject only, while the
        # comps and all downstream outputs use the combined building sqft
        if special_multi and "char_bldg_sf" in card_df:
            subject_chars["char_bldg_sf"] = card_df["char_bldg_sf"]

        # Comps
        comps_list = []
        for _, comp in comps_df.iterrows():
            comp_dict = {
                "comp_num": comp["comp_num"],
                "pin": comp["comp_pin"],
                "pin_pretty": pin_pretty(comp["comp_pin"]),
                "is_subject_pin_sale": comp["is_subject_pin_sale"],
                "sale_price": comp["meta_sale_price"],
                "sale_price_short": comp["sale_price_short"],
                "sale_price_per_sq_ft": comp["sale_price_per_sq_ft"],
                "sale_date": comp["sale_month_year"],
                "document_num": comp["comp_document_num"],
                "property_address": comp["property_address"],
                "meta_nbhd_code": comp["meta_nbhd_code"],
            }

            # Make preds human-readable
            for pred in preds_cleaned:
                if pred not in comp_dict and pred in comp:
                    comp_dict[pred] = comp[pred]

            comps_list.append(comp_dict)

        # Comp summary
        comp_summary = {
            "sale_year_range_prefix": (
                "between" if " and " in comps_df["sale_year_range"].iloc[0] else "in"
            ),
            "sale_year_range": comps_df["sale_year_range"].iloc[0]
            if not comps_df.empty
            else "",
            "avg_sale_price": comps_df["comps_avg_sale_price"].iloc[0],
            "avg_price_per_sqft": comps_df["comps_avg_price_per_sqft"].iloc[0],
        }
        # Complete the card
        front["cards"].append(
            {
                "pin_pretty": pin_pretty(tp["meta_pin"]),
                "card_num": int(card_num),
                "char_class_detailed": card_df["char_class_detailed"],
                "location": {
                    k: v
                    for k, v in {
                        "property_address": card_df["property_address"],
                        "municipality": card_df["loc_property_city"],
                        "township": tp["meta_township_name"],
                        "meta_nbhd_code": card_df["meta_nbhd_code"],
                        "loc_school_elementary_district_name": card_df[
                            "school_elementary_district_name"
                        ],
                        "loc_school_secondary_district_name": card_df[
                            "school_secondary_district_name"
                        ],
                        "loc_latitude": card_df["loc_latitude"],
                        "loc_longitude": card_df["loc_longitude"],
                    }.items()
                },
                "chars": subject_chars,
                "has_subject_pin_sale": bool(comps_df["is_subject_pin_sale"].any()),
                "pred_card_initial_fmv": card_df["pred_card_initial_fmv"],
                "pred_card_initial_fmv_per_sqft": card_df[
                    "pred_card_initial_fmv_per_sqft"
                ],
                "comps": comps_list,
                "comp_summary": comp_summary,
                "predictors": preds_cleaned,
            }
        )

    return front


def convert_to_builtin_types(obj) -> object:
    """
    Recursively convert numpy types to native Python types in a nested structure.
    This is so the frontmatter doesn't through data type errors when being passed
    to the hugo template.
    """

    # Standardize NA string representations
    if isinstance(obj, str) and obj in {"nan"}:
        return ""

    if isinstance(obj, dict):
        return {k: convert_to_builtin_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_builtin_types(v) for v in obj]
    elif isinstance(obj, np.ndarray):
        if obj.size == 1:
            # Collapse 1-element arrays to their scalar value,
            # this is currently here because of municipality
            # behaviour in vw_pin10_location
            return convert_to_builtin_types(obj.item())
        else:
            # keep multi-element arrays as lists
            return [convert_to_builtin_types(v) for v in obj.tolist()]
    elif obj is pd.NA:  # pandas NA scalar
        return ""
    # Wrap NaN in quotes, otherwise the .nan breaks html map rendering
    elif isinstance(obj, (float, np.floating)) and np.isnan(obj):
        return ""
    elif isinstance(obj, np.generic):
        return obj.item()
    return obj


def convert_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cast expected columns in df to appropriate dtypes before building dicts.
    This avoids manual casting during dict construction later.
    """

    # Define conversions
    dtype_conversions = {
        "comp_num": "Int64",
        "char_yrblt": "Int64",
        "char_beds": "Int64",
        "char_fbath": "Int64",
        "char_hbath": "Int64",
        "loc_latitude": "float",
        "loc_longitude": "float",
        "is_subject_pin_sale": "boolean",
    }

    for col, dtype in dtype_conversions.items():
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="raise").astype(dtype)

    return df


def write_json(front_dict: dict, outfile: str | Path) -> None:
    """Write the frontmatter dict to a JSON file."""
    front_dict = convert_to_builtin_types(front_dict)

    json_bytes: bytes = orjson.dumps(
        front_dict,
        option=orjson.OPT_INDENT_2,
    )
    Path(outfile).write_text(json_bytes.decode("utf-8") + "\n", encoding="utf-8")


def label_percent(s: pd.Series) -> pd.Series:
    """0.123 → '12%' (handles NaNs)."""
    return s.mul(100).round(0).astype("Int64").astype(str).str.replace("<NA>", "") + "%"


def label_dollar(s: pd.Series) -> pd.Series:
    """45000 → '$45,000' (handles NaNs)."""
    return s.apply(lambda x: f"${x:,.0f}" if pd.notna(x) else "")


def format_df(df: pd.DataFrame, chars_recode=False) -> pd.DataFrame:
    """
    Format the DataFrame for frontmatter output.
    """

    # Columns like year shouldn't be formatted with commas,
    # so we preserve them as integers
    INT_COLS = {
        "time_sale_year",
        "time_sale_day",
        "acs5_median_household_total_occupied_year_built",
        "char_yrblt",
        "meta_pin_num_cards",
    }

    # Columns that should be preserved as numeric
    NUMERIC_PRESERVE = {"loc_latitude", "loc_longitude"}

    def _fmt(x):
        if isinstance(x, (int, np.integer)):
            return f"{x:,}"
        if isinstance(x, (float, np.floating)):
            txt = f"{x:,.2f}".rstrip("0").rstrip(".")
            return txt
        return x

    # Recode character columns to human-readable values
    if chars_recode:
        chars_to_recode = [
            col for col in df.columns if col.startswith("char_") and col != "char_apts"
        ]

        df = ccao.vars_recode(
            data=df.copy(),
            cols=chars_to_recode,
            code_type="long",
            as_factor=False,
            dictionary=ccao.vars_dict,
        )

    # Generate comps summary stats needed for frontmatter
    if "meta_sale_price" in df.columns:
        df["comps_avg_sale_price"] = df.groupby(["pin", "card"])[
            "meta_sale_price"
        ].transform("mean")
    if "sale_price_per_sq_ft" in df.columns:
        df["comps_avg_price_per_sqft"] = df.groupby(["pin", "card"])[
            "sale_price_per_sq_ft"
        ].transform("mean")

    formatted_df = (
        # Convert data to INT for columns that should be integers (year, etc)
        df.pipe(
            lambda d: d.assign(
                **{
                    col: pd.to_numeric(d[col], errors="raise").astype("Int64")
                    for col in INT_COLS
                    if col in d.columns
                }
            )
        )
        # Round lat/long to 5 decimal places, a balance between precision and
        # readability
        .pipe(
            lambda d: d.assign(
                loc_latitude=d["loc_latitude"].apply(round, ndigits=5),
                loc_longitude=d["loc_longitude"].apply(round, ndigits=5),
            )
        )
        # Format percentage columns
        .pipe(
            lambda d: d.assign(
                **{
                    c: label_percent(d[c])
                    for c in d.filter(regex=r"^acs5_percent").columns
                }
            )
        )
        # Format $ columns
        .pipe(
            lambda d: d.assign(
                **{
                    c: label_dollar(d[c])
                    for c in d.columns
                    if c
                    in {
                        "comps_avg_sale_price",
                        "comps_avg_price_per_sqft",
                        "meta_sale_price",
                        "sale_price_per_sq_ft",
                        "acs5_median_household_renter_occupied_gross_rent",
                        "pred_pin_final_fmv_round",
                        "pred_card_initial_fmv",
                        "pred_card_initial_fmv_per_sqft",
                    }
                    or c.startswith("acs5_median_income")
                }
            )
        )
        # Generic numerics → comma-separated strings (except preserved cols)
        .pipe(
            lambda d: d.assign(
                **{
                    c: d[c].apply(_fmt)
                    for c in d.select_dtypes(include="number").columns
                    if c not in NUMERIC_PRESERVE and c not in INT_COLS
                }
            )
        )
    )

    return formatted_df


def run_athena_query(cursor, sql: str, params: dict = None) -> pd.DataFrame:
    cursor.execute(sql, parameters=params)
    return cursor.as_pandas()


def main() -> None:
    args = parse_args()

    project_root = Path(sp.getoutput("git rev-parse --show-toplevel"))
    os.chdir(project_root)

    # Athena connection (one per run)
    cursor = connect(
        # We add '+ "/"' to the end of the line below because enabling unload
        # requires that the staging directory end with a slash. Add rstrip because
        # the missing '/' seems to depend on different environments
        s3_staging_dir=os.getenv("AWS_ATHENA_S3_STAGING_DIR").rstrip("/") + "/",
        region_name=os.getenv("AWS_REGION"),
        cursor_class=PandasCursor,
    ).cursor(unload=True)

    # Extract assessment year for model run, which we'll use to determine
    # the output path
    assessment_year_df = run_athena_query(
        cursor,
        "SELECT assessment_year FROM model.metadata WHERE run_id = %(run_id)s",
        {"run_id": args.run_id},
    )
    if assessment_year_df.empty:
        raise ValueError(f"No model metadata found for model run {args.run_id}")

    assessment_year = assessment_year_df.iloc[0]["assessment_year"]

    assessment_clauses = ["assessment_year = %(assessment_year)s"]
    params_assessment = {"assessment_year": assessment_year}

    # Shard by township **only** in the assessment query
    if args.township:
        assessment_clauses.append("meta_township_code = %(township)s")
        params_assessment["township"] = args.township

    if args.pin:
        pins: list[str] = list(set(args.pin))  # de-dupe
        pin_params = {f"pin{i}": p for i, p in enumerate(pins)}
        placeholders = ",".join(f"%({k})s" for k in pin_params)

        assessment_clauses.append(f"meta_pin IN ({placeholders})")
        params_assessment = {**params_assessment, **pin_params}

    where_assessment = " AND ".join(assessment_clauses)

    assessment_sql = f"""
        SELECT *
        FROM {constants.PINVAL_ASSESSMENT_CARD_TABLE}
        WHERE {where_assessment}
    """

    print("Querying data from Athena ...")
    df_assessment_all = format_df(
        run_athena_query(cursor, assessment_sql, params_assessment),
        chars_recode=False,
    )
    print("Shape of df_assessment_all:", df_assessment_all.shape)

    if df_assessment_all.empty:
        raise ValueError(
            f"No assessment rows returned for the following params: {params_assessment}"
        )

    comps_sql = f"""
        SELECT comp.*
        FROM {constants.PINVAL_COMP_TABLE} AS comp
        INNER JOIN (
            SELECT DISTINCT meta_pin
            FROM {constants.PINVAL_ASSESSMENT_CARD_TABLE}
            WHERE {where_assessment}
        ) AS card
          ON comp.pin = card.meta_pin
        WHERE comp.run_id = %(run_id)s
    """

    params_comps = {
        "run_id": args.run_id,
        **params_assessment,
    }

    print("Executing comps query …")
    start_q = time.time()

    df_comps_all = run_athena_query(cursor, comps_sql, params_comps)
    print(f"Comps query finished in {time.time() - start_q:.2f}s")
    print("Shape of df_comps_all:", df_comps_all.shape)
    # Transform values in character columns to human-readable values.
    # This is already done in the assessment query, so we only need to do it here.
    # We don't need to do it if the comps are empty, in which case we're
    # probably working on a township that was not reassessed
    if not df_comps_all.empty:
        df_comps_all = format_df(convert_dtypes(df_comps_all), chars_recode=True)

    # Crosswalk for making column names human-readable
    model_vars: list[str] = ccao.vars_dict["var_name_model"].tolist()

    pretty_vars: list[str] = ccao.vars_rename(
        data=model_vars,
        names_from="model",
        names_to="pretty",
        output_type="vector",
        dictionary=ccao.vars_dict,
    )

    key_map: dict[str, str] = dict(zip(model_vars, pretty_vars))
    # Manually define mapping for the "Combined Bldg. SF" label, which is not
    # part of `ccao.vars_dict`
    key_map["combined_bldg_sf"] = "Combined Bldg. Sq. Ft."

    PRESERVE = {"loc_latitude", "loc_longitude"}

    def pretty(k: str) -> str:
        return k if k in PRESERVE else key_map.get(k, k)

    # Declare outputs paths
    md_outdir = project_root / "hugo" / "content" / "pinval-reports"
    md_outdir.mkdir(parents=True, exist_ok=True)

    start_time_dict_groupby = time.time()

    # Group dfs by PIN in dict for theoretically faster access
    df_assessments_by_pin = df_assessment_all.groupby("meta_pin")
    df_comps_by_pin = (
        {} if df_comps_all.empty else dict(tuple(df_comps_all.groupby("pin")))
    )
    end_time_dict_groupby = time.time()

    del df_assessment_all
    del df_comps_all
    gc.collect()
    print(
        f"Grouping by PIN took {end_time_dict_groupby - start_time_dict_groupby:.2f} seconds"
    )

    # Iterate over each unique PIN and output frontmatter
    print("Iterating pins to generate frontmatter")
    start_time = time.time()
    pin_groups = df_assessments_by_pin.groups
    for i, pin in enumerate(pin_groups):
        if i % 5000 == 0:
            print(f"Processing PIN {i + 1} of {len(pin_groups)}")

        # Use get_group to lower memory use when iterating grouped DF
        df_target = df_assessments_by_pin.get_group(pin)

        md_path = md_outdir / f"{pin}.md"

        df_comps = df_comps_by_pin.get(pin)

        front = build_front_matter(
            df_target,
            df_comps,
            pretty_fn=pretty,
            environment=args.environment,
        )
        front["url"] = f"/{assessment_year}/{pin}.html"

        write_json(front, md_path)

    elapsed_time = time.time() - start_time
    print(
        f"✓ Completed generating frontmatter for {len(df_assessments_by_pin)} PINs in {elapsed_time:.4f} seconds."
    )

    # ------------------------------------------------------------------
    # Optional Hugo build
    # ------------------------------------------------------------------

    if not args.skip_html:
        # Clean up memory before running Hugo, this prevents github runners
        # from running out of memory
        for _v in (
            df_assessments_by_pin,
            df_comps_by_pin,
        ):
            del _v
        gc.collect()

        # Generate the HTML files using Hugo
        print("Running Hugo …")
        proc = sp.run(["hugo", "--minify"], cwd=project_root / "hugo", text=True)
        if proc.returncode != 0:
            raise RuntimeError("Hugo build failed.")

        # Remove markdown files now that HTML is baked.
        for md_file in md_outdir.glob("*.md"):
            md_file.unlink(missing_ok=True)
        print("✓ Hugo build complete — markdown cleaned up.")


if __name__ == "__main__":
    main()
