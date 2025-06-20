#!/usr/bin/env python3
"""
Generate PINVAL report markdown (and optionally HTML) files for a given model
run‑id. A user may ask for either **one or more explicit PINs** *or* for **all
PINs that belong to a triad** (city, north, south). Exactly one of the two must
be supplied. If the user passes an empty string for either of the --pin or --triad
arguments, the script will ignore that argument.

Examples
--------
Generate two specific PINs:
    $ python3 generate_pinval.py \
          --run-id 2025-02-11-charming-eric \
          --pin 01011000040000 10112040080000

Generate every PIN in the north triad:
    $ python3 generate_pinval.py \
          --run-id 2025-02-11-charming-eric \
          --triad north
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

from constants import RUN_ID_MAP, TRIAD_CHOICES


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
        choices=list(
            RUN_ID_MAP.keys()
        ),  # Temporarily limits run_ids to those in the map
        help="Model run‑ID used by the Athena PINVAL tables (e.g. 2025-02-11-charming-eric)",
    )

    parser.add_argument(
        "--triad",
        choices=TRIAD_CHOICES,
        help="Generate reports for all PINs in this triad (mutually‑exclusive with --pin)",
    )

    parser.add_argument(
        "--pin",
        nargs="*",
        metavar="PIN",
        help="One or more Cook County PINs to generate reports for (mutually‑exclusive with --triad)",
    )

    parser.add_argument(
        "--skip-html",
        action="store_true",
        help="Generate only frontmatter files; skip running the Hugo build step",
    )

    parser.add_argument(
        "--township",
        help=(
            "Restrict triad mode to a single Cook County township code "
            "(two-digit string, e.g. 01, 23). Ignored unless --triad is used."
        ),
    )

    args = parser.parse_args()

    # ── Validation ────────────────────────────────────────────────────────────
    if args.pin == [""]:
        # Cast empty string to null
        args.pin = None

    provided_pin = bool(args.pin)
    provided_triad = bool(args.triad)

    if provided_pin == provided_triad:
        # Either both supplied *or* neither supplied → invalid
        parser.error("Exactly one of --triad or --pin must be provided, but not both.")

    return args


# Declare functions
# ────────────────────────────────────────────────────────────────────────────
def pin_pretty(raw_pin: str) -> str:
    """Convert 14‑digit Cook County PIN → canonical xx‑xx‑xxx‑xxx‑xxxx format."""

    return f"{raw_pin[:2]}-{raw_pin[2:4]}-{raw_pin[4:7]}-{raw_pin[7:10]}-{raw_pin[10:]}"


def _clean_predictors(raw: np.ndarray | list | str) -> list[str]:
    """
    Return a *clean* list of raw predictor column names.
    """

    # Parse numpy arrays and Arrow lists into plain Python lists
    if isinstance(raw, np.ndarray):
        raw = raw.tolist()

    # Clean up existing lists
    if isinstance(raw, list):
        return [str(x).strip() for x in raw if str(x).strip()]

    # Fix list parsing
    txt = str(raw).strip()
    if txt.startswith("[") and txt.endswith("]"):
        txt = txt[1:-1]

    return [p.strip() for p in txt.split(",") if p.strip()]


def build_front_matter(
    df_target_pin: pd.DataFrame,
    df_comps: pd.DataFrame,
    pretty_fn: typing.Callable[[str], str],
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

    # Header
    tp = df_target_pin.iloc[0]  # all cards share the same PIN-level chars
    preds_cleaned: list[str] = _clean_predictors(tp["model_predictor_all_name"])

    front: dict = {
        "layout": "report",
        "title": "Cook County Assessor's Model Value Report (Experimental)",
        "assessment_year": tp["assessment_year"],
        "final_model_run_date": pd.to_datetime(tp["final_model_run_date"]).strftime(
            "%B %d, %Y"
        ),
        "pin": tp["meta_pin"],
        "pin_pretty": pin_pretty(tp["meta_pin"]),
        "pred_pin_final_fmv_round": f"${tp['pred_pin_final_fmv_round']:,.2f}",
        "cards": [],
        "var_labels": {k: pretty_fn(k) for k in preds_cleaned},
    }

    # Exit early if there is a reason the PIN has no comps, in which case we
    # will leave it up to the doc to display the reason for the missing
    # report
    if tp["no_comp_reason"] is not None:
        front["layout"] = "missing"
        front["class_code"] = tp["meta_class"]
        front["class_description"] = tp["class_description"]
        front["parcel_next_assessment_year"] = tp["parcel_next_assessment_year"]
        front["assessment_triad_name"] = tp["assessment_triad_name"]
        front["parcel_triad_name"] = tp["parcel_triad_name"]
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

        # Comps
        comps_list = []
        for _, comp in comps_df.iterrows():
            comp_dict = {
                "comp_num": comp["comp_num"],
                "pin": comp["comp_pin"],
                "pin_pretty": pin_pretty(comp["comp_pin"]),
                "is_subject_pin_sale": comp["is_subject_pin_sale"],
                "sale_price": f"${float(comp['meta_sale_price']):,.0f}",
                "sale_price_short": comp["sale_price_short"],
                "sale_price_per_sq_ft": f"${float(comp['sale_price_per_sq_ft']):,.0f}",
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
        sale_prices = comps_df["meta_sale_price"]
        sqft_prices = comps_df["sale_price_per_sq_ft"]

        comp_summary = {
            "sale_year_range_prefix": (
                "between" if " and " in comps_df["sale_year_range"].iloc[0] else "in"
            ),
            "sale_year_range": comps_df["sale_year_range"].iloc[0]
            if not comps_df.empty
            else "",
            "avg_sale_price": "${:,.0f}".format(sale_prices.mean()),
            "avg_price_per_sqft": "${:,.0f}".format(sqft_prices.mean()),
        }

        # Complete the card
        front["cards"].append(
            {
                "card_num": int(card_num),
                "location": {
                    k: v
                    for k, v in {
                        "property_address": card_df["property_address"],
                        "municipality": card_df.get("loc_tax_municipality_name"),
                        "township": card_df["meta_township_code"],
                        "meta_nbhd_code": card_df["meta_nbhd_code"],
                        "loc_school_elementary_district_name": card_df.get(
                            "school_elementary_district_name"
                        ),
                        "loc_school_secondary_district_name": card_df.get(
                            "school_secondary_district_name"
                        ),
                        "loc_latitude": float(card_df["loc_latitude"]),
                        "loc_longitude": float(card_df["loc_longitude"]),
                    }.items()
                },
                "chars": subject_chars,
                "has_subject_pin_sale": bool(comps_df["is_subject_pin_sale"].any()),
                "pred_card_initial_fmv": "${:,.0f}".format(
                    card_df["pred_card_initial_fmv"]
                ),
                "pred_card_initial_fmv_per_sqft": "${:,.2f}".format(
                    card_df.get(
                        "pred_card_initial_fmv_per_sqft",
                        card_df["pred_card_initial_fmv"] / card_df["char_bldg_sf"],
                    )
                ),
                "comps": comps_list,
                "comp_summary": comp_summary,
                "predictors": preds_cleaned,
            }
        )

    _format_dict_numbers(
        front,
        exclude_keys={
            "loc_latitude",
            "loc_longitude",
            "char_yrblt",
            "has_subject_pin_sale",
            "is_subject_pin_sale",
        },
    )
    return front


def convert_to_builtin_types(obj) -> object:
    """
    Recursively convert numpy types to native Python types in a nested structure.
    This is so the frontmatter doesn't through data type errors when being passed
    to the hugo template.
    """

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
            df[col] = pd.to_numeric(df[col], errors="coerce").astype(dtype)

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


def format_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Format the DataFrame for frontmatter output.
    """
    return (
        df
        # Formate percentage columns
        .pipe(
            lambda d: d.assign(
                **{
                    c: label_percent(d[c])
                    for c in d.filter(regex=r"^acs5_percent").columns
                }
            )
        )
        # Round up all numeric columns except for geo cols needed for mapping
        .pipe(
            lambda d: d.assign(
                **{
                    c: d[c].round(2)
                    for c in d.select_dtypes(include="number").columns
                    if c not in {"loc_latitude", "loc_longitude"}
                }
            )
        )
        # Format $ columns
        .pipe(
            lambda d: d.assign(
                **{
                    c: label_dollar(d[c])
                    for c in d.columns
                    if c == "acs5_median_household_renter_occupied_gross_rent"
                    or c.startswith("acs5_median_income")
                }
            )
        )
    )


def _format_numeric(val):
    """
    123456  -> '123,456'
    1234.5  -> '1,234.5'
    42.0000 -> '42'
    Anything non-numeric is returned unchanged.
    """
    if isinstance(val, (int, np.integer)):
        return f"{val:,}"
    if isinstance(val, (float, np.floating)):
        txt = f"{val:,.2f}".rstrip("0").rstrip(".")
        return txt
    return val


def _format_dict_numbers(obj, exclude_keys: set[str] = None):
    """
    Recursively walk a mapping / sequence and replace every numeric leaf
    with its comma-separated string form.
    """
    exclude_keys = exclude_keys or set()

    if isinstance(obj, dict):
        for k, v in obj.items():
            if k in exclude_keys:
                continue
            obj[k] = _format_dict_numbers(v, exclude_keys)
        return obj

    if isinstance(obj, list):
        return [_format_dict_numbers(v, exclude_keys) for v in obj]

    return _format_numeric(obj)


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

    if args.triad:
        assessment_clauses: list[str] = [
            "run_id = %(run_id)s",
            "assessment_triad = %(triad)s",
        ]
        params_assessment: dict[str, str] = {
            "run_id": args.run_id,
            "triad": args.triad.lower(),
        }
        # Shard by township **only** in the assessment query
        if args.township:
            assessment_clauses.append("meta_township_code = %(township)s")
            params_assessment["township"] = args.township

        where_assessment = " AND ".join(assessment_clauses)
    else:
        pins: list[str] = list(set(args.pin))  # de-dupe
        pin_params = {f"pin{i}": p for i, p in enumerate(pins)}
        placeholders = ",".join(f"%({k})s" for k in pin_params)

        where_assessment = f"run_id = %(run_id)s AND meta_pin IN ({placeholders})"
        params_assessment = {"run_id": args.run_id, **pin_params}

    assessment_sql = f"""
        SELECT *
        FROM pinval.vw_assessment_card
        WHERE {where_assessment}
    """

    print("Querying data from Athena ...")
    df_assessment_all = format_df(
        run_athena_query(cursor, assessment_sql, params_assessment)
    )
    print("Shape of df_assessment_all:", df_assessment_all.shape)

    if df_assessment_all.empty:
        raise ValueError("No assessment rows returned for the given parameters")

    # Get the comps
    comps_run_id = RUN_ID_MAP[args.run_id]

    comps_sql = f"""
        SELECT comp.*
        FROM pinval.vw_comp AS comp
        INNER JOIN (
            SELECT DISTINCT meta_pin
            FROM pinval.vw_assessment_card
            WHERE {where_assessment}
        ) AS card
          ON comp.pin = card.meta_pin
        WHERE comp.run_id = %(run_id_comps)s
    """

    params_comps = {
        "run_id_comps": comps_run_id,
        **params_assessment,
    }

    print("Executing comps query …")
    start_q = time.time()

    df_comps_all = run_athena_query(cursor, comps_sql, params_comps)
    print(f"Comps query finished in {time.time() - start_q:.2f}s")
    df_comps_all = format_df(convert_dtypes(df_comps_all))

    print("Shape of df_comps_all:", df_comps_all.shape)
    if df_comps_all.empty:
        raise ValueError("No comps rows returned for the given parameters — aborting.")

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

    PRESERVE = {"loc_latitude", "loc_longitude"}

    def pretty(k: str) -> str:
        return k if k in PRESERVE else key_map.get(k, k)

    # Declare outputs paths
    md_outdir = project_root / "hugo" / "content" / "pinval-reports"
    md_outdir.mkdir(parents=True, exist_ok=True)

    start_time_dict_groupby = time.time()

    # Group dfs by PIN in dict for theoretically faster access
    df_assessments_by_pin = dict(tuple(df_assessment_all.groupby("meta_pin")))
    df_comps_by_pin = dict(tuple(df_comps_all.groupby("pin")))
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
    for i, (pin, df_target) in enumerate(df_assessments_by_pin.items()):
        if i % 5000 == 0:
            print(f"Processing PIN {i + 1} of {len(df_assessments_by_pin)}")

        md_path = md_outdir / f"{pin}.md"

        df_comps = df_comps_by_pin.get(pin)

        front = build_front_matter(df_target, df_comps, pretty_fn=pretty)
        year = args.run_id[:4]
        front["url"] = f"/{year}/{pin}.html"

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
