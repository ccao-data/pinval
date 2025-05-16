#!/usr/bin/env python3
"""
Generate PINVAL report markdown (and optionally HTML) files for a given model
run‑id. A user may ask for either **one or more explicit PINs** *or* for **all
PINs that belong to a triad** (city, north, south). Exactly one of the two must
be supplied.

Examples
--------
Generate two specific PINs:
    $ ./scripts/generate_pinval.py \
          --run-id 2025-02-11-charming-eric \
          --pin 01011000040000 10112040080000

Generate every PIN in the city triad:
    $ ./scripts/generate_pinval.py \
          --run-id 2025-02-11-charming-eric \
          --triad city

# Notes
to add packages to config
- uv pip install

to get a proper env in ipython: python -m IPython
"""

from __future__ import annotations

import argparse
import time
import os
import shutil
import subprocess as sp
import sys
from pathlib import Path
from typing import Iterable, List, Sequence

import ccao
import numpy as np
import pandas as pd
import yaml
from pyathena import connect
from pyathena.pandas.cursor import PandasCursor
from pyathena.pandas.util import as_pandas

# Argparse interface
# ────────────────────────────────────────────────────────────────────────────
TRIAD_CHOICES: tuple[str, ...] = ("city", "north", "south")


def parse_args() -> argparse.Namespace:  # noqa: D401  (We *return* Namespace)
    """Parse command‑line arguments and perform basic validation."""

    parser = argparse.ArgumentParser(
        description="Generate PINVAL report markdown (and optionally HTML) files",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--run-id",
        required=True,
        help="Model run‑ID used by the Athena PINVAL tables (e.g. 2025-02-11-charming-eric)",
    )

    parser.add_argument(
        # TODO: Resolve this mismatch: https://github.com/ccao-data/data-architecture/pull/814#event-17638024529
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
        help="Generate only markdown files; skip running the Hugo build step",
    )

    args = parser.parse_args()

    # ── Validation ────────────────────────────────────────────────────────────
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


def _clean_predictors(raw) -> list[str]:
    """Return clean list of predictor column names from *raw* (string / list)."""

    if pd.isna(raw):
        return []

    if isinstance(raw, list):
        return [str(x).strip() for x in raw if str(x).strip()]

    txt = str(raw).strip()
    if txt.startswith("[") and txt.endswith("]"):
        txt = txt[1:-1]
    return [p.strip() for p in txt.split(",") if p.strip()]


def build_front_matter(df_target_pin: pd.DataFrame, df_comps: pd.DataFrame) -> dict:
    """
    Assemble the front‑matter structure for *every* card that exists in the
    two data sets 
    """
    # Assume exactly one PIN row in df_target_pin
    tp = df_target_pin.iloc[0]

    front = {
        "layout": "report",
        "title": "Cook County Assessor's Model Value Report (Experimental)",
        "assessment_year": str(int(tp["meta_year"]) + 1),
        "final_model_run_date": pd.to_datetime(tp["final_model_run_date"]).strftime("%B %d, %Y"),
        "pin": tp["meta_pin"],
        "pin_pretty": pin_pretty(tp["meta_pin"]),
        "pred_pin_final_fmv_round": tp["pred_pin_final_fmv_round"],
        "cards": [],
    }

    # Iterate through each card number present in df_target_pin 
    for card_num, card_df in df_target_pin.groupby("meta_card_num"):
        card_df = card_df.iloc[0]

        # Pull matching comps for this card – keep display order
        comps_df = (
            df_comps[df_comps["card"] == card_num]
            .sort_values("comp_num")
            .reset_index(drop=True)
        )

        # Make columns human readable
        predictors = _clean_predictors(card_df["model_predictor_all_name"])

        # Comps List
        comps_list = []

        subject_chars = {
            pred: card_df[pred]
            for pred in predictors
            if pred in card_df
        }

        
        for _, comp in comps_df.iterrows():
            # TODO: Probably a way to collapse this all, but also need to re-order this
            comp_dict = {
                "comp_num": comp["comp_num"],
                "pin": comp["comp_pin"],
                "pin_pretty": comp["comp_pin"],
                "is_subject_pin_sale": comp["is_subject_pin_sale"],
                "sale_price": comp["meta_sale_price"],
                "sale_price_short": comp["sale_price_short"],
                "sale_price_per_sq_ft": comp["sale_price_per_sq_ft"],
                "sale_date": comp["sale_month_year"],
                "document_num": comp["comp_document_num"],
                "property_address": comp["property_address"],
                "meta_nbhd_code": comp["meta_nbhd_code"],
            }
            for pred in predictors:
                # Keep preds unique
                if pred not in comp_dict and pred in comp:
                    comp_dict[pred] = comp[pred]

            comps_list.append(comp_dict)

        # Comp summary information, could potentially refactor this into dbt model
        sale_prices = comps_df["meta_sale_price"].dropna()
        sqft_prices = comps_df["sale_price_per_sq_ft"].dropna()

        comp_summary = {
            "sale_year_range_prefix": "between" if " and " in comps_df["sale_year_range"].iloc[0] else "in",
            "sale_year_range": comps_df["sale_year_range"].iloc[0],
            #TODO: Factor out these means into sql view
            "avg_sale_price": "${:,.2f}".format(sale_prices.mean()),
            "avg_price_per_sqft": "${:,.2f}".format(sqft_prices.mean()),
        }

        # Build card dict
        front["cards"].append(
            {
                "card_num": int(card_num),
                "location": {
                    "property_address": card_df["property_address"],
                    "municipality": card_df.get("meta_municipality", None),
                    "township": card_df["meta_township_code"],
                    "meta_nbhd_code": card_df["meta_nbhd_code"],
                    "loc_school_elementary_district_name": card_df[
                        "loc_school_elementary_district_name"
                    ],
                    "loc_school_secondary_district_name": card_df[
                        "loc_school_secondary_district_name"
                    ],
                    "loc_latitude": float(card_df["loc_latitude"]),
                    "loc_longitude": float(card_df["loc_longitude"]),
                },
                "chars": subject_chars,
                "has_subject_pin_sale": bool(comps_df["is_subject_pin_sale"].any()),
                "pred_card_initial_fmv": "${:,.2f}".format(card_df["pred_card_initial_fmv"]),
                "pred_card_initial_fmv_per_sqft": "${:,.2f}".format(
                    card_df.get(
                        "pred_card_initial_fmv_per_sqft",
                        card_df["pred_card_initial_fmv"] / card_df["char_bldg_sf"]
                    )
                ),
                "comps": comps_list,
                "comp_summary": comp_summary,
                "predictors": predictors,
            }
        )


    return front


def convert_to_builtin_types(obj):
    """
    Recursively convert numpy types to native Python types in a nested structure.
    This is so the frontmatter doesn't through data type errors when being passed
    to the hugo template.
    """
    if isinstance(obj, dict):
        return {k: convert_to_builtin_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_builtin_types(v) for v in obj]
    elif isinstance(obj, (float, np.floating)) and np.isnan(obj):
        return "nan"  # Wrap NaN in quotes, otherwise the .nan breaks html map rendering
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


def make_human_readable(front_dict: dict, vars_dict: pd.DataFrame) -> dict:
    """
    Rename machine‑readable field names to pretty labels
    """
    key_map = dict(zip(vars_dict["var_name_model"], vars_dict["var_name_pretty"]))

    # Keys that should not be renamed
    preserve_keys = {
        # loc_fields are called by the Leaflet map, could change the ref there or here
        "loc_latitude", "loc_longitude",
        "Latitude", "Longitude",
        #TODO: Figure out why this doesn't render in html if we remove it
        "meta_nbhd_code",
    }

    def rename_dict_keys(d: dict) -> dict:
        return {
            k if k in preserve_keys else key_map.get(k, k): v
            for k, v in d.items()
        }

    def rename_list_values(values: list[str]) -> list[str]:
        return [
            v if v in preserve_keys else key_map.get(v, v)
            for v in values
        ]

    for card in front_dict.get("cards", []):
        # rename keys inside "location"
        if "location" in card:
            card["location"] = rename_dict_keys(card["location"])

        # rename keys inside each comp
        if "comps" in card:
            card["comps"] = [rename_dict_keys(comp) for comp in card["comps"]]

        # rename keys inside the subject‑property characteristics
        if "chars" in card:
            card["chars"] = rename_dict_keys(card["chars"])

        # rename predictor names (list of strings)
        if "predictors" in card:
            card["predictors"] = rename_list_values(card["predictors"])

    return front_dict


def write_md(front_dict: dict, outfile: str | Path) -> None:
    #Writes the front matter to a markdown file.
    
    # Use C-accelerated YAML dumper if available
    try:
        dumper = yaml.CSafeDumper
    except AttributeError:
        dumper = yaml.SafeDumper  # fallback if C bindings not available

    front_dict = convert_to_builtin_types(front_dict)

    yaml_block = yaml.dump(front_dict, Dumper=dumper, sort_keys=False, allow_unicode=False)

    md_text = f"---\n{yaml_block}---\n"

    Path(outfile).write_text(md_text, encoding="utf8")


def label_percent(s: pd.Series) -> pd.Series:
    """0.123 → '12%' (handles NaNs)."""
    return s.mul(100).round(0).astype('Int64').astype(str).str.replace('<NA>', 'NA') + '%'

def label_dollar(s: pd.Series) -> pd.Series:
    """45000 → '$45,000' (handles NaNs)."""
    return s.apply(lambda x: f"${x:,.0f}" if pd.notna(x) else "NA")

def format_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Re‑creates the dplyr pipeline:
      1. Percent‑format cols that start with 'acs5_percent'
      2. Round every remaining numeric col to 2 decimals
      3. Dollar‑format renter‑gross‑rent and all median‑income cols
    Returns a *new* DataFrame; original is untouched.
    """

    return (
        df
        # 1 ─ percent columns ----------------------------------------------------
        .pipe(lambda d: d.assign(**{
            c: label_percent(d[c])
            for c in d.filter(regex=r'^acs5_percent').columns
        }))
        # 2 ─ round remaining numerics ------------------------------------------
        .pipe(lambda d: d.assign(**{
            c: d[c].round(2)
            for c in d.select_dtypes(include='number').columns
            if c not in {"loc_latitude", "loc_longitude"}   # skip the two geo cols
        }))
        # 3 ─ dollar columns -----------------------------------------------------
        .pipe(lambda d: d.assign(**{
            c: label_dollar(d[c])
            for c in (
                ['acs5_median_household_renter_occupied_gross_rent'] +
                [c for c in d.columns if c.startswith('acs5_median_income')]
            )
        }))
    )


def run_athena_query(cursor, sql: str) -> pd.DataFrame:
    cursor.execute(sql)
    return as_pandas(cursor)


def main() -> None:
    args = parse_args()

    project_root = Path(
        sp.getoutput("git rev-parse --show-toplevel")
    )
    os.chdir(project_root)

    # Athena connection (one per run)
    cursor = connect(
        # We add '+ "/"' to the end of the line below because enabling unload
        # requires that the staging directory end with a slash
        s3_staging_dir=os.getenv("AWS_ATHENA_S3_STAGING_DIR") + "/",
        region_name=os.getenv("AWS_REGION"),
        cursor_class=PandasCursor,
        #TODO: figure out why unload=true is not working, might be bc it is view and not a table
    ).cursor()

    if args.triad:
        where_assessment = f"run_id = '{args.run_id}' AND assessment_triad = '{args.triad.lower()}'"
    else:
        pins: list[str] = list(dict.fromkeys(args.pin))  # de‑dupe
        pins_quoted = ",".join(f"'{p}'" for p in pins)
        where_assessment = f"run_id = '{args.run_id}' AND meta_pin IN ({pins_quoted})"

    print("querying data from athena")

    assessment_sql = f"""
        SELECT *
        FROM z_ci_811_improve_pinval_models_for_hugo_frontmatter_integration_pinval.vw_assessment_card
        WHERE {where_assessment}
        LIMIT 1000
    """

    df_assessment_all = format_df(run_athena_query(cursor, assessment_sql))
    print("Shape of df_assessment_all:", df_assessment_all.shape)

    if df_assessment_all.empty:
        sys.exit("No assessment rows returned for the given parameters — aborting.")

    all_pins: list[str] = df_assessment_all["meta_pin"].unique().tolist()

    # Get the comps for all the pins
    #TODO: This probably needs a refactor since a 500k list filter operation
    # might break athena
    pins_quoted_for_comps = ",".join(f"'{p}'" for p in all_pins)
    comps_sql = f"""
        SELECT *
        FROM z_ci_811_improve_pinval_models_for_hugo_frontmatter_integration_pinval.vw_comp
        WHERE run_id = '2025-04-25-fancy-free-billy' AND pin IN ({pins_quoted_for_comps})
    """
    # WHERE run_id = '{args.run_id}' AND pin IN ({pins_quoted_for_comps})
    # WHERE run_id =  AND pin IN ({pins_quoted_for_comps})

    df_comps_all = run_athena_query(cursor, comps_sql)
    df_comps_all["pin_pretty"] = df_comps_all["pin"].apply(pin_pretty)
    df_comps_all = format_df(convert_dtypes(df_comps_all))

    print("Shape of df_comps_all:", df_comps_all.shape)
    if df_comps_all.empty:
        sys.exit("No comps rows returned for the given parameters — aborting.")

    # Temp solution for human-readable transformation
    vars_dict = ccao.vars_dict

    # Declare outputs paths
    md_outdir = project_root / "content" / "pinval-reports"
    md_outdir.mkdir(parents=True, exist_ok=True)

    # Group dfs by PIN in dict for theoretically faster access
    df_assessments_by_pin = dict(tuple(df_assessment_all.groupby("meta_pin")))
    df_comps_by_pin = dict(tuple(df_comps_all.groupby("pin")))

    # Iterate over each unique PIN and output frontmatter
    print("Iterating pins to generate frontmatter")
    start_time = time.time()
    for pin in all_pins:
        run_id_pin_id = f"{args.run_id}__{pin}"
        md_path = md_outdir / f"{run_id_pin_id}.md"

        df_target = df_assessments_by_pin.get(pin)
        df_comps = df_comps_by_pin.get(pin)

        front = build_front_matter(df_target, df_comps)
        front = make_human_readable(front, vars_dict)

        write_md(front, md_path)
    elapsed_time = time.time() - start_time  # End timer        
    print(f"✓ Completed generating frontmatter for {len(all_pins)} PINs in {elapsed_time:.4f} seconds.")

    # ------------------------------------------------------------------
    # Optional Hugo build
    # ------------------------------------------------------------------

    if not args.skip_html:
        print("Running Hugo …")
        proc = sp.run(["hugo", "--minify"], cwd=project_root, text=True)
        if proc.returncode != 0:
            sys.exit("Hugo build failed.")

        # Remove markdown files now that HTML is baked.
        for md_file in md_outdir.glob("*.md"):
            md_file.unlink(missing_ok=True)
        print("✓ Hugo build complete — markdown cleaned up.")


# Main function
if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted by user", file=sys.stderr)
        sys.exit(130)