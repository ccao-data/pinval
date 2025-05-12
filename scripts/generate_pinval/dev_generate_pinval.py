#!/usr/bin/env python3
"""
Script to generate PINVAL reports using a specified model
run ID and either a triad or list of PINs.

# Notes
to add packages to config
- uv pip install
"""

import argparse
import subprocess as sp
import sys
import os
from pathlib import Path
import pyarrow
import ast
import pandas as pd
import numpy as np
import yaml
from pyathena import connect
from pyathena.pandas.util import as_pandas
from pyathena.pandas.cursor import PandasCursor

root = sp.getoutput("git rev-parse --show-toplevel")
os.chdir(root)

# Connect to Athena
cursor = connect(
    # We add '+ "/"' to the end of the line below because enabling unload
    # requires that the staging directory end with a slash
    s3_staging_dir=os.getenv("AWS_ATHENA_S3_STAGING_DIR") + "/",
    region_name=os.getenv("AWS_REGION"),
    cursor_class=PandasCursor,
    #TODO: figure out why unload=true is not working, might be bc it is view and not a table
).cursor()

RUN_ID = "2025-02-11-charming-eric"
PIN = "01011000040000"

COMPS_QUERY = f"""
SELECT *
    FROM z_ci_811_improve_pinval_models_for_hugo_frontmatter_integration_pinval.vw_comp
WHERE run_id = '{RUN_ID}'
and pin = '{PIN}'
"""

TARGET_PIN_QUERY = f"""
SELECT *
    FROM pinval.vw_assessment_card
WHERE
    meta_pin = '{PIN}'
    and run_id = '{RUN_ID}'
"""

# Execute query and return as pandas df
cursor.execute(COMPS_QUERY)
df_comps = as_pandas(cursor)

cursor.execute(TARGET_PIN_QUERY)
df_target_pin = as_pandas(cursor)





# - - - - - - - - - 
# TESTING MAIN LOGIC
# - - - - - - - - - 
def pin_pretty(raw_pin: str) -> str:
    """
    Convert a 14-digit Cook County PIN such as 14331000240000 → 14-33-100-024-0000
    """
    return f"{raw_pin[:2]}-{raw_pin[2:4]}-{raw_pin[4:7]}-{raw_pin[7:10]}-{raw_pin[10:]}"



# ---------- core builder ----------------------------------------------------
def build_front_matter(df_target_pin: pd.DataFrame, df_comps: pd.DataFrame) -> dict:
    """
    Assemble the front‑matter structure for *every* card that exists in the
    two data sets (usually only card 1 for residential).
    """
    # Assume exactly one PIN row in df_target_pin
    tp = df_target_pin.iloc[0]

    front = {
        "layout": "report",
        "title": "Cook County Assessor's Model Value Report (Experimental)",
        # Assessment year is usually meta_year + 1
        "assessment_year": str(int(tp["meta_year"]) + 1),
        "final_model_run_date": pd.to_datetime(tp["final_model_run_date"]).strftime("%B %d, %Y"),
        "pin": tp["meta_pin"],
        "pin_pretty": pin_pretty(tp["meta_pin"]),
        "pred_pin_final_fmv_round": tp["pred_pin_final_fmv_round"],
        "cards": [],
    }

    # --- iterate through each card number present in df_target_pin ----------
    for card_num, card_df in df_target_pin.groupby("meta_card_num"):
        card_df = card_df.iloc[0]  # there is only one physical row per card

        # pull matching comps for this card – keep display order
        comps_df = (
            df_comps[df_comps["card"] == card_num]
            .sort_values("comp_num")
            .reset_index(drop=True)
        )

        # ------------- comps list -------------------------------------------
        comps_list = []
        for _, comp in comps_df.iterrows():
            comps_list.append(
                {
                    "comp_num": int(comp["comp_num"]),
                    "pin": comp["comp_pin"],
                    "pin_pretty": pin_pretty(comp["comp_pin"]),
                    "is_subject_pin_sale": bool(comp["is_subject_pin_sale"]),
                    "sale_price": comp["meta_sale_price"],
                    "sale_price_short": comp["sale_price_short"],
                    "sale_price_per_sq_ft": comp["sale_price_per_sq_ft"],
                    "sale_date": comp["sale_month_year"],
                    "document_num": comp["comp_document_num"],
                    "property_address": comp["property_address"],
                    # main physical characteristics -------------
                    "char_class": comp["char_class"],
                    "char_yrblt": int(comp["char_yrblt"]),
                    "char_bldg_sf": comp["char_bldg_sf"],
                    "char_land_sf": comp["char_land_sf"],
                    "char_beds": int(comp["char_beds"]),
                    "char_fbath": int(comp["char_fbath"]),
                    "char_hbath": int(comp["char_hbath"]),
                    "meta_nbhd_code": comp["meta_nbhd_code"],
                    "loc_latitude": float(comp["loc_latitude"]),
                    "loc_longitude": float(comp["loc_longitude"]),
                }
            )

        # ------------- comp‑summary block ----------------------------------
        # Use only those comps with a valid numeric sale_price
        sale_prices = comps_df["meta_sale_price"].dropna()
        sqft_prices = comps_df["sale_price_per_sq_ft"].dropna()

        comp_summary = {
            "sale_year_range_prefix": "between" if " and " in comps_df["sale_year_range"].iloc[0] else "in",
            "sale_year_range": comps_df["sale_year_range"].iloc[0],
            "avg_sale_price": sale_prices.mean(),
            "avg_price_per_sqft": sqft_prices.mean(),
        }

        # ------------- the predictors list ----------------------------------
        predictors_raw = card_df["model_predictor_all_name"]
        print(predictors_raw)
        predictors = [p.strip() for p in str(predictors_raw).split(",")]
        #predictors = (
        #    ast.literal_eval(predictors_raw) if isinstance(predictors_raw, str) else list(predictors_raw)
        #)

        # ------------- build the full card dict -----------------------------
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
                "chars": {
                    "char_class": card_df["char_class"],
                    "char_yrblt": int(card_df["char_yrblt"]),
                    "char_bldg_sf": card_df["char_bldg_sf"],
                    "char_land_sf": card_df["char_land_sf"],
                    "char_beds": int(card_df["char_beds"]),
                    "char_fbath": int(card_df["char_fbath"]),
                    "char_hbath": int(card_df["char_hbath"]),
                },
                "has_subject_pin_sale": bool(comps_df["is_subject_pin_sale"].any()),
                "pred_card_initial_fmv": card_df["pred_card_initial_fmv"],
                "pred_card_initial_fmv_per_sqft": card_df.get(
                    "pred_card_initial_fmv_per_sqft",
                    card_df["pred_card_initial_fmv"] / card_df["char_bldg_sf"]
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
    This is so the frontmatter doesn't through data type errors.
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

# ---------- dump to markdown ------------------------------------------------

def write_md(front_dict: dict, outfile: str | Path) -> None:
    # Convert all numpy types to built-in Python types
    front_dict = convert_to_builtin_types(front_dict)

    yaml_block = yaml.safe_dump(front_dict, sort_keys=False, width=100)
    md_text = f"---\n{yaml_block}---\n"

    Path(outfile).write_text(md_text, encoding="utf8")
    print(f"Wrote {outfile}")


# ------------------ USAGE EXAMPLE ------------------------------------------
front = build_front_matter(df_target_pin, df_comps)
write_md(front, "report.md")