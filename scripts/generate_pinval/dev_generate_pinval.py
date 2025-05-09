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
