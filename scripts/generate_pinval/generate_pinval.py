#!/usr/bin/env python3
"""
Script to generate PINVAL reports using a specified model
run ID and either a triad or list of PINs.
"""

import argparse
import subprocess as sp
import sys
import os
from pathlib import Path


root = sp.getoutput("git rev-parse --show-toplevel")
os.chdir(root)


def parse_args() -> argparse.Namespace:
    """Parse and validate command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Generate PINVAL reports for a given model run-ID and either "
            "a list of PINs or (future) a triad. "
            "Exactly one of --triad or --pin must be supplied."
        )
    )

    parser.add_argument(
        "--run-id",
        required=True,
        help="Model run ID used for comps and training/assessment data",
    )
    parser.add_argument(
        "--triad",
        choices=["city", "north", "south"],
        help="Triad whose reports should be generated "
        "(disallowed until Hugo workflow is ready)",
    )
    parser.add_argument(
        "--pin",
        nargs="*",
        metavar="PIN",
        help="One or more space-separated PINs to generate reports for",
    )

    args = parser.parse_args()

    # Validation
    has_triad = bool(args.triad)
    has_pin_list = args.pin is not None and len(args.pin) > 0

    # Must provide exactly one of the two switches.
    if has_triad == has_pin_list:
        parser.error(
            "Exactly one of --triad or --pin must be supplied "
            "(and not both)."
        )

    # Hugo branch not available yet, triad not in use
    if has_triad:
        parser.error(
            "--triad is not supported until the Hugo workflow lands; "
            "use --pin instead."
        )

    return args

def render_report(pin: str, run_id: str, qmd_path: Path, outdir: Path) -> None:
    """Render a single Quarto report for one PIN."""
    cmd = [
        "quarto", "render", str(qmd_path),
        "--output-dir", str(outdir),
        # Use -P to pass individual parameter values directly to Quarto, each
        # -P expects a YAML-style key:value pair.
        # Quote the PIN value so it is interpreted as a string in YAML/R.
        # Without quotes, Quarto treats it as a number, which breaks the quarto doc
        "-P", f'pin:"{pin}"',
        "-P", f"run_id:{run_id}",
    ]
    try:
        sp.run(cmd, check=True)
        print(f"✔ Rendered report for PIN {pin}")
    except sp.CalledProcessError as exc:
        sys.exit(
            f"❌ Quarto rendering failed for PIN {pin} "
            f"(return-code {exc.returncode})."
        )

  
def main() -> None:
    print(f"📁 Current working directory: {Path.cwd()}")
    args = parse_args()

    qmd_file = Path(root) / "pinval.qmd"
    output_dir = Path(root)

    print(f"📄 Looking for QMD file at: {qmd_file.resolve()}")
    print(f"📂 Output directory will be: {output_dir.resolve()}")

    output_dir.mkdir(parents=True, exist_ok=True)

    for pin in args.pin:
        render_report(pin, args.run_id, qmd_file, output_dir)


if __name__ == "__main__":
    main()
