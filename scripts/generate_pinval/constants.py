"""Constants that are shared between scripts in this module"""

# Temporary solution for run_id mapping, a problem that occurs when the model run_id
# differs between the model values and the comps.
# Currently we don't need more than one run because comps and cards are using the same
# final run ID, but we're keeping this map around with the expectation that we may
# need it for a more permanent fix
RUN_ID_MAP = {"2025-02-11-charming-eric": "2025-02-11-charming-eric"}

# It's helpful to factor these tables out into shared constants because we often
# need to switch to dev tables for testing
PINVAL_ASSESSMENT_CARD_TABLE = "z_dev_jecochr_pinval.vw_assessment_card"
PINVAL_COMP_TABLE = "z_dev_jecochr_pinval.vw_comp"
