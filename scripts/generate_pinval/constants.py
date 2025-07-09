"""Constants that are shared between scripts in this module"""

# Temporary solution for run_id mapping, a problem that occurs when the model run_id
# differs between the model values and the comps
RUN_ID_MAP = {"2025-02-11-charming-eric": "2025-04-25-fancy-free-billy"}

# It's helpful to factor these tables out into shared constants because we often
# need to switch to dev tables for testing
PINVAL_ASSESSMENT_CARD_TABLE = "pinval.vw_assessment_card"
PINVAL_COMP_TABLE = "pinval.vw_comp"
