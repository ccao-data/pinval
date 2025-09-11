"""Constants that are shared between scripts in this module"""

# It's helpful to factor these tables out into shared constants because we often
# need to switch to dev tables for testing
PINVAL_ASSESSMENT_CARD_TABLE = "pinval.vw_assessment_card"
PINVAL_COMP_TABLE = (
    "z_ci_jeancochrane_deploy_alt_comps_algorithm_to_staging_pinval.vw_comp"
)
PINVAL_DATA_DICT_TABLE = (
    "z_ci_jeancochrane_deploy_alt_comps_algorithm_to_staging_pinval.vars_dict"
)
