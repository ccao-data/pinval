"""Constants that are shared between scripts in this module"""

# It's helpful to factor these tables out into shared constants because we often
# need to switch to dev tables for testing
PINVAL_ASSESSMENT_CARD_TABLE = (
    "z_ci_add_proration_data_to_pinval_assessment_pin_pinval.vw_assessment_card"
)
PINVAL_COMP_TABLE = "pinval.vw_comp"
PINVAL_DATA_DICT_TABLE = "pinval.vars_dict"
