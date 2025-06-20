"""Constants that are shared between scripts in this module"""

# Valid assessment triads. Empty string disables the triad filter
TRIAD_CHOICES: tuple[str, ...] = ("city", "north", "south", "")

# Temporary solution for run_id mapping, a problem that occurs when the model run_id
# differs between the model values and the comps
RUN_ID_MAP = {"2025-02-11-charming-eric": "2025-04-25-fancy-free-billy"}
