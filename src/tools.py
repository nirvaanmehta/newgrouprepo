# tools registry for ai data analysis agent

import src.checks as checks
import src.io_utils as io_utils
import src.modeling as modeling
import src.plotting as plotting
import src.profiling as profiling
import src.summaries as summaries

# Import your new prediction functions
from src.modeling import (
    multiple_linear_regression,
    predict_sleep_quality,
    predict_stress_from_sleep_phone
)

# COMBINE all tools into ONE dictionary (don't overwrite!)
TOOLS = {
    # summaries
    "summarize_numeric": summaries.summarize_numeric,
    "summarize_categorical": summaries.summarize_categorical,
    "missingness_table": summaries.missingness_table,
    # profiling
    "basic_profile": profiling.basic_profile,
    "split_columns": profiling.split_columns,
    # modeling (existing + new)
    "multiple_linear_regression": multiple_linear_regression,
    "predict_sleep_quality": predict_sleep_quality,              # NEW
    "predict_stress_from_sleep_phone": predict_stress_from_sleep_phone,  # NEW
    # plotting
    "plot_missingness": plotting.plot_missingness,
    "plot_corr_heatmap": plotting.plot_corr_heatmap,
    "plot_histograms": plotting.plot_histograms,
    "plot_bar_charts": plotting.plot_bar_charts,
    # checks
    "assert_json_safe": checks.assert_json_safe,
    "target_check": checks.target_check,
    # io
    "ensure_dirs": io_utils.ensure_dirs,
    "read_data": io_utils.read_data,
}

# COMBINE all descriptions 
TOOL_DESCRIPTIONS = {
    "plot_bar_charts": "Bar chart of category counts for categorical columns (NOT associations with numeric variables).",
    "plot_cat_num_boxplot": "Boxplot showing the distribution of a numeric variable grouped by a categorical variable (categorical–numeric association).",
    "multiple_linear_regression": "Fit a multiple linear regression model to predict an outcome from predictors.",
    "predict_sleep_quality": "Predict sleep quality score (1-10) from phone usage before sleep (minutes) and sleep duration (hours).",
    "predict_stress_from_sleep_phone": "Predict stress level (1-10) from sleep duration (hours) and phone usage before sleep (minutes).",
}