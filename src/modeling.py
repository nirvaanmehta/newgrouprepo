from typing import Optional, List, Dict, Any
import pandas as pd

def multiple_linear_regression(
    df: pd.DataFrame, outcome: str, predictors: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    (Student task): Fit a multiple linear regression model.

    Requirements:
    - Outcome must be numeric; raise ValueError otherwise
    - If predictors is None:
        use ALL numeric columns except outcome
    - Drop rows with missing values in outcome or predictors before fitting
    - Fit the model using least squares:
        y = intercept + b1*x1 + b2*x2 + ...
    - Return a JSON-safe dictionary containing:
        outcome, predictors, n_rows_used, r_squared, adj_r_squared,
        intercept, coefficients (dict)

    Hints: use statsmodels package:
    import statsmodels.api as sm
    X = df[predictors]
    X = sm.add_constant(X)
    y = df[outcome]
    model = sm.OLS(y, X).fit()

    IMPORTANT:
    - Convert any numpy/pandas scalars to Python floats/ints before returning.
    """
    import statsmodels.api as sm
    
    if outcome not in df.columns: 
        raise ValueError(f"Outcome column '{outcome}' not found in the dataframe.")
    if not pd.api.types.is_numeric_dtype(df[outcome]):
        raise ValueError("Outcome variable must be numeric.")
    if predictors is None: 
        numeric_cols = df.select_dtypes(include = ["number"]).columns.tolist()
        predictors = [c for c in numeric_cols if c != outcome]
    if not predictors: 
        raise ValueError("No valid predictors were provided.")
    
    model_df = df[[outcome] + predictors].dropna()

    if model_df.empty: 
        raise ValueError("No rows remaining after dropping missing values.")
    
    X = model_df[predictors]
    X = sm.add_constant(X)
    y = model_df[outcome]

    model = sm.OLS(y, X).fit()

    coef_dict = {
        str(k): float(v)
        for k, v in model.params.items()
        if k != "const"
    }

    results = {
        "outcome": str(outcome),
        "predictors": [str(p) for p in predictors],
        "n_rows_used": int(model.nobs),
        "r_squared": float(model.rsquared),
        "adj_r_squared": float(model.rsquared_adj),
        "intercept": float(model.params["const"]),
        "coefficients": coef_dict
    }
    return results

def predict_sleep_quality(
    df: pd.DataFrame,
    phone_mins: float,
    sleep_hours: float,
    report_dir: Optional[str] = None
) -> Dict[str, Any]:
    """
    Predict sleep quality score based on phone usage before sleep and sleep duration.
    
    Args:
        df: DataFrame with columns 'phone_usage_before_sleep_minutes', 
            'sleep_duration_hours', 'sleep_quality_score'
        phone_mins: Minutes spent on phone before sleep
        sleep_hours: Hours of sleep
        report_dir: Optional directory to save plots (ignored in this tool)
    
    Returns:
        Dictionary with predicted sleep quality and model info
    """
    from sklearn.linear_model import LinearRegression
    import numpy as np
    
    # Required columns
    required_cols = ['phone_usage_before_sleep_minutes', 'sleep_duration_hours', 'sleep_quality_score']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in dataframe. Available: {df.columns.tolist()}")
    
    # Prepare data
    X = df[['phone_usage_before_sleep_minutes', 'sleep_duration_hours']].dropna()
    y = df.loc[X.index, 'sleep_quality_score']
    
    if len(X) < 10:
        raise ValueError(f"Need at least 10 rows for training. Only {len(X)} available.")
    
    # Train model
    model = LinearRegression()
    model.fit(X, y)
    
    # Make prediction
    predicted = model.predict([[phone_mins, sleep_hours]])[0]
    predicted = max(1.0, min(10.0, float(predicted)))  # Clamp to 1-10 range
    
    # Calculate R-squared
    y_pred = model.predict(X)
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r_squared = float(1 - (ss_res / ss_tot)) if ss_tot > 0 else 0.0
    
    return {
        "tool": "predict_sleep_quality",
        "input_phone_minutes": float(phone_mins),
        "input_sleep_hours": float(sleep_hours),
        "predicted_sleep_quality": predicted,
        "r_squared": r_squared,
        "n_rows_used": int(len(X)),
        "coefficients": {
            "phone_usage_before_sleep": float(model.coef_[0]),
            "sleep_duration_hours": float(model.coef_[1])
        },
        "intercept": float(model.intercept_),
        "interpretation": _interpret_sleep_quality(predicted, phone_mins, sleep_hours)
    }


def predict_stress_from_sleep_phone(
    df: pd.DataFrame,
    sleep_hours: float,
    phone_mins: float,
    report_dir: Optional[str] = None
) -> Dict[str, Any]:
    """
    Predict stress level based on sleep duration and phone usage before sleep.
    
    Args:
        df: DataFrame with columns 'sleep_duration_hours', 
            'phone_usage_before_sleep_minutes', 'stress_level'
        sleep_hours: Hours of sleep
        phone_mins: Minutes spent on phone before sleep
        report_dir: Optional directory to save plots (ignored in this tool)
    
    Returns:
        Dictionary with predicted stress level and model info
    """
    from sklearn.linear_model import LinearRegression
    import numpy as np
    
    # Required columns
    required_cols = ['sleep_duration_hours', 'phone_usage_before_sleep_minutes', 'stress_level']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in dataframe. Available: {df.columns.tolist()}")
    
    # Prepare data
    X = df[['sleep_duration_hours', 'phone_usage_before_sleep_minutes']].dropna()
    y = df.loc[X.index, 'stress_level']
    
    if len(X) < 10:
        raise ValueError(f"Need at least 10 rows for training. Only {len(X)} available.")
    
    # Train model
    model = LinearRegression()
    model.fit(X, y)
    
    # Make prediction
    predicted = model.predict([[sleep_hours, phone_mins]])[0]
    predicted = max(1.0, min(10.0, float(predicted)))  # Clamp to 1-10 range
    
    # Calculate R-squared
    y_pred = model.predict(X)
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r_squared = float(1 - (ss_res / ss_tot)) if ss_tot > 0 else 0.0
    
    return {
        "tool": "predict_stress_from_sleep_phone",
        "input_sleep_hours": float(sleep_hours),
        "input_phone_minutes": float(phone_mins),
        "predicted_stress_level": predicted,
        "r_squared": r_squared,
        "n_rows_used": int(len(X)),
        "coefficients": {
            "sleep_duration_hours": float(model.coef_[0]),
            "phone_usage_before_sleep": float(model.coef_[1])
        },
        "intercept": float(model.intercept_),
        "interpretation": _interpret_stress(predicted, sleep_hours, phone_mins)
    }


# Helper functions for interpretations
def _interpret_sleep_quality(predicted: float, phone_mins: float, sleep_hours: float) -> str:
    """Generate interpretation for sleep quality prediction."""
    if predicted >= 8:
        quality = "Excellent"
    elif predicted >= 6:
        quality = "Good"
    elif predicted >= 4:
        quality = "Fair"
    else:
        quality = "Poor"
    
    text = f"Predicted sleep quality: {quality} ({predicted:.1f}/10). "
    
    if phone_mins > 60:
        text += f"Your phone usage ({phone_mins} min) is high. Try reducing to 30 min."
    elif sleep_hours < 7:
        text += f"Your sleep ({sleep_hours}h) is below recommended 7-9 hours."
    else:
        text += "Your habits are in a healthy range. Keep it up!"
    
    return text


def _interpret_stress(predicted: float, sleep_hours: float, phone_mins: float) -> str:
    """Generate interpretation for stress prediction."""
    if predicted >= 8:
        level = "High"
    elif predicted >= 5:
        level = "Moderate"
    else:
        level = "Low"
    
    text = f"Predicted stress level: {level} ({predicted:.1f}/10). "
    
    if sleep_hours < 6:
        text += f"Low sleep ({sleep_hours}h) is a major stress risk factor. "
    if phone_mins > 60:
        text += f"High phone use before bed ({phone_mins} min) worsens sleep quality."
    
    if sleep_hours >= 7 and phone_mins <= 30:
        text += "Your sleep and phone habits are optimal for stress management."
    
    return text 