# Decline Curve Analysis (DCA) Documentation

## Overview

This Python module implements **Decline Curve Analysis (DCA)**, a fundamental technique in petroleum engineering used to forecast future production rates and estimate recoverable reserves from oil and gas wells. The implementation supports three classical Arps decline models and provides comprehensive visualization and analysis capabilities.

## Features

- **Multiple Decline Models**: Exponential, Hyperbolic, and Harmonic
- **Automatic Model Selection**: Finds the best-fitting model based on R² values
- **EUR Calculation**: Computes Estimated Ultimate Recovery with economic limits
- **Advanced Visualization**: Interactive 2x2 subplot dashboard with Plotly
- **Datetime Support**: Handles both numeric time arrays and datetime objects
- **Parameter Sensitivity Analysis**: Evaluates impact of parameter variations

## Installation

```bash
pip install numpy pandas plotly scipy
```

## Core Concepts

### Arps Decline Models

The module implements three decline curve models based on J.J. Arps' empirical relationships:

1. **Exponential Decline** (b = 0)
   - Constant percentage decline rate
   - Most conservative forecast
   - Common in mature, stabilized wells

2. **Hyperbolic Decline** (0 < b < 1)
   - Variable decline rate that decreases over time
   - Most flexible model
   - Typical in fractured reservoirs

3. **Harmonic Decline** (b = 1)
   - Decline rate inversely proportional to time
   - Most optimistic forecast
   - Observed in naturally fractured formations

## Class Reference

### `DeclineCurveAnalysis`

#### Initialization

```python
dca = DeclineCurveAnalysis(model_type="exponential")
```

**Parameters:**
- `model_type` (str): One of `"exponential"`, `"hyperbolic"`, or `"harmonic"`

**Attributes:**
- `qi`: Initial production rate (bbl/day)
- `di`: Initial decline rate (1/day)
- `b`: Hyperbolic exponent (0 for exponential, 1 for harmonic)
- `r_squared`: Goodness of fit metric
- `popt`: Optimized parameters from curve fitting
- `pcov`: Covariance matrix of parameters

### Methods

#### `fit(time_data, rate_data)`

Fits the selected decline model to historical production data.

**Parameters:**
- `time_data` (np.ndarray): Time values (days or datetime objects)
- `rate_data` (np.ndarray): Production rates (bbl/day)

**Returns:**
- `popt` (array): Optimized model parameters

**Example:**
```python
time = np.array([0, 30, 60, 90, 120, 150])
rate = np.array([1000, 850, 720, 610, 520, 450])

dca = DeclineCurveAnalysis(model_type="exponential")
params = dca.fit(time, rate)
print(f"qi={dca.qi:.2f}, di={dca.di:.4f}, R²={dca.r_squared:.4f}")
```

#### `fit_best_model(time_data, rate_data)`

Automatically tests all three models and selects the one with the highest R² value.

**Returns:**
- `best_model` (str): Name of the best-fitting model
- `r_squared` (float): R² value of the best model
- `popt` (array): Optimized parameters

**Example:**
```python
model, r2, params = dca.fit_best_model(time, rate)
print(f"Best model: {model} with R²={r2:.4f}")
```

#### `forecast(time_forecast)`

Generates production rate forecasts for specified time points.

**Parameters:**
- `time_forecast` (np.ndarray): Future time points for prediction

**Returns:**
- `rates` (np.ndarray): Forecasted production rates

**Example:**
```python
future_time = np.linspace(0, 365, 100)
forecasted_rates = dca.forecast(future_time)
```

#### `calculate_eur(economic_limit=50, max_time=600)`

Calculates Estimated Ultimate Recovery (EUR) using numerical integration.

**Parameters:**
- `economic_limit` (float): Minimum economic production rate (bbl/day)
- `max_time` (int): Maximum forecast period (days)

**Returns:**
- `eur` (float): Total recoverable volume (bbl)

**Example:**
```python
eur = dca.calculate_eur(economic_limit=50, max_time=1800)
print(f"EUR: {eur:,.0f} barrels")
```

#### `create_detailed_plot(time_data, rate_data, economic_limit=50)`

Generates a comprehensive 2×2 interactive visualization dashboard.

**Subplots:**
1. **Production Decline**: Historical data vs. forecast curve
2. **Cumulative Production**: Integrated production over time with EUR annotation
3. **Residual Analysis**: Deviation between actual and predicted values
4. **Parameter Sensitivity**: Impact of ±10% parameter variations

**Returns:**
- `fig` (go.Figure): Plotly figure object

**Example:**
```python
fig = dca.create_detailed_plot(time, rate, economic_limit=50)
fig.show()
```

## Complete Usage Example

```python
import numpy as np
import pandas as pd
from decline_curve_analysis import DeclineCurveAnalysis

# Generate sample production data
dates = pd.date_range(start='2023-01-01', periods=12, freq='M')
rates = np.array([1200, 980, 820, 710, 620, 550, 490, 445, 410, 380, 355, 335])

# Initialize and fit model
dca = DeclineCurveAnalysis()
best_model, r2, params = dca.fit_best_model(dates, rates)

print(f"Best Model: {best_model}")
print(f"R² Score: {r2:.4f}")
print(f"Parameters: qi={dca.qi:.2f}, di={dca.di:.6f}, b={dca.b:.4f}")

# Calculate EUR
eur = dca.calculate_eur(economic_limit=100, max_time=3650)
print(f"Estimated Ultimate Recovery: {eur:,.0f} bbl")

# Generate forecast
future_dates = pd.date_range(start=dates[0], periods=36, freq='M')
forecast = dca.forecast((future_dates - dates[0]).days.to_numpy())

# Create visualization
fig = dca.create_detailed_plot(dates, rates, economic_limit=100)
fig.show()
```

## Technical Details

### Curve Fitting

The module uses `scipy.optimize.curve_fit` with:
- **Initial guesses**: `qi = rate_data[0]`, `di = 0.01`, `b = 0.5`
- **Bounds**: All parameters constrained to physically meaningful ranges
- **Max iterations**: 10,000 for convergence

### R² Calculation

Model fit quality is assessed using the coefficient of determination:

```
R² = 1 - (SS_res / SS_tot)
```

Where:
- `SS_res`: Sum of squared residuals
- `SS_tot`: Total sum of squares

### EUR Integration

Recoverable reserves are computed using trapezoidal numerical integration:

```
EUR = ∫[t=0 to t_econ] q(t) dt
```

Where `t_econ` is the time when production drops to the economic limit.

## Best Practices

1. **Data Quality**: Ensure production data is stabilized (post-cleanup period)
2. **Model Selection**: Use `fit_best_model()` when decline behavior is uncertain
3. **Economic Limits**: Set realistic abandonment rates based on operating costs
4. **Forecast Horizon**: Limit predictions to 2-3× historical data timespan
5. **Validation**: Review residual plots for systematic patterns indicating poor fit

## Limitations

- Assumes single-phase flow behavior
- Does not account for operational changes (workovers, stimulation)
- Extrapolation beyond data range introduces uncertainty
- Ignores regulatory or mechanical constraints on production



## License

This module is provided for educational and professional use in petroleum engineering applications.

## Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for bugs and feature requests.

## Contact

For questions or support, please open an issue in the repository.
adomfosu2000@gmail.com
