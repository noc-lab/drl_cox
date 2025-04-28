import numpy as np
import pandas as pd
import copy

def perturb(train, var=1.0, num_col=8, ratio=0.3, seed=None):
    """
    Return a perturbed copy of `train` (NumPy array or pandas DataFrame).

    Parameters
    ----------
    train : np.ndarray or pd.DataFrame
        Original dataset. Last two columns assumed to be [time, event].
    var : float, default=1.0
        Magnitude factor for both feature and y perturbations.
    num_col : int, default=8
        Number of leading columns to treat as numerical features.
    ratio : float in (0,1), default=0.3
        Fraction of rows to perturb (will perturb ratio/2 “positive” and ratio/2 “negative”).
    seed : int or None
        Random seed for reproducibility.

    Returns
    -------
    perturbed : same type as `train`
        Copy of `train` with perturbations applied.
    """
    # Copy and convert to NumPy for in-place operations
    is_df = isinstance(train, pd.DataFrame)
    arr = train.values.copy() if is_df else train.copy()
    rng = np.random.default_rng(seed)

    n_rows = arr.shape[0]
    n_perturb = int(n_rows * ratio)

    # Compute stats once
    feat_means = arr[:, :num_col].mean(axis=0)
    # Scale for Gaussian: sqrt(|var * mean / 2|)
    noise_scales = np.sqrt(np.abs(var * feat_means / 2.0))

    # Minimum allowed y (time or risk score) plus 0.5
    y_idx = -2
    min_y = arr[:, y_idx].min() + 0.5

    # Apply ± perturbations on features, and multiplicative noise on y
    for i in range(n_perturb):
        # feature perturbation
        arr[i, :num_col] += rng.normal(loc=var * feat_means,
                                        scale=noise_scales,
                                        size=num_col)
        # y perturbation (multiplicative noise around 1.0 ± var)
        y_noise = rng.normal(loc=1.0 + var, scale=var)
        arr[i, y_idx] = max(min_y, arr[i, y_idx] * y_noise)

    # If original was DataFrame, re-wrap
    if is_df:
        perturbed = pd.DataFrame(arr, columns=train.columns, index=train.index)
    else:
        perturbed = arr

    return perturbed

def shift(train, var=1, num_col=8, ratio=0.3, seed=None):
    """
    Apply controlled but still strong shifts to a fraction of the data, ensuring y does not change excessively.

    - Features: large uniform shifts plus a random directional bias.
    - y: original y plus a bounded delta (no more than var * total y-range), plus small noise.
    - Final y clipped to remain within safe bounds.
    """
    # RNG for reproducibility
    rng = np.random.default_rng(seed)

    # Copy and convert to float array
    arr = copy.deepcopy(train)
    arr = np.array(arr, dtype=float)

    n_rows = arr.shape[0]
    n_shift = int(n_rows * ratio)

    # y index and range
    y_idx = -2
    y_vals = arr[:, y_idx]
    y_min, y_max = y_vals.min(), y_vals.max()
    y_range = y_max - y_min

    # Random direction for feature shifts
    direction = rng.laplace(loc=0.0, scale=1, size=var)

    # For first n_shift rows, shift features and y
    for i in range(n_shift):
        # Feature shift: uniform in [-5*var,5*var] plus directional bias
        arr[i, :var] = rng.normal(scale=1,loc=1, size=var) + direction

        # # Compute bounded delta for y: projection clipped to ±var * y_range
        # proj = np.dot(arr[i, :num_col], direction)
        # delta_y = np.clip(proj, -var * y_range, var * y_range)

        # # Small Gaussian noise around zero (scale = var*0.1*y_range)
        # noise = rng.normal(loc=0.0, scale=var * 0.1 * y_range)

        # # New y = original y + bounded delta + noise
        # orig_y = train[i][-2] if isinstance(train, (list, np.ndarray)) else train.iloc[i, y_idx]
        # new_y = orig_y + delta_y + noise

        # # Final clip to [y_min + epsilon, y_max - epsilon]
        # arr[i, y_idx] = np.clip(new_y, y_min + 0.1 * y_range, y_max - 0.1 * y_range)

    return arr



