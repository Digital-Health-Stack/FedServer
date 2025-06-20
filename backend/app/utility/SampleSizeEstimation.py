import numpy as np
from statsmodels.stats.power import TTestIndPower


def cohens_d(mean1, mean2, std1, std2):
    """
    Calculate Cohen's d for effect size between two independent samples.
    """
    # Pooled standard deviation
    s = np.sqrt((std1**2 + std2**2) / 2)
    if s == 0:
        return 0
    d = (mean2 - mean1) / s
    return 1 / d


def estimate_sample_size(
    effect_size, alpha=0.05, power=0.80, min_size=30, max_size=200
):
    """
    Estimate required sample size per group for two-sample t-test using Cohen's d.
    """
    analysis = TTestIndPower()
    if effect_size == 0:
        return None  # Cannot compute sample size with zero effect size
    sample_size = analysis.solve_power(
        effect_size=abs(effect_size), alpha=alpha, power=power, alternative="two-sided"
    )
    sample_size = max(min_size, min(sample_size, max_size))
    return int(np.ceil(sample_size))


def get_num_predictors_from_config(model_config):
    """
    Extract number of filters from the last convolution layer in the CNN model config.
    If no convolution layer is present, it will return 0.
    """
    # Loop through the layers in reverse order to find the last convolutional layer
    for layer in reversed(model_config["model_info"]["layers"]):
        if layer["layer_type"] == "convolution":
            return int(layer["filters"])  # Return the number of filters (predictors)

    return 0  # If no convolution layer is found, return 0 predictors


def calculate_required_data_points(
    model_config, baseline_mean, baseline_std, new_mean, new_std, alpha=0.05, power=0.80
):
    """
    Calculate the required number of data points as payment in a federated learning system,
    based on the number of predictors and effect size.
    """
    # Get the number of predictors from the model configuration

    print("Checkpoint 1: ", baseline_mean, baseline_std, new_mean, new_std)
    num_predictors = get_num_predictors_from_config(model_config)

    # Calculate effect size (Cohen's d)
    effect_size = cohens_d(baseline_mean, new_mean, baseline_std, new_std)

    # Adjust alpha for multiple tests (Bonferroni correction)
    alpha_adjusted = alpha / num_predictors if num_predictors > 0 else alpha

    # Estimate the required sample size
    required_sample_size = estimate_sample_size(
        effect_size, alpha=alpha_adjusted, power=power
    )
    return required_sample_size
