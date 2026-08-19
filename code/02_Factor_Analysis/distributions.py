import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import fit, chi, lognorm , gamma, expon
from scipy.optimize import minimize
from scipy.special import psi, gammaln 
from scipy.special import gamma as gamma_chi

def log_normal_log_likelihood(params, data):
    """Calculate the log-likelihood for log-normal distribution."""
    mu, sigma = params
    if sigma <= 0:
        return -np.inf  # Return negative infinity if sigma is non-positive
    return np.sum(lognorm.logpdf(data, s=sigma, scale=np.exp(mu)))

def fit_log_normal_distribution(data):
    """Fit the log-normal distribution to the data."""
    # Log-transform data
    log_data = np.log(data[data > 0])  # Exclude non-positive data for log transformation
    # Initial parameter estimates (mean and std deviation of log data)
    mean_log_data = np.mean(log_data)
    std_log_data = np.std(log_data)
    initial_guess = [mean_log_data, std_log_data]

    # Minimize the negative log likelihood
    result = minimize(
        lambda params: -log_normal_log_likelihood(params, data),
        initial_guess,
        bounds=[(-np.inf, np.inf), (0.001, None)]  # Bounds to ensure sigma is positive
    )
    
    # Extract the estimated parameters
    mu_est, sigma_est = result.x
    print("Estimated mu:", mu_est)
    print("Estimated sigma:", sigma_est)

    # Plot histogram of the data
    plt.figure(figsize=(8, 5))
    count, bins, ignored = plt.hist(data, bins=30, density=True, alpha=0.6, color='lightgray', label='Sampled Data Histogram')
    x = np.linspace(min(data), max(data), num=300)
    plt.plot(x, lognorm.pdf(x, s=sigma_est, scale=np.exp(mu_est)), 'r-', lw=2, label="Fitted Log-Normal PDF")
    
    # Add labels, legend, and grid
    plt.title(f'Data and PDF for Log-Normal Distribution with μ={mu_est:.2f}, σ={sigma_est:.2f}')
    plt.xlabel('Data')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True)

    # Show the complete plot
    plt.show()

    return mu_est, sigma_est

def calculate_log_normal_aic(data, params):
    """ Calculate Akaike Information Criterion (AIC) for the log-normal distribution fit to data."""
    data = data[data > 0]
    if params is not None:
        # Calculate maximum log-likelihood
        max_log_likelihood = log_normal_log_likelihood(params, data)
        # Number of parameters in the log-normal distribution is 2 (mu and sigma)
        k = 2
        # Compute AIC
        aic = 2 * k - 2 * max_log_likelihood
        return aic
    else:
        return None

def gamma_log_likelihood(params, data):
    """ Calculate the log-likelihood for gamma distribution """
    alpha, beta = params
    data = data[data > 0]
    # Ensure positive parameters for the gamma distribution
    if alpha <= 0 or beta <= 0:
        return -np.inf  # log-likelihood of negative infinity if parameters are non-physical
    return np.sum(gamma.logpdf(data, a=alpha, scale=1/beta))

def fit_gamma_distribution(data):

    # Initial guesses for alpha and beta can be based on method of moments estimates
    mean_data = np.mean(data)
    std_data = np.std(data)
    initial_alpha = mean_data**2 / std_data**2
    initial_beta = mean_data / std_data**2

    # Minimize the negative log-likelihood
    initial_params = [initial_alpha, initial_beta]
        # Minimize the negative log likelihood
    result = minimize(
        lambda params: -gamma_log_likelihood(params, data),
        initial_params,
        bounds=[(0.001, None), (0.001, None)]  # Bounds to keep parameters positive
    )
    
    # Estimated parameters
    alpha_est, beta_est = result.x
    print("Estimated alpha (shape):", alpha_est)
    print("Estimated beta (rate):", beta_est)

    # Plot histogram of the data
    plt.figure(figsize=(8, 5))
    plt.hist(data, bins=30, density=True, alpha=0.6, color='lightgray', label='Sampled Data Histogram')
    value_space = np.linspace(0, max(data), num=300)
    plt.plot(value_space, gamma.pdf(value_space, a=alpha_est, scale=1/beta_est), 'r-', lw=2, label="Fitted Gamma PDF")
    
    # Add labels, legend, and grid
    plt.title(f'Data and PDF for Gamma Distribution with α={alpha_est:.2f}, β={beta_est:.2f}')
    plt.xlabel('Data')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True)

    # Show the complete plot
    plt.show()

    return alpha_est, beta_est

def calculate_gamma_aic(data , params):
    """ Calculate Akaike Information Criterion (AIC) for the Gamma distribution fit to data """
    # Fit the model and find optimal parameters
    if params is not None:
        # Recover params
        alpha, beta = params
        # Calculate maximum log-likelihood
        max_log_likelihood = gamma_log_likelihood(params, data)
        # Number of parameters in the gamma distribution is 2 (alpha, beta)
        k = 2
        # Compute AIC
        aic = 2 * k - 2 * max_log_likelihood
        return aic
    else:
        return None

def exponential_log_likelihood(rate, data):
    """Calculate the log-likelihood for exponential distribution."""
    if rate <= 0:
        return -np.inf  # Return negative infinity if rate is non-positive
    return np.sum(expon.logpdf(data, scale=1/rate))


def fit_exponential_distribution(data):
    """Fit the exponential distribution to the data."""
    # Initial parameter estimate (inverse of the mean of the data)
    initial_rate = 1 / np.mean(data)
    
    # Minimize the negative log likelihood
    result = minimize(
        lambda rate: -exponential_log_likelihood(rate, data),
        initial_rate,
        bounds=[(0.001, None)]  # Bounds to ensure rate is positive
    )
    
    # Estimated lambda
    lambda_est = result.x[0]
    print("Estimated lambda:", lambda_est)

    # Plot histogram of the data
    plt.figure(figsize=(8, 5))
    plt.hist(data, bins=30, density=True, alpha=0.6, color='lightgray', label='Sampled Data Histogram')
    value_space = np.linspace(0, max(data), num=300)
    plt.plot(value_space, lambda_est * np.exp(-lambda_est * value_space), 'r-', lw=2, label="Fitted Exponential PDF")
    
    # Add labels, legend, and grid
    plt.title(f'Data and PDF for Exponential Distribution with lambda {lambda_est:.2f}')
    plt.xlabel('Data')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True)
    
    # Show the complete plot
    plt.show()

    return lambda_est


def calculate_exponential_aic(data , params):
    """ Calculate Akaike Information Criterion (AIC) for the exponential distribution fit to data."""

    data = data[data > 0]
    
    if params is not None:
        # Calculate maximum log-likelihood
        max_log_likelihood = exponential_log_likelihood(params, data)
        # Number of parameters in the exponential distribution is 1 (rate)
        k = 1
        # Compute AIC
        aic = 2 * k - 2 * max_log_likelihood
        return aic
    else:
        return None

def exponential_significance_classify(data , lambda_est , classify = 'Significant' , thresh = 0.05) : 
    classes = 1 - expon.cdf(data , scale = 1/lambda_est)
    return [np.nan if np.isnan(i) else classify if i < thresh else f'Not {classify}' for i in classes]

def lognorm_significance_classify(data , sig_est , mu_est , classify = 'Significant' , thresh = 0.05) : 
    classes = 1 - lognorm.cdf(data, s=sig_est, scale=np.exp(mu_est))
    return [np.nan if np.isnan(i) else classify if i < thresh else f'Not {classify}' for i in classes]

def gamma_significance_classify(data , alpha_est , beta_est , classify = 'Significant' , thresh = 0.05) : 
    classes = 1 - gamma.cdf(data, a=alpha_est, scale=1/beta_est) 
    return [np.nan if np.isnan(i) else classify if i < thresh else f'Not {classify}' for i in classes]