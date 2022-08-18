from scipy.stats import norm

def expected_improvement(y_star: float, mu: float, sigma: float):
    if sigma <= 0:
        return max(y_star - mu)
    z = (y_star-mu)/sigma
    return sigma*(z*norm.cdf(z) + norm.pdf(z))

def constrained_expected_improvement(y_star: float, mu: float, sigma: float, probabilities=None):
    ei = expected_improvement(y_star, mu, sigma)
    if probabilities is not None:
        for proba in probabilities:
            ei *= proba
    return ei