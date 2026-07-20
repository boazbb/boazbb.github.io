import numpy as np

class Gaussian:
    def __init__(self, mean: np.ndarray, amplitude: float, sigma: np.ndarray):
        self.mean = np.asarray(mean, dtype=float)
        self.amplitude = float(amplitude)
        self.sigma = np.asarray(sigma, dtype=float)
        assert self.mean.shape == (2,), "Mean must be a 2D vector."
        assert self.sigma.shape == (2, 2), "Sigma must be a 2x2 matrix."
        self.inv_sigma = np.linalg.inv(self.sigma)
        self.det_sigma = np.linalg.det(self.sigma)

    def __call__(self, x_or_point, y=None):
        """Accept either a (2,) point array or separate (x, y) scalar/grid arrays."""
        if y is None:
            p = np.asarray(x_or_point, dtype=float)
            x, y_ = p[0], p[1]
        else:
            x, y_ = np.asarray(x_or_point, dtype=float), np.asarray(y, dtype=float)
        dx = x - self.mean[0]
        dy = y_ - self.mean[1]
        a, b = self.inv_sigma[0, 0], self.inv_sigma[0, 1]
        c, d = self.inv_sigma[1, 0], self.inv_sigma[1, 1]
        exponent = -0.5 * (a * dx**2 + (b + c) * dx * dy + d * dy**2)
        return self.amplitude * np.exp(exponent)

    def gradient(self, point: np.ndarray) -> np.ndarray:
        p = np.asarray(point, dtype=float)
        return -self(p) * (self.inv_sigma @ (p - self.mean))


class GaussianBowl:
    def __init__(self, gaussian_list: list[Gaussian]):
        self.gaussians = gaussian_list
        self.norm = max(g.amplitude for g in gaussian_list) if gaussian_list else 0.0

    def __call__(self, x_or_point, y=None):
        """Accept either a (2,) point array or separate (x, y) scalar/grid arrays."""
        if y is None:
            p = np.asarray(x_or_point, dtype=float)
            x, y_ = p[0], p[1]
        else:
            x, y_ = np.asarray(x_or_point, dtype=float), np.asarray(y, dtype=float)
        output = x**2 + y_**2
        for gaussian in self.gaussians:
            output = output - gaussian(x, y_)
        return output + self.norm

    def gradient(self, point: np.ndarray) -> np.ndarray:
        p = np.asarray(point, dtype=float)
        grad = 2.0 * p
        for gaussian in self.gaussians:
            grad = grad - gaussian.gradient(p)
        return grad
