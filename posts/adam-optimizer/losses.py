import numpy as np
from dataclasses import dataclass

@dataclass
class Gaussian:
    x: float = 0.0
    y: float = 0.0
    amplitude: float = 1.0
    width: float = 0.2

class GaussianBowl:
    def __init__(self, gaussian_list: list[Gaussian]):
        self.gaussians = gaussian_list
        self.norm = max(g.amplitude for g in gaussian_list) if gaussian_list else 0.0

    def __call__(self, x, y):
        # The quadratic bowl:
        output = x**2 + y**2
        for g in self.gaussians:
            output -= g.amplitude * np.exp(-((x - g.x)**2 + (y - g.y)**2) / g.width)
        return output + self.norm
    
    def gradient(self, x, y):
        common_terms = []
        for g in self.gaussians:
            common_terms.append(
                2*g.amplitude * np.exp(-((x - g.x)**2 + (y - g.y)**2) / g.width) / g.width
            )
        
        grad_x = 2 * x
        grad_y = 2 * y
        for g, common in zip(self.gaussians, common_terms):
            grad_x += (x - g.x) * common
            grad_y += (y - g.y) * common

        return grad_x, grad_y