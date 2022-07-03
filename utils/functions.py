"""
データ生成用の関数たち
"""

import math
import numpy as np

class AsinX():
    def __init__(self, a: float) -> None:
        self.a = a
    
    def  __call__(self, x: float) -> np.float64:
        return self.a * np.sin(math.radians(x))

class AcosX():
    def __init__(self, a: float) -> None:
        self.a = a

    def __call__(self, x: float) -> np.float64:
        return self.a * np.cos(math.radians(x))

class LinearFunc():
    def __init__(self, a: float, b: float) -> None:
        self.a = a
        self.b = b
    
    def __call__(self, x: float) -> np.float64:
        return self.a * x + self.b

class SquareFunc():
    def __init__(self, a: float, b: float, c: float) -> None:
        self.a = a
        self.b = b
        self.c = c
    
    def __call__(self, x: float) -> np.float64:
        return self.a * x**2 + self.b * x + self.c

class CubicFunc():
    def __init__(self, a: float, b:float, c: float, d: float) -> None:
        self.a = a
        self.b = b
        self.c = c
        self.d = d
    
    def __call__(self, x:float) -> np.float64:
        return self.a*x**3 + self.b*x**2 + self.c*x + self.d

class MyFunc():
    def __init__(self, a: float, b:float, c: float) -> None:
        self.a = a
        self.b = b
        self.c = c
    
    def __call__(self, x: float) -> np.float64:
        return self.a*np.sin(math.radians(x)) + self.b*np.cos(math.radians(self.c*(x)))

class MyFunc2():
    def __init__(self, a: float, b:float, c: float) -> None:
        self.a = a
        self.b = b
        self.c = c

    def __call__(self, x: float) -> np.float64:
        return self.a*np.cos(math.radians(x)) + self.b*np.sin(math.radians(self.c*(x)))