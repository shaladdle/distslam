import numpy as np
from math import cos, sin, pi, sqrt, acos

# a and b are 2d Points
class Vector2(np.matrix):
    def __new__(cls, x, y):
        obj = np.asarray([[x], [y]]).view(cls)
        return obj
    def __array_finalize__(self, obj):
        pass
    def length(self):
        return np.linalg.norm(self)
    def normalize(self):
        return self / self.length()

# a and b are Vector2
def dot(a, b):
    return np.linalg.norm(a.transpose() * b)
    #return a.x * b.x + a.y * b.y

# a and b are Vector2
def interior_angle(a, b):
   return acos(dot(a, b) / (a.length() * b.length())) 



