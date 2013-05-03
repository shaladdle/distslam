from math import sqrt
from vector import *

# eye and center are Point objects; ray is a Vector2; radius is a float
def line_circle(eye, ray, center, radius):

    e_minus_c = Vector2(center.x - eye.x, center.y - eye.y)
    #e_minus_c = Vector2(eye.x - center.x, eye.y - center.y)

    discriminant = (dot(ray, (e_minus_c)))**2 - dot(ray, ray) * \
            (dot((e_minus_c), (e_minus_c)) - radius**2)

    if (discriminant < 0):
        return -1;

    # return lower value of t (ie minus discriminant)
    # not sure why it has to be negative of standard formula,
    # maybe because of our weird coordinate space
    # t = (-1.0 * dot(ray, (e_minus_c)) - sqrt(discriminant)) / dot(ray, ray)
    t = (dot(ray, (e_minus_c)) - sqrt(discriminant)) / dot(ray, ray)

    return t
