import graphics as g
import numpy as np
import intersect
import vector
from math import acos, cos, sin, pi, sqrt

def matdot(m1, m2):
    return np.dot([x for [x] in np.array(m1)]
                 ,[x for [x] in np.array(m2)]
                 )

class FourRectangle:
    def __init__(self, points, sense_max):
        tl, bl, br, tr = points;
        self.lines = [ g.Line(tl,bl)
                     , g.Line(bl,br)
                     , g.Line(br,tr)
                     , g.Line(tr,tl)
                     ]
        self.sight_range = sense_max

    def draw(self, win):
        ret = {}
        absret = {}
        for l in self.lines:
            l.draw(win)

    def undraw(self):
        for l in self.lines:
            l.undraw()

class Landmark:
    def __init__(self, win, ident, rad, center):
        self.ident = ident
        self.center = center
        self.rad = rad

        self.win = win
        self.circ = g.Circle(center, rad)
        self.circ.draw(win)

class Simulator:
    def __init__(self, win, start_pt, width, height):
        self.motion_noise = 0
        self.sense_noise = 0.0

        self.sense_max = 100
        self.sense_fov = 2 * pi / 3
        self.fov_markers = []

        # Start the robot at some provided position, and start it with
        # a heading of 0 (should be pointing east)
        self.robot_pos = start_pt

        self.width = width
        self.height = height

        # Hang on to the window handle
        self.win = win

        # Create robot
        self.topc = g.Point(self.robot_pos.x - 10, self.robot_pos.y + 15)
        self.botc = g.Point(self.robot_pos.x + 10, self.robot_pos.y - 15)
        self.robotrect = g.Rectangle(self.topc, self.botc)
        rmat = np.matrix([[self.robot_pos.x]
                         ,[self.robot_pos.y]
                         ])
        self.robotrect.draw(self.win)

    def set_landmarks(self, landmarks):
        self.landmarks = landmarks

    def do_motors(self, u, noise_xy, noise_s):
        u = np.array(u)
        noise = []
        for [delta], n in zip(u[:3], (noise_xy, noise_xy)):
            if delta == 0 or 0 == n:
                noise.append(0)
            else:
                noise.append(np.random.normal(0, sqrt(abs(delta * n))))

        u += np.matrix(noise).reshape((3, 1))
        math_dx, math_dy = (d for [d] in u[:2])
        math_dy = u[1][0]
        math_x = self.robot_pos.x + math_dx
        math_y = self.height - self.robot_pos.y + math_dy

        # the position of the robot in math frame
        math_pos = np.matrix([[math_x]
                             ,[math_y]])

        # compute the box points unrotated
        math_topc = np.matrix([[math_x - 10], [math_y + 15]])
        math_botc = np.matrix([[math_x + 10], [math_y - 15]])


        corners = [ np.matrix([[math_x - 10], [math_y + 15]])
                  , np.matrix([[math_x - 10], [math_y - 15]])
                  , np.matrix([[math_x + 10], [math_y - 15]])
                  , np.matrix([[math_x + 10], [math_y + 15]])
                  ]
        cpoints = (g.Point(x, self.height - y) for [x], [y] in corners)

        # redraw the box with the rotated box
        self.robotrect.undraw()
        self.robotrect = FourRectangle(cpoints, self.sense_max);
        rmat = np.matrix([[self.robot_pos.x]
                         ,[self.robot_pos.y]
                         ])
        print('redrawing robot at: ({}, {})\n'.format(math_x, math_y))
        self.robotrect.draw(self.win)

        # update the robot position
        self.robot_pos = g.Point(math_x, self.height - math_y)

        ret = {}
        robot_pos = vector.Vector2(self.robot_pos.x, self.robot_pos.y)
        for c in self.landmarks:
            c_pt = vector.Vector2(c.center.x, c.center.y)
            dist = (robot_pos - c_pt).length()
            if dist < self.sense_max:
                noise_x = np.random.normal(0, sqrt(noise_s))
                noise_y = np.random.normal(0, sqrt(noise_s))
                loclmvec = np.matrix([[c.center.x + noise_x], \
                                      [c.center.y + noise_y]]) - rmat
                ret[c.ident] = loclmvec[0, 0], -loclmvec[1, 0]

        return ret

    def get_true_state(self):
        ret = []

        ret.append([self.robot_pos.x])
        ret.append([self.robot_pos.y])

        for l in self.landmarks:
            ret.append([l.center.x])
            ret.append([l.center.y])

        return np.matrix(ret)
