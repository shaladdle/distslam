import graphics as g
import numpy as np
from math import cos, sin, pi, sqrt, acos
import vector

class FourRectangle:
    def __init__(self, points):
        tl, bl, br, tr = points;
        self.lines = [ g.Line(tl,bl)
                     , g.Line(bl,br)
                     , g.Line(br,tr)
                     , g.Line(tr,tl)
                     ]
        self.eye = g.Point((tr.x + br.x) / 2.0, (tr.y + br.y) / 2.0)
        self.front = g.Circle(self.eye, 3.0)
        self.fov = pi / 3.0 # degrees or radians?
        self.sight_range = 10 # temp for now
        self.sight_lines = [g.Line(self.eye,
            g.Point(self.sight_range * cos(self.fov), 
                self.sight_range * sin(self.fov)))]

    def draw(self, win):
        for l in self.lines:
            l.draw(win)
        self.front.color = "red"
        self.front.draw(win)
        for l in self.sight_lines:
            l.draw(win)

    def undraw(self):
        for l in self.lines:
            l.undraw()
        self.front.undraw()
        for l in self.sight_lines:
            l.undraw()

def addP(a, b):
    return g.Point(a.x + b.x, a.y + b.y)

class Landmark:
    def __init__(self, win, ident, rad, center):
        self.ident = ident
        self.center = center
        self.rad = rad

        self.win = win
        self.circ = g.Circle(center, rad)
        self.circ.draw(win)

class Simulator:
    def __init__(self, win, start_pt, start_hdg, width, height):
        self.motion_noise = 0.2
        self.sense_noise = 0.01

        # Start the robot at some provided position, and start it with
        # a heading of 0 (should be pointing east)
        self.robot_pos = start_pt
        self.robot_hdg = start_hdg

        self.width = width
        self.height = height

        # Hang on to the window handle
        self.win = win

        # Create robot
        self.topc = g.Point(self.robot_pos.x - 10, self.robot_pos.y + 15)
        self.botc = g.Point(self.robot_pos.x + 10, self.robot_pos.y - 15)
        self.robotrect = g.Rectangle(self.topc, self.botc)
        self.robotrect.draw(self.win)

    def set_landmarks(self, landmarks):
        self.landmarks = landmarks

    def do_motors(self, u):
        u = np.array(u)
        noise = np.random.randn(3, 1)
        noise[2][0] /= 10
        if (u[0][0] == u[1][0] == 0):
            noise[0][0] = 0
            noise[1][0] = 0
        if (u[2][0] == 0):
            noise[2][0] = 0
            
        u += self.motion_noise * noise

        # math coordinates
        disp = np.linalg.norm(u[:2])
        math_hdg = -self.robot_hdg + u[2][0]
        math_dx = disp * cos(math_hdg)
        math_dy = disp * sin(math_hdg)
        math_x = self.robot_pos.x + math_dx
        math_y = self.height - self.robot_pos.y + math_dy

        rotmat = np.matrix([[cos(math_hdg), -sin(math_hdg)]
                           ,[sin(math_hdg),  cos(math_hdg)]
                           ])

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

        corners_local_frame =    (corner - math_pos for corner in corners)
        corners_local_rot =      (rotmat * corner   for corner in corners_local_frame)
        corners_rot_glob_frame = (corner + math_pos for corner in corners_local_rot)
        corners_arr_rot_glob   = (np.array(corner)  for corner in corners_rot_glob_frame)
        cpoints =                (g.Point(x, self.height - y) for [x], [y] in corners_arr_rot_glob)

        # redraw the box with the rotated box
        self.robotrect.undraw()
        self.robotrect = FourRectangle(cpoints);
        self.robotrect.draw(self.win)

        # update the robot position
        self.robot_pos = g.Point(math_x, self.height - math_y)
        self.robot_hdg = -math_hdg % (2 * pi)

    def sense(self):
        ret = []
        for l in self.landmarks:
            ret.append([l.center.x])
            ret.append([self.height - l.center.y])

        return np.matrix(ret) + (self.sense_noise * np.random.randn(len(self.landmarks) * 2,1))

    def get_true_state(self):
        ret = []

        ret.append([self.robot_pos.x])
        ret.append([self.robot_pos.y])

        for l in self.landmarks:
            ret.append([l.center.x])
            ret.append([l.center.y])

        return np.matrix(ret)
