import graphics as g
import numpy as np
from math import cos, sin

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
        u += self.motion_noise * np.random.randn(3,1)
        print("u after noise " + str(u))

        print("self.robot_pos.x " + str(self.robot_pos.x))
        print("self.robot_pos.y " + str(self.robot_pos.y))

        # math coordinates
        math_x = self.robot_pos.x + u[0][0]
        math_y = -self.robot_pos.y + u[1][0]
        math_hdg = -self.robot_hdg + u[2][0]
        print("math_hdg\n" + str(math_hdg))

        rotmat = np.matrix([[cos(math_hdg), -sin(math_hdg)]
                           ,[sin(math_hdg),  cos(math_hdg)]
                           ])
        print("rotmat\n" + str(rotmat))

        # the position of the robot in math frame
        math_pos = np.matrix([[math_x]
                             ,[math_y]])
        print("math_pos\n" + str(math_pos))

        # compute the box points unrotated
        math_topc = np.matrix([[math_x - 10], [math_y + 15]])
        math_botc = np.matrix([[math_x + 10], [math_y - 15]])
        print("math_topc\n" + str(math_topc))
        print("math_botc\n" + str(math_botc))
        print("before rotation")

        # put the points at the origin
        math_topc -= math_pos
        math_botc -= math_pos

        # rotate around the origin
        math_topc = rotmat * math_topc
        math_botc = rotmat * math_botc

        # put the points back to wherever they go
        math_topc += math_pos
        math_botc += math_pos
        print("math_topc\n" + str(math_topc))
        print("math_botc\n" + str(math_botc))
        print("after rotation")

        # graphics coordinates for the box we will draw
        math_topc = np.array(math_topc)
        math_botc = np.array(math_botc)

        # turn into points in the graphics frame
        topc_pt = g.Point(math_topc[0][0], -math_topc[1][0])
        botc_pt = g.Point(math_botc[0][0], -math_botc[1][0])
        print("topc_pt " + str(topc_pt.x) + ", " + str(topc_pt.y))
        print("botc_pt " + str(botc_pt.x) + ", " + str(botc_pt.y))

        # update the robot position
        self.robot_pos = g.Point(math_x, -math_y)
        self.robot_hdg += -math_hdg

        # redraw the box with the rotated box
        self.robotrect.undraw()
        self.robotrect = g.Rectangle(topc_pt, botc_pt)
        self.robotrect.draw(self.win)

    def sense(self):
        ret = []
        for l in self.landmarks:
            ret.append([l.center.x])
            ret.append([l.center.y])

        return np.matrix(ret) + (self.sense_noise * np.random.randn(len(self.landmarks) * 2,1))

    def get_true_state(self):
        ret = []

        ret.append([self.robot_pos.x])
        ret.append([self.robot_pos.y])

        for l in self.landmarks:
            ret.append([l.center.x])
            ret.append([l.center.y])

        return np.matrix(ret)
