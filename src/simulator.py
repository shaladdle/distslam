import graphics as g
import numpy as np

def addP(a, b):
    return g.Point(a.x + b.x, a.y + b.y)

class Simulator:
    def __init__(self, win, start_pt, width, height):

        self.motion_noise = 0.2
        self.sense_noise = 0.01

        # Start the robot at some provided position, and start it with
        # a heading of 0 (should be pointing east)
        self.robot_pos = start_pt
        self.robot_hdg = 0 

        self.width = width
        self.height = height

        # Hang on to the window handle
        self.win = win

        # Create robot
        self.topc = g.Point(self.robot_pos.x - 10, self.robot_pos.y + 15)
        self.botc = g.Point(self.robot_pos.x + 10, self.robot_pos.y - 15)
        self.robotrect = g.Rectangle(self.topc, self.botc)
        self.robotrect.draw(self.win)

    def do_motors(self, u):
        u += self.motion_noise * np.random.randn(2,1)
        dx = u[0][0]
        dy = -u[1][0]
        self.robot_pos = addP(self.robot_pos, g.Point(dx, dy))
        self.robotrect.move(dx, dy)

    def sense(self):
        ret = np.zeros((2,1))
        ret[0][0] = self.robot_pos.x
        ret[1][0] = self.robot_pos.y

        return ret + (self.sense_noise * np.random.randn(2,1))
