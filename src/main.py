import numpy as np
from time import sleep
from simulator import *
import graphics as g
import random

class Cobot:
    global width, height, lm
    def __init__(self, P, u=None, x=None):
        self.P = P
        if u is None:
            self.u = np.zeros((3, 1))
            self.u.fill(1)
        if x is None:
            self.x = np.zeros((len(lm) * 2 + 3,1))
            self.x[0][0] = 200 + random.randrange(-width / 2, width / 2)
            self.x[1][0] = 300 # height
            self.x[2][0] = 0

class EstimateDrawer:
    def __init__(self, win):
        self.win = win
        self.points = []
        self.states = []

    def draw(self, states):
        global height
        for x, P in self.states:
            for item in x:
                item.undraw()

            for item in P:
                item.undraw()

        self.states = []

        colors = [ "red", "blue" ]
        for (x, P), color in zip(states, colors):
            x = np.array(x)
            P = np.array(P)

            state_x, state_P = [], []
            for i in range(3, (x.shape[0] - 3) // 2):
                c = g.Circle(g.Point(x[i*2][0], height - x[i*2+1][0]), 10)
                c.setOutline(color)
                c.draw(self.win)
                state_x.append(c)

            self.states.append((state_x, state_P))


def kalman_predict(F, G, P, x, u):
    x_p = F * x + G * u
    P_p = F * P * F.transpose()
    return (x_p, P_p)

def kalman_update(H, R, P, x, z):
    y = z - H * x
    S = H * P * H.transpose() + R
    K = P * H.transpose() * np.linalg.inv(S)
    x = x + K * y
    P = (np.identity(K.shape[0]) - K * H) * P

    return (x, P)

# All these helper functions pretty much just use the matrix sizes
def getR(z):
    return np.matrix(np.identity(z.shape[0]))

def getH(x, z):
    a = np.zeros((z.shape[0], 3))
    b = np.identity(z.shape[0])
    ret = np.concatenate((a,b), axis=1)

    return np.matrix(ret)

def getG(x, u):
    ret = [[1,0,0]
          ,[0,1,0]
          ,[0,0,1]]

    for i in range(x.shape[0] - 3):
        ret.append([0, 0, 0])

    return np.matrix(ret)

def getF(x):
    return np.matrix(np.identity(x.shape[0]))

if __name__ == "__main__":
    global width, height, lm
    width = 400
    height = 400
    win = g.GraphWin('SLAM Simulator', width, height)

    lm_points = [ g.Point(30,  40),
                  g.Point(25,  10),
                  g.Point(300, 200),
                  g.Point(200, 150),
                  g.Point(50,  230),
                  g.Point(100, 20)
                ]

    lm = []
    for i in range(len(lm_points)):
        lm.append(Landmark(win, i, 10, lm_points[i]))

    print(len(lm))


    # set initial state and covariance matrix
    numbots = 2
    P = np.identity(len(lm) * 2 + 3) * 10
    P[0][0] = 100
    P[1][1] = 100
    P[2][2] = 100
    P = np.matrix(P)
    cobot_sim = []
    for _ in range(numbots):
        cobot = Cobot(P)
        coords = np.array(cobot.x)
        sim = Simulator(win, g.Point(coords[0][0], coords[1][0]), coords[2][0], width, height)
        sim.set_landmarks(lm)
        cobot_sim.append((cobot, sim))

    # this shit should be functions, not a class
    ed = EstimateDrawer(win)

    while(1):
        # Close the application if someone closes the graphics window
        if win.isClosed():
            break

        for cobot, sim in cobot_sim:
            sim.do_motors(cobot.u)
            z = sim.sense()

            # Get matrices for this iteration
            F = getF(cobot.x)
            G = getG(cobot.x, cobot.u)
            cobot.x, cobot.P = kalman_predict(F, G, cobot.P, cobot.x, cobot.u)

            H = getH(cobot.x, z)
            R = getR(z)
            cobot.x, cobot.P = kalman_update(H, R, cobot.P, cobot.x, z)

            print('cobot.x:')
            print(cobot.x)

        sleep(1)
        ed.draw((cobot.x, cobot.P) for cobot, _ in cobot_sim)
