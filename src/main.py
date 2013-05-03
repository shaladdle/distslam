import numpy as np
from time import sleep
from simulator import *
import graphics as g
import random
from math import sin, cos, sqrt
from vector import Vector2

np.set_printoptions(linewidth=300,precision=4,suppress=True)

class Cobot:
    global width, height, lm
    def __init__(self, P, u=None, x=None):
        self.lm_ids = []
        self.P = P
        self.reverse = False
        if u is None:
            self.u = np.zeros((3, 1))
        if x is None:
            self.x = np.zeros((3,1))
            self.x[0][0] = 200 + random.randrange(-width / 2, width / 2)
            self.x[1][0] = 200 # height
            self.x[2][0] = 0

    def add_new_landmarks(self, meas):
        xtmp = self.x.copy()
        xtmp = [ e for [e] in np.array(xtmp) ]

        x_r = xtmp[0]
        y_r = xtmp[1]

        numadded = 0
        for lid in list(meas.keys()):
            # if the id is not already in the state (this landmark has never 
            # been seen before)
            if lid not in self.lm_ids:
                numadded += 2

                # 1. add it to the current state vector to bring it into
                #    world frame
                xtmp.append(x_r + meas[lid][0])
                xtmp.append(y_r + meas[lid][1])

                # 2. add it to lm_ids
                self.lm_ids.append(lid)

                # 3. add it to the covariance matrix
                newP = np.identity(self.x.shape[0] + numadded) * meas_uncertainty
                newP[:self.P.shape[0],:self.P.shape[1]] = self.P

                relmat = np.matrix([[1, 0, 1]
                                   ,[0, 1, 1]
                                   ])

                lidx = self.lm_ids.index(lid)
                i = 3 + lidx * 2
                # 4. add some covariance with the robot's pose
                newP[i:i+2,0:3] = relmat
                newP[0:3,i:i+2] = relmat.transpose()

                self.P = newP

                # 5. remove it from the measurement dict
                del meas[lid]

        xtmp = [ [e] for e in xtmp ]
        xtmp = np.matrix(xtmp)
        xtmp.reshape(self.x.shape[0] + numadded, 1)

        self.x = xtmp

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

        colors = ["red", "blue"]
        lm_scale = 5
        r_scale = 5
        for (x, P), color in zip(states, colors):
            x = np.array(x)
            P = np.array(P)
            p_diags = np.diag(P)

            covx, covy = (p * r_scale for p in p_diags[:2])
            [rx], [ry], rt = (p for p in x[:3])
            rpoint1 = g.Point(rx - covx, height - (ry - covy))
            rpoint2 = g.Point(rx + covx, height - (ry + covy))
            r = g.Oval(rpoint1, rpoint2)
            r.setOutline(color)
            r.draw(self.win)

            t_scale = (covx + covy) / 2
            b = g.Point(rx + t_scale * 1.2 * cos(-rt),
                    height - ry + t_scale * 1.2 * sin(-rt))
            rl = g.Line(g.Point(rx + (t_scale * 0.8) * cos(-rt), 
                height - ry + (t_scale * 0.8) * sin(-rt)), b)
            rl.setOutline(color)
            rl.draw(self.win)

            state_x, state_P = [], []
            state_x.append(r)
            state_x.append(rl)
            ellipse_data = zip(x[3::2], x[4::2], p_diags[3::2], p_diags[4::2])

            for [px], [py], *cov in ellipse_data:
                covx, covy = (p * lm_scale for p in cov)
                p1 = Vector2(px - covx, height - (py - covy))
                p2 = Vector2(px + covx, height - (py + covy))
                c = g.Oval(g.Point(p1[0], p1[1]), g.Point(p2[0], p2[1]))
                c.setOutline(color)
                c.draw(self.win)
                state_x.append(c)

            self.states.append((state_x, state_P))

def kalman_predict(F, G, Q, P, x, u):
    x_p = F * x + G * u
    P_p = F * P * F.transpose() + Q
    return (x_p, P_p)

def kalman_update(H, R, P, x, z):
    y = z - H * x
    S = H * P * H.transpose() + R
    K = P * H.transpose() * np.linalg.inv(S)
    x = x + K * y
    P = (np.identity(K.shape[0]) - K * H) * P

    return (x, P)

def getQ(x):
    ret = np.zeros((x.shape[0], x.shape[0]))
    ret[0:3,0:3] = np.identity(3)
    return np.matrix(ret)

def getHzR(cobot, meas):
    zids = [ i for i in meas.keys() ]
    z = []
    for i in zids:
        z.append(meas[i][0])
        z.append(meas[i][1])

    H = np.zeros((2 * len(zids), 3 + 2 * len(cobot.lm_ids)))
    for lid in zids:
        zidx = zids.index(lid) * 2
        xidx = 3 + cobot.lm_ids.index(lid) * 2

        H[zidx:zidx+2,0:2] = np.identity(2) * (-1)
        H[zidx:zidx+2,xidx:xidx+2] = np.identity(2)

    z = np.matrix([ [e] for e in z ])

    R = 2 * np.identity(z.shape[0]) + np.ones((z.shape[0],z.shape[0]))

    return np.matrix(H), z, np.matrix(R)

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

def main():
    global width, height, lm, meas_uncertainty
    meas_uncertainty = 10
    width = 400
    height = 400
    win = g.GraphWin('SLAM Simulator', width, height)


    lm_points = [ g.Point(30,  40)
                , g.Point(25,  10)
                , g.Point(300, 200)
                , g.Point(100, 200)
                , g.Point(150, 200)
                , g.Point(200, 200)
                , g.Point(250, 200)
                , g.Point(50,  230)
                , g.Point(100, 20)
                ]

    lm = []
    for i, lmp in enumerate(lm_points):
        lm.append(Landmark(win, i, 10, lmp))

    # set initial state and covariance matrix
    numbots = 2
    P = np.matrix(np.identity((3)))
    cobot_sim = []
    for _ in range(numbots):
        cobot = Cobot(P)
        sim = Simulator(win, g.Point(cobot.x[0,0], height - cobot.x[1,0]), - cobot.x[2,0], width, height)
        sim.set_landmarks(lm)
        cobot_sim.append((cobot, sim))

    # this shit should be functions, not a class
    ed = EstimateDrawer(win)
    ed.draw((cobot.x, cobot.P) for cobot, _ in cobot_sim)

    def timestep():
        # Close the application if someone closes the graphics window
        if win.isClosed():
            return

        for cobot, sim in cobot_sim:
            meas = sim.do_motors(cobot.u, cobot.reverse)

            # add previously unseen landmarks to the state
            cobot.add_new_landmarks(meas)

            # get matrices for kalman predict
            Q = getQ(cobot.x)
            F = getF(cobot.x)
            G = getG(cobot.x, cobot.u)
            cobot.x, cobot.P = kalman_predict(F, G, Q, cobot.P, cobot.x, cobot.u)

            if not meas:
                continue

            # compute H, z, and R
            H, z, R = getHzR(cobot, meas)
            cobot.x, cobot.P = kalman_update(H, R, cobot.P, cobot.x, z)

        ed.draw((cobot.x, cobot.P) for cobot, _ in cobot_sim)

    def makeGo(cobot):
        def go(event):
            # set velocities
            if event.keysym in ('Up', 'w', 'Down', 's'):
                cobot.u[1][0] = .4 * sin(cobot.x[2][0])
                cobot.u[0][0] = .4 * cos(cobot.x[2][0])
                if event.keysym in ('Up', 'w'):
                    cobot.reverse = False
                if event.keysym in ('Down', 's'):
                    cobot.reverse = True

            elif event.keysym in ('a', 'd', 'Left', 'Right'):
                disp = np.linalg.norm(cobot.u[:2])
                cobot.u[1][0] = disp * sin(cobot.x[2][0])
                cobot.u[0][0] = disp * cos(cobot.x[2][0])
        return go
    
    def makeStop(cobot):
        def stop(event):
            if event.keysym in ('Up', 'w', 'Down', 's'):
                cobot.u[:2] = np.zeros((2, 1))
            if event.keysym in ('Left', 'Right', 'a', 'd'):
                cobot.u[2][0] = 0
        return stop

    def makeTurn(cobot, theta, go):
        def turn(event):
            cobot.u[2][0] = theta
            go(event)
        return turn

    cobots = list(zip(*cobot_sim))[0]

    go = makeGo(cobots[1])
    win.bind("<KeyPress-w>", go)
    win.bind("<KeyPress-s>", go)
    win.bind("<KeyPress-a>", makeTurn(cobots[1], .05, go))
    win.bind("<KeyPress-d>", makeTurn(cobots[1], -.05, go))
    stop = makeStop(cobots[1])
    win.bind("<KeyRelease-w>", stop)
    win.bind("<KeyRelease-s>", stop)
    win.bind("<KeyRelease-a>", stop)
    win.bind("<KeyRelease-d>", stop)

    go = makeGo(cobots[0])
    win.bind("<KeyPress-Up>", go)
    win.bind("<KeyPress-Down>", go)
    win.bind("<KeyPress-Left>", makeTurn(cobots[0], .05, go))
    win.bind("<KeyPress-Right>", makeTurn(cobots[0], -.05, go))
    stop = makeStop(cobots[0])
    win.bind("<KeyRelease-Up>", stop)
    win.bind("<KeyRelease-Down>", stop)
    win.bind("<KeyRelease-Left>", stop)
    win.bind("<KeyRelease-Right>", stop)
    win.pack()
    win.focus_set()

    iters = 0
    while True:
        timestep()

        sleep(0.01)

        iters += 1
        if not iters % 50:
            print(cobots[0].P)
            print(cobots[0].x)

if __name__ == "__main__":
    main()
