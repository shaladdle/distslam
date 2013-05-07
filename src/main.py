import numpy as np
from time import sleep
from simulator import *
import graphics as g
import random
from math import sin, cos, sqrt
from vector import Vector2
from sys import exit
from threading import Lock
import traceback

np.set_printoptions(linewidth=300,precision=4,suppress=True)

class Cobot:
    global width, height, lm
    def __init__(self, P, u=None, x=None):
        global nextident
        self.ident = nextident
        nextident += 1

        self.lm_ids = []
        self.P = P
        self.lock = Lock()
        if u is None:
            self.u = np.zeros((2, 1))
        else:
            self.u = u
        if x is None:
            self.x = np.zeros((2, 1))
            self.x[0][0] = random.randrange(50, width-50)
            self.x[1][0] = random.randrange(50, width-50)
        else:
            self.x = x

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

                relmat = np.identity(2)

                lidx = self.lm_ids.index(lid)
                i = 2 + lidx * 2
                # 4. add some covariance with the robot's pose
                newP[i:i+2,0:2] = relmat
                newP[0:2,i:i+2] = relmat.transpose()

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
        lm_scale = 1
        r_scale = 1
        for (x, P), color in zip(states, colors):
            x = np.array(x)
            P = np.array(P)
            p_diags = np.diag(P)

            covx, covy = (p * r_scale for p in p_diags[:2])
            [rx], [ry] = (p for p in x[:2])
            rpoint1 = g.Point(rx - covx, height - (ry - covy))
            rpoint2 = g.Point(rx + covx, height - (ry + covy))
            r = g.Oval(rpoint1, rpoint2)
            r.setOutline(color)
            print('drawing {} robot at ({}, {})'.format(color, rx, ry))
            r.draw(self.win)

            state_x, state_P = [], []
            state_x.append(r)
            ellipse_data = zip(x[2::2], x[3::2], p_diags[2::2], p_diags[3::2])

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
    print('F:\n{}\nG:\n{}\nQ:\n{}\nP:\n{}\nx:\n{}\nu:\n{}\n'.format(F, G, Q, P, x, u))
    x_p = F * x + G * u
    P_p = F * P * F.transpose() + Q
    print('predicted x:\n{}\n\npredicted P:\n{}\n'.format(x_p, P_p))
    return (x_p, P_p)

def kalman_update(H, R, P, x, z):
    print('#################################\n          kalman_update\n')
    #print("kalman update")
    print('inputs:\n')
    print('H:\n{}\nR:\n{}\nP\n{}\nx:{}\nz:{}\n'.format(H, R, P, x, z))
    v = z - H * x
    print('calculated v:\n{}'.format(v))
    S = H * P * H.transpose() + R
    print('calculated S:\n{}'.format(S))
    W = P * H.transpose() * np.linalg.inv(S)
    print('updated W:\n{}\n'.format(W))
    print('1 - W * H:\n{}\n'.format(np.identity(W.shape[0]) - W * H))
    x = x + W * v
    print('updated x:\n{}\n'.format(x))
    #P = P - W*S*W.transpose()#(np.identity(K.shape[0]) - K * H) * P
    P = (np.identity(W.shape[0]) - W * H) * P
    print('updated P:\n{}\n'.format(P))
    return x, P

def getQ(x, u):
    noopu = np.matrix([[0]
                      ,[0]
                      ])
    ret = np.zeros((x.shape[0], x.shape[0]))

    if (noopu == u).all():
        return ret

    global noise_xy
    variances = (noise_xy, noise_xy)
    variances = [v * abs(delta) for v, [delta] in zip(variances, u[:2])]
    print('variances: {}\n'.format(variances))

    ret[0:2,0:2] = np.diag(variances)

    return np.matrix(ret)

def getHzR(cobot, meas):
    zids = [ i for i in meas.keys() ]
    z = []
    for i in zids:
        z.append(meas[i][0])
        z.append(meas[i][1])

    H = np.zeros((2 * len(zids), 2 + 2 * len(cobot.lm_ids)))

    z = np.matrix([ [e] for e in z ])

    global noise_s
    zsize = z.shape[0]
    measv = [noise_s for _ in range(zsize)]
    R = np.diag(measv)

    for lid in zids:
        zidx = zids.index(lid) * 2
        xidx = 2 + cobot.lm_ids.index(lid) * 2

        H[zidx:zidx+2,0:2] = np.identity(2) * (-1)
        H[zidx:zidx+2,xidx:xidx+2] = np.identity(2)
        
        for oi, olid in enumerate(zids):
            if olid != lid:
                ozidx = oi * 2
                R[zidx : zidx + 2, ozidx : ozidx + 2] = 2* np.identity(2) 

    return np.matrix(H), z, np.matrix(R)

def getG(x, u):
    ret = np.zeros((u.shape[0], x.shape[0]))
    ret[0:2,0:2] = np.identity(2)

    return np.matrix(ret)

def getF(x):
    return np.matrix(np.identity(x.shape[0]))

def main():
    global width, height, lm, \
            meas_uncertainty, nextident, \
            noise_xy, noise_s
    nextident = 0
    meas_uncertainty = 10
    width = 800
    height = 800
    noise_xy = .05
    noise_s = 1
    win = g.GraphWin('SLAM Simulator', width, height)

    lm_points = []
    for i in range(30):
        lx = random.randrange(50, width-50)
        ly = random.randrange(50, height-50)
        lm_points.append(g.Point(lx, ly))

    lm = []
    for i, lmp in enumerate(lm_points):
        lm.append(Landmark(win, i, 10, lmp))

    # set initial state and covariance matrix
    numbots = 2
    # TODO Ideally this should work fine, but I'm not sure
    P = np.matrix(np.zeros((2,2)))
    cobot_sim = []
    for _ in range(numbots):
        cobot = Cobot(P)
        sim = Simulator(win, g.Point(cobot.x[0,0], height - cobot.x[1,0]), width, height)
        sim.set_landmarks(lm)
        cobot_sim.append((cobot, sim))

    ed = EstimateDrawer(win)
    ed.draw((cobot.x, cobot.P) for cobot, _ in cobot_sim)

    def timestep():
        global time
        print('XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX\n         TIMESTEP {}'.format(time))
        time += 1
        # Close the application if someone closes the graphics window
        if win.isClosed():
            return

        local_states = []
        for robot_i, (cobot, sim) in enumerate(cobot_sim):
            print('***********************')
            print('robot {}\n'.format(robot_i))
            # startup
            with cobot.lock:
                local_u = cobot.u.copy()
                local_x = cobot.x.copy()

            global noise_xy, noise_s, height
            meas, simx = sim.do_motors(local_u, noise_xy, noise_s)
            print('robot_pos: ({}, {})\n'.format(sim.robot_pos.x, height - sim.robot_pos.y))
            # add previously unseen landmarks to the state
            try:
                with cobot.lock:
                    cobot.add_new_landmarks(meas)
                    local_x = cobot.x
                    local_P = cobot.P
            except ValueError as e:
                traceback.print_exc()
                print(e)
                print(meas)
                print(cobot.x)
                print(cobot.P)
                exit()

            # get matrices for kalman predict
            Q = getQ(local_x, local_u)
            F = getF(local_x)
            G = getG(local_x, local_u)
            local_x, local_P = kalman_predict(F, G, Q, local_P, local_x, local_u)
            print('updated x:\n{}\n\nupdated P:\n{}\n'.format(local_x, local_P))
            
            if meas:
                try:
                    # compute H, z, and R
                    H, z, R = getHzR(cobot, meas)
                    local_x, local_P = kalman_update(H, R, local_P, local_x, z)
                except ValueError as e:
                    traceback.print_exc()
                    print(H)
                    print(z)
                    print(R)
                    exit()
            
            local_states.append((local_x, local_P))

            # cleanup
            with cobot.lock:
                cobot.u = local_u
                cobot.x = local_x
                cobot.P = local_P

        ed.draw((x, P) for x, P in local_states)

    class BadEventException(Exception):
        'For events not recognized by an event handler'
        pass

    def makeGo(cobot):
        def go(event):
            # set velocities
            with cobot.lock:
                if event.keysym in ('Up', 'w'):
                    cobot.u[0][0] = 0
                    cobot.u[1][0] = 25
                elif event.keysym in ('Down', 's'):
                    cobot.u[0][0] = 0
                    cobot.u[1][0] = -25
                elif event.keysym in ('Left', 'a'):
                    cobot.u[0][0] = -25
                    cobot.u[1][0] = 0
                elif event.keysym in ('Right', 'd'):
                    cobot.u[0][0] = 25
                    cobot.u[1][0] = 0
                else:
                    raise BadEventException('event {} not recognized'.format(event.keysym))
        return go

    def makeStop(cobot):
        def stop(event):
            with cobot.lock:
                cobot.u = np.zeros((2,1))
        return stop

    cobots = list(zip(*cobot_sim))[0]

    go = makeGo(cobots[1])
    win.bind("<KeyPress-w>", go)
    win.bind("<KeyPress-s>", go)
    win.bind("<KeyPress-a>", go)
    win.bind("<KeyPress-d>", go)
    stop = makeStop(cobots[1])
    win.bind("<KeyRelease-w>", stop)
    win.bind("<KeyRelease-s>", stop)
    win.bind("<KeyRelease-a>", stop)
    win.bind("<KeyRelease-d>", stop) 
    go = makeGo(cobots[0])
    win.bind("<KeyPress-Up>", go)
    win.bind("<KeyPress-Down>", go)
    win.bind("<KeyPress-Left>", go)
    win.bind("<KeyPress-Right>", go)
    stop = makeStop(cobots[0])
    win.bind("<KeyRelease-Up>", stop)
    win.bind("<KeyRelease-Down>", stop)
    win.bind("<KeyRelease-Left>", stop)
    win.bind("<KeyRelease-Right>", stop)

    #timestep
    global time
    time = 0
    win.bind("<KeyRelease-1>", lambda e: timestep())
    win.pack()
    win.focus_set()

    #bigcobot = BigCobot()
    #ed2 has a method draw_big which isn't working..
    ed2 = EstimateDrawer(win)
    iters = 0
    while True:
        win.update()

if __name__ == "__main__":
    main()
