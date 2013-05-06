import numpy as np
from time import sleep
from simulator import *
import graphics as g
import random
from math import sin, cos, sqrt
from vector import Vector2
from sys import exit
from threading import Lock

np.set_printoptions(linewidth=300,precision=4,suppress=True)

class Cobot:
    global width, height, lm
    def __init__(self, P, u=None, x=None):
        global nextident
        self.ident = nextident
        nextident += 1

        self.lm_ids = []
        self.P = P
        self.reverse = False
        self.lock = Lock()
        if u is None:
            self.u = np.zeros((3, 1))
        if x is None:
            self.x = np.zeros((3,1))
            self.x[0][0] = random.randrange(50, width-50)
            self.x[1][0] = random.randrange(50, width-50)
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

class BigCobot(Cobot):
    def __init__(self):
        self.cob_ids = []

    # do matrix stuff to add just one landmark
    def add_lm(self, cobot, lid):
        src_idx = cobot.lm_ids.index(lid)

        # append to the list of lm_ids
        dst_idx = len(lm_ids)
        lm_ids.append(lid)

        # append to x
        lx, ly = zip(cobot.x[3::2], cobot.x[4::2])[src_idx]
        self.x.append(lx)
        self.x.append(ly)

        # expand P, copy over the old matrix, and then set the appropriate
        # regions to keep covariances
        oldPsize = self.P.shape[0]
        newPsize = oldPsize + 2
        newP = np.zeros((newPsize, newPsize))
        newP[:oldPsize,:oldPsize] = self.P

        # here we want to copy over the covariance of this landmark with this
        # cobot
        src_row = 0
        src_end_row = 3
        src_col = 3 + src_idx * 2
        src_end_col = src_col + 2

        cob_idx = self.cob_ids.index(cobot.ident)
        dst_row = 3 * cob_idx
        dst_end_row = dst_row + 3
        dst_col = 3 * len(self.cob_ids) + 2 * dst_idx
        dst_end_col = dst_col + 2

        # copy over this region and its transpose
        newP[dst_row:dst_end_row,dst_col:dst_end_col] = cobot.P[src_row:src_end_row,src_col:src_end:col]
        newP[dst_col:dst_end_col,dst_row:dst_end_row] = cobot.P[src_col:src_end:col,src_row:src_end_row]

        # now we want to go through all the other landmarks that cobot knows
        # about, and copy the covariance of that other landmark with the
        # landmark we are trying to add
        for oth_lid in self.lm_ids:
            if oth_lid != lid and oth_lid in cobot.lm_ids:
                oth_src_idx = cobot.lm_ids.index(oth_lid)
                oth_dst_idx = self.lm_ids.index(oth_lid)

                src_row = 3 + oth_src_idx * 2
                src_end_row = oth_src_idx + 2
                src_col = 3 + src_idx * 2
                src_end_col = src_col + 2

                cob_idx = self.cob_ids.index(cobot.ident)
                dst_row = 3 * len(self.cob_ids) + 2 * oth_dst_idx
                dst_end_row = dst_row + 3
                dst_col = 3 * len(self.cob_ids) + 2 * dst_idx
                dst_end_col = dst_col + 2

                newP[dst_row:dst_end_row,dst_col:dst_end_col] = cobot.P[src_row:src_end_row,src_col:src_end:col]
                newP[dst_col:dst_end_col,dst_row:dst_end_row] = cobot.P[src_col:src_end:col,src_row:src_end_row]

    # here we actually append to our matrix, and set the covariances in the
    # big covariance matrix based off of the little ones
    def add_new(self, cobot):
        for lid in cobot.lm_ids:
            if lid not in self.lm_ids:
                self.add_lm(cobot, lid)

    def getHzR(self, update_list, cobot):
        zsize = len(update_list)
        z = np.zeros(zsize)
        R = np.zeros((zsize, zsize))
        H = np.zeros((zsize, self.x.shape[0]))

        # have to do several things here, all part of removing parts of the
        # cobot's state that were added by add_new and properly bringing
        # through any covariance information:
        # 1. Copy cobot pose from x to z
        # 2. Copy cobot pose covariance from P to R
        # 3. Copy landmark estimates from x to z
        # 4. Copy landmark variances
        # 5. Copy landmark covariances between robot pose and landmarks
        # 6. Copy landmark covariances between landmarks and other landmarks

        # also have to handle H
        # 1. Line up robot pose in z with robot pose in bigx
        # 2. Line up landmark pos in z with landmark pos in bigx

        # Steps 1 and 2
        z[0:3,0] = cobot.x[0:3,0]
        R[0:3,0:3] = cobot.P[0:3,0:3]

        # Set part of H corresponding to this robot's pose (step 1 for H)
        big_cob_xidx = 3 * self.cob_ids.index(cobot.ident)
        H[big_cob_xidx:big_cob_xidx+3,big_cob_xidx:big_cob_xidx+3] = np.identity(3)

        covered = set()
        for uidx, lm_id in enumerate(update_list):
            zidx = 3 + uidx * 2
            xidx = 3 + cobot.lm_ids.index(lm_id) * 2

            # Steps 3 and 4
            z[zidx:zidx+2,0] = cobot.x[xidx:xidx+2,0]
            R[zidx:zidx+2,zidx:zidx+2] = cobot.P[xidx:xidx+2,xidx:xidx+2]

            # Step 5
            R[0:3,zidx:zidx+2] = cobot.P[0:3,xidx:xidx+2]
            R[zidx:zidx+2,0:3] = cobot.P[xidx:xidx+2,0:3]

            # Step 6
            for olmidx, olm_id in enumerate(cobot.lm_ids):
                if olm_id != lm_id and olm_id in update_list and not covered((olm_id, lm_id)):
                    covered.add((olm_id, lm_id))
                    covered.add((lm_id, olm_id))

                    ozidx = 3 + update_list.index(olm_id) * 2
                    oxidx = 3 + olmidx * 2

                    R[ozidx:ozidx+2,zidx:zidx+2] = P[oxidx:oxidx+2,xidx:xidx+2]
                    R[zidx:zidx+2,ozidx:ozidx+2] = P[xidx:xidx+2,oxidx:oxidx+2]

            # set part of H corresponding to this landmark's position (step 2 for H)
            big_lm_xidx = 3 * len(self.cob_ids)
            H[zidx:zidx+2,big_lm_xidx:big_lm_xidx+2] = np.identity(2)

        return H, z, R

    def merge_estimates(self, cobots):
        for cobot in cobots:
            self.add_new(cobot)

        for cobot in cobots:
            update_list = []
            for lid in cobot.lm_ids:
                if lid in self.lm_ids:
                    update_list.append(lid)

            # Get the state and covariance of the cobot with only the
            # landmarks we have seen before. Name them z and R since
            # they are the measurement and measurement covariance for
            # this kalman update
            H, z, R = self.getHzR(update_list, cobot)

            self.x, self.P = kalman_update(H, R, self.P, self.x, z)
            print(P)

def getBigH(bigcobot, cobot):
    H = np.zeros(cobot.x.shapes[0], bigcobot.x.shapes[0])

    cob_idx = 3 * bigcobot.cob_ids.index(cobot.ident)
    H[0:3,cob_idx:cob_idx+3] = np.identity(3)

    for lid in cobot.lm_ids:
        lit_idx = 3 + cobot.lm_ids.index(lid) * 2
        big_idx = 3 * len(bigcobot.lm_ids) + bigcobot.lm_ids.index(lid) * 2

        H[lit_idx:lit_idx+2,big_idx:big_idx+2] = np.identity(2)

    return H

def strip_lm(lid, cobot):
    lidx = cobot.lm_ids.index(lid)

    # delete from lm_ids
    cobot.lm_ids.pop(lidx)

    # delete from state
    xlen = cobot.x.shape[0]
    newx = np.matrix(np.zeros(xlen-2))
    cutstart = 3 + 2 * lidx
    newx[:cutstart,0] = cobot.x[:custart,0]
    newx[cutstart+2:,0] = cobot.x[custart+2:,0]
    cobot.x = newx

    # delete from covariance matrix
    newP = np.zeros((xlen-2,xlen-2))
    newP[:cutstart,:cutstart] = cobot.P[:cutstart,:cutstart]
    newP[cutstart+2:,:cutstart] = cobot.P[cutstart+2:,:cutstart]
    newP[:cutstart,cutstart+2:] = cobot.P[:cutstart,cutstart+2:]
    newP[cutstart+2:,cutstart+2:] = cobot.P[cutstart+2:,cutstart+2:]
    cobot.P = np.matrix(newP)

def combine_estimates(cobots):
    if len(cobots) == 0:
        return None

    bigcobot = BigCobot()

    # find out how many landmarks there are
    bigcobot.lm_ids = []
    for cobot in cobots:
        bigcobot.lm_ids += cobot.lm_ids
    bigcobot.lm_ids = sorted(set(bigcobot.lm_ids))

    bigcobot.cob_ids = [ cobot.ident for cobot in cobots ]

    numlms = len(bigcobot.lm_ids)

    if numlms == 0:
        return None

    xsize = 3 * len(bigcobot.cob_ids) + 2 * numlms
    bigcobot.x = np.matrix(np.zeros((xsize, 1)))
    bigcobot.P = np.matrix(np.zeros((xsize, xsize)))
    for cobot in cobots:
        # 1. Update this robot's position right from the cobot's state estimate
        cob_idx = bigcobot.cob_ids.index(cobot.ident)
        bigcobot.x[3 * cob_idx: 3 * cob_idx + 3, 0] = cobot.x[0:3,0]

        # 2. Update the covariance of the cobot's state from the right spot 
        #    in its corresponding covariance matrix
        bigcobot.P[3 * cob_idx: 3 * cob_idx + 3,
                   3 * cob_idx: 3 * cob_idx + 3] = cobot.P[0:3,0:3]

        for lid in cobot.lm_ids:
            lit_idx = cobot.lm_ids.index(lid)
            big_idx = bigcobot.lm_ids.index(lid)

            if lid not in bigcobot.lm_ids:
                bigcobot.lm_ids.append(lid)

                lx, ly = strip_lm(lid, cobot)
                bigcobot.x.append(lx)
                bigcobot.x.append(ly)
            else:
                bigcobot.x[2*big_idx:2*big_idx+2,0] = cobot.x[2*lit_idx:2*lit_idx+2,0]

    return bigcobot

class EstimateDrawer:
    def __init__(self, win):
        self.win = win
        self.points = []
        self.states = []

    def draw_big(self, bigcobot):
        global height
        for x, P in self.states:
            for item in x:
                item.undraw()

            for item in P:
                item.undraw()

        states = [ (bigcobot.x, []) ]
        self.states = []

        colors = [ "purple" ]
        lm_scale = 5
        r_scale = 5
        for (x, P), color in zip(states, colors):
            x = np.array(x)
            P = np.array(P)
            p_diags = np.diag(P)

            state_x, state_P = [], []
            numcobs = len(bigcobot.cob_ids)
            for [px], [py], in zip(x[3*numcobs+3::2], x[3*numcobs+4::2]):
                c = g.Circle(g.Point(px, height - py), 20)
                c.setOutline(color)
                c.draw(self.win)
                state_x.append(c)

            self.states.append((state_x, state_P))

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
    P_p = F * P * F.transpose()# + Q
    return (x_p, P_p)

def kalman_update(H, R, P, x, z):
    y = z - H * x
    print("P")
    print(P)
    print("H")
    print(H)
    print("H*P")
    print(H*P)
    print("P*H^T")
    print(P.copy()*H.copy().transpose())
    S = H * P * H.transpose() + R
    print("S")
    print(S)
    K = P * H.transpose() * np.linalg.inv(S)
    x = x + K * y
    P = (np.identity(K.shape[0]) - K * H) * P
    return (x, P)

def getQ(x, u):
    noopu = np.matrix([[0]
                      ,[0]
                      ,[0]
                      ])
    ret = np.zeros((x.shape[0], x.shape[0]))

    if (noopu == u).all():
        return ret

    ret[0:3,0:3] = np.identity(3)

    return np.matrix(ret)

def getHzR(cobot, meas):
    zids = [ i for i in meas.keys() ]
    z = []
    for i in zids:
        z.append(meas[i][0])
        z.append(meas[i][1])

    H = np.zeros((2 * len(zids), 3 + 2 * len(cobot.lm_ids)))

    z = np.matrix([ [e] for e in z ])
    R = np.identity(z.shape[0])

    for lid in zids:
        zidx = zids.index(lid) * 2
        xidx = 3 + cobot.lm_ids.index(lid) * 2

        H[zidx:zidx+2,0:2] = np.identity(2) * (-1)
        H[zidx:zidx+2,xidx:xidx+2] = np.identity(2)
        
        for oi, olid in enumerate(zids):
            if olid != lid:
                ozidx = oi * 2
                R[zidx : zidx + 2, ozidx : ozidx + 2] = 2* np.identity(2) 

    #return np.matrix(H), z, np.matrix(R)
    return np.matrix(H), z, np.shape(np.zeros((z.shape[0],z.shape[0])))

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
    global width, height, lm, meas_uncertainty, nextident
    nextident = 0
    meas_uncertainty = 10
    width = 800
    height = 800
    win = g.GraphWin('SLAM Simulator', width, height)

    lm_points = []
    for i in range(30):
        lx = random.randrange(50, width-50)
        ly = random.randrange(50, width-50)
        lm_points.append(g.Point(lx, ly))

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
            print("Acquiring lock (timestep)", cobot.ident)
            with cobot.lock:
                meas, simx = sim.do_motors(cobot.u, cobot.reverse)
                print("le robot thinks u = ")
                print(cobot.u)
                print("robot estimate of itself = ")
                print(cobot.x[:3])
                print("sim robot pos= ")
                print(simx)

                if (cobot.x[:3] != simx).all():
                    print("Exiting because states are not equal")
                    exit()

                # add previously unseen landmarks to the state
                cobot.add_new_landmarks(meas)
                print("add_new_landmarks")

                # get matrices for kalman predict
                Q = getQ(cobot.x, cobot.u)
                F = getF(cobot.x)
                G = getG(cobot.x, cobot.u)
                print("did matrices")
                cobot.x, cobot.P = kalman_predict(F, G, Q, cobot.P, cobot.x, cobot.u)

                print("did predict")
                if not meas or True:
                    print("Releasing lock (timestep)", cobot.ident)
                    continue

                # compute H, z, and R
                H, z, R = getHzR(cobot, meas)
                cobot.x, cobot.P = kalman_update(H, R, cobot.P, cobot.x, z)


        ed.draw((cobot.x, cobot.P) for cobot, _ in cobot_sim)

    def makeGo(cobot):
        def go(event):
            # set velocities
            print("Acquiring lock (makeGo)", cobot.ident)
            with cobot.lock:
                if event.keysym in ('Up', 'w', 'Down', 's'):
                    cobot.u[0][0] = 2 * cos(cobot.x[2][0])
                    cobot.u[1][0] = 2 * sin(cobot.x[2][0])
                    cobot.u[2][0] = 0
                    if event.keysym in ('Up', 'w'):
                        cobot.reverse = False
                    if event.keysym in ('Down', 's'):
                        cobot.reverse = True
                elif event.keysym in ('Left', 'Right', 'a', 'd'):
                    cobot.u[0][0] = 0
                    cobot.u[1][0] = 0
                    if event.keysym in ('Left', 'a'):
                        cobot.u[2][0] = 0.05
                    if event.keysym in ('Right', 'd'):
                        cobot.u[2][0] = -0.05
            print("Releasing lock (makeGo)", cobot.ident)
        return go
    
    def makeStop(cobot):
        with cobot.lock:
            def stop(event):
                cobot.u = np.zeros((3,1))
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
    win.pack()
    win.focus_set()

    #bigcobot = BigCobot()
    #ed2 has a method draw_big which isn't working..
    ed2 = EstimateDrawer(win)
    iters = 0
    while True:
        timestep()

        sleep(0.01)

if __name__ == "__main__":
    main()
