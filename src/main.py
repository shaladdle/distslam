import numpy as np
from time import sleep
from simulator import *
import graphics as g

def kalman_predict(F, G, P, x, u):
    x_p = F * x + G * u
    P_p = F * P * F.transpose()
    return (x_p, P_p)

def kalman_update(H, R, P, x, z):
    y = z - H * x
    S = H * P * H.transpose() + R
    print(S)
    K = P * H.transpose() * np.linalg.inv(S)
    x = x + K * y
    P = (np.identity(K.shape[0]) - K * H) * P

    return (x, P)

# All these helper functions pretty much just use the matrix sizes
def getR(z):
    return np.matrix(np.ones(z.shape))

def getH(x, z):
    a = np.zeros((z.shape[0], 2))
    b = np.identity(z.shape[0])
    ret = np.concatenate((a,b), axis=1)

    return np.matrix(ret)

def getG(x, u):
    ret = [[1,0],
           [0,1]]

    for i in range(x.shape[0] - 2):
        ret.append([0, 0])

    return np.matrix(ret)

def getF(x):
    return np.matrix(np.identity((x.shape[0])))

if __name__ == "__main__":
    width = 400
    height = 400
    win = g.GraphWin('SLAM Simulator', width, height)

    sim = Simulator(win, g.Point(200, 200), width, height)
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

    sim.set_landmarks(lm)

    x = np.matrix(np.zeros((len(lm) * 2 + 2,1)))
    P = np.identity(x.shape[0]) * 0.1
    P[0][0] = 100
    P[1][1] = 100
    P = np.matrix(P)

    u = np.zeros((2,1))
    u.fill(1)

    while(1):
        # Close the application if someone closes the graphics window
        if win.isClosed():
            break

        sim.do_motors(u)
        z = sim.sense()

        print(z)

        # Get matrices for this iteration
        F = getF(x)
        G = getG(x, u)
        (x, P) = kalman_predict(F, G, P, x, u)

        H = getH(x, z)
        R = getR(z)
        (x, P) = kalman_update(H, R, P, x, z)

        sleep(0.5)
