import numpy as np
from time import sleep
from simulator import *
from kalman import *
import graphics as g

if __name__ == "__main__":
    width = 400
    height = 400
    win = g.GraphWin('SLAM Simulator', width, height)

    sim = Simulator(win, g.Point(200, 200), width, height)
    x0 = np.zeros((3,1))
    kalman = Kalman(x0)

    u = np.zeros((2,1))
    u.fill(1)

    while(1):
        sim.do_motors(u)
        z = sim.sense()

        print(z)

        sleep(0.5)

"""
        kalman.predict(u)
        kalman.update(z)
        
        print kalman.get_estimate()
        print kalman.get_covariance()
"""
