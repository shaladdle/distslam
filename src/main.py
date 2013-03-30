import numpy as np
from time import sleep
from simulator import *
from kalman import *

if __name__ == "__main__":
    sim = Simulator()
    kalman = Kalman()

    while(1):
        sim.do_motors()
        z = sim.sense()
        u = sim.read_odometry()

        kalman.predict(u)
        kalman.update(z)
        
        print kalman.get_estimate()
        print kalman.get_covariance()

        sleep(0.5)
