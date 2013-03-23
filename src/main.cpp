#include <iostream>
#include <cmath>

#include "./stdafx.h"
#include "./simulator.h"
#include "./agent.h"

int main(int argc, char *argv[]) {
    Simparams params;
    params.sensor_max_dist = 10;
    params.sensor_angle_range = 0.75 * M_PI;
    params.width = 100;
    params.height = 100;
    params.num_landmarks = 10;

    Simulator sim(params);
    Agent agent((World*)&sim);

    // Agent will just loop, doing whatever logic we tell it to 
    agent.DoLoop();

    return 0;
}
