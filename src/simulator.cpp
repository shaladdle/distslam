#include <iostream>
#include <vector>

#include "./world.h"
#include "./simulator.h"

void Simulator::ExecMotorCmd(float lspeed, float rspeed, float duration) {
    std::cout << "Inside simulator ExecMotorCmd" << std::endl;
}

Odometry Simulator::ReadOdometry() {
    std::cout << "Inside simulator ReadOdometry" << std::endl;

    return Odometry();
}

std::vector<Landmark> Simulator::SenseLandmarks() {
    std::cout << "Inside simulator ExecMotorCmd" << std::endl;

    return std::vector<Landmark>();
}
