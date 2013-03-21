#include <iostream>
#include <vector>

#include "./world.h"
#include "./simulator.h"

void Simulator::ExecMotorCmd(float lspeed, float rspeed, float duration) {
    std::cout << "Inside simulator ExecMotorCmd" << std::endl;
}

std::vector<Point> Simulator::SenseLandmarks() {
    std::cout << "Inside simulator ExecMotorCmd" << std::endl;

    return std::vector<Point>();
}
