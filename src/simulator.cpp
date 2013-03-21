#include <iostream>
#include <vector>
#include <cstdlib>

#include "./world.h"
#include "./simulator.h"

Simulator::Simulator(int num_landmarks, float _width, float _height) : arena_width(_width), arena_height(_height) {
    last_lm_id = 0;

    for (int i = 0; i < num_landmarks; i++) {
        landmarks.push_back(randLandmark());
    }

    robot.position.x = 0;
    robot.position.y = 0;
    robot.orientation = 0;
}

Simulator::Simulator() : arena_width(0), arena_height(0) {
}

Simulator::~Simulator() {
}

Landmark Simulator::randLandmark() {
    Landmark ret;

    ret.location.x = (float)(rand() % ((int)arena_width * 100)) / 100.0;
    ret.location.y = (float)(rand() % ((int)arena_height * 100)) / 100.0;;
    ret.id = last_lm_id;
    last_lm_id++;

    return ret;
}

void Simulator::ExecMotorCmd(float lspeed, float rspeed, float duration) {
    std::cout << "Inside simulator ExecMotorCmd" << std::endl;
}

Odometry Simulator::ReadOdometry() {
    std::cout << "Inside simulator ReadOdometry" << std::endl;

    return Odometry();
}

std::vector<Landmark> Simulator::SenseLandmarks() {
    std::cout << "Inside simulator SenseLandmarks" << std::endl;

    for (int i = 0; i < landmarks.size(); i++) {
        std::cout << "LM" << landmarks[i].id << " = "
                  << "(" << landmarks[i].location.x << ", "
                  << landmarks[i].location.y << ")" << std::endl;
    }

    // For now we will just return a noiseless measurement
    return landmarks;
}
