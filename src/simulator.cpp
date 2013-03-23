#include <iostream>
#include <vector>
#include <cstdlib>
#include <cmath>

#include "./stdafx.h"
#include "./world.h"
#include "./simulator.h"

Simulator::Simulator(Simparams params) {
    arena_width = params.width;
    arena_height = params.height;

    last_lm_id = 0;

    for (int i = 0; i < params.num_landmarks; i++) {
        landmarks.push_back(randLandmark());
    }

    robot.position.x = 0;
    robot.position.y = 0;
    robot.orientation = 0;
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
    float dist = duration * std::min(lspeed, rspeed);

    lastpose = robot;

    robot.position.x += dist * sin(robot.orientation);
    robot.position.y += dist * cos(robot.orientation);

    std::cout << "Old pose was " << lastpose << ", new pose is " << robot << std::endl;
}

Odometry Simulator::ReadOdometry() {
    Odometry ret;

    ret.translation = robot.position - lastpose.position;
    ret.rotation = 0;

    std::cout << "ReadOdometry() = " << ret << std::endl;

    return ret;
}

std::vector<Landmark> Simulator::SenseLandmarks() {
    std::cout << "Inside simulator SenseLandmarks" << std::endl;

    // We really want to do some kind of bounding box based on sensor
    // properties like max range

    for (int i = 0; i < landmarks.size(); i++) {
        std::cout << "LM" << landmarks[i].id << " = "
                  << "(" << landmarks[i].location.x << ", "
                  << landmarks[i].location.y << ")" << std::endl;
    }

    // For now we will just return a noiseless measurement of all landmarks
    return landmarks;
}
