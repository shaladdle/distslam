#include <iostream>
#include <unistd.h>

#include "./agent.h"

void Agent::DoLoop() {
    while (true) {
        std::cout << std::endl
                  << "Phase 1: Do planning" << std::endl
                  << "Phase 2: Motor commands" << std::endl;

        world->ExecMotorCmd(0, 0, 0);

        std::cout << "Phase 3: Read odometry" << std::endl;

        world->ReadOdometry();

        std::cout << "Phase 4: Kalman predict" << std::endl
                  << "Phase 5: Read sensors" << std::endl;

        world->SenseLandmarks();

        std::cout << "Phase 6: Kalman update" << std::endl;

        sleep(1);
    }
}
