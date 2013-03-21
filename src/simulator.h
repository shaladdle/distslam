#ifndef __SIMULATOR_H__
#define __SIMULATOR_H__

#include <vector>

#include "./world.h"

class Simulator : public World {
    public:
        void ExecMotorCmd(float lspeed, float rspeed, float duration);
        std::vector<Point> SenseLandmarks();
};

#endif
