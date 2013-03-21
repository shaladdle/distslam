#ifndef __WORLD_H__
#define __WORLD_H__

#include <vector>

class Point {
    public:
        float x, y;
};

class Odometry {
    public:
        Point translation;
        float rotation;
};

class World {
    public:
        virtual void ExecMotorCmd(float lspeed, float rspeed, float duration) = 0;
        virtual std::vector<Point> SenseLandmarks() = 0;
};

#endif
