#ifndef __WORLD_H__
#define __WORLD_H__

#include <vector>

class Point {
    public:
        float x, y;
};

class Landmark {
    public:
        Point location;
        int id;
};

class Pose {
    public:
        Point position;
        float orientation;
};

class Odometry {
    public:
        Point translation;
        float rotation;
};

class World {
    public:
        virtual void ExecMotorCmd(float lspeed, float rspeed, float duration) = 0;
        virtual Odometry ReadOdometry() = 0;
        virtual std::vector<Landmark> SenseLandmarks() = 0;
};

#endif
