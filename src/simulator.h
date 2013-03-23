#ifndef __SIMULATOR_H__
#define __SIMULATOR_H__

#include <vector>

#include "./world.h"

class Simparams {
    public:
        float sensor_max_dist, sensor_angle_range;
        float num_landmarks;
        float width, height;
};

class Simulator : public World {
    private:
        // Internal state
        std::vector<Landmark> landmarks;
        Pose robot, lastpose;
        float arena_width, arena_height;
        int last_lm_id;

        // Helper functions
        Landmark randLandmark();
    public:
        Simulator(Simparams params);
        ~Simulator();

        // World API
        void ExecMotorCmd(float lspeed, float rspeed, float duration);
        Odometry ReadOdometry();
        std::vector<Landmark> SenseLandmarks();
};

#endif
