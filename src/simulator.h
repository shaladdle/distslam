#ifndef __SIMULATOR_H__
#define __SIMULATOR_H__

#include <vector>

#include "./world.h"

class Simulator : public World {
    private:
        std::vector<Landmark> landmarks;
        Pose robot;
        const float arena_width, arena_height;
        int last_lm_id;

        Landmark randLandmark();
    public:
        Simulator(int num_landmarks, float width, float height);
        Simulator();
        ~Simulator();
        void ExecMotorCmd(float lspeed, float rspeed, float duration);
        Odometry ReadOdometry();
        std::vector<Landmark> SenseLandmarks();
};

#endif
