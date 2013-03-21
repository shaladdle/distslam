#ifndef __AGENT_H__
#define __AGENT_H__

#include "./world.h"

class Agent {
    private:
        World *world;
    public:
        Agent(World *_world) {
            world = _world;
        }
        void DoLoop();
};

#endif
