#include <iostream>

#include "./simulator.h"
#include "./agent.h"

int main(int argc, char *argv[]) {
    Simulator sim;
    Agent agent((World*)&sim);

    // Agent will just loop, doing whatever logic we tell it to 
    agent.DoLoop();

    return 0;
}
