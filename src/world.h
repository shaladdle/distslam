#ifndef __WORLD_H__
#define __WORLD_H__

#include <ostream>
#include <vector>

class Point {
    private:
        friend std::ostream & operator << (std::ostream & out, const Point &p) {
            return out << "(" << p.x << ", " << p.y << ")";
        }
    public:
        float x, y;
        Point operator+( const Point& other ) {
            Point ret;
            ret.x = x + other.x;
            ret.y = y + other.y;
            return ret;
        }
        Point operator-( const Point& other ) {
            Point ret;
            ret.x = x - other.x;
            ret.y = y - other.y;
            return ret;
        }
};

class Landmark {
    public:
        Point location;
        int id;
};

class Pose {
    private:
        friend std::ostream & operator << (std::ostream & out, const Pose &p) {
            return out << "(" << p.position.x << 
                          ", " << p.position.y << 
                          ", " << p.orientation << 
                          ")";
        }
    public:
        Point position;
        float orientation;
};

class Odometry {
    private:
        friend std::ostream & operator << (std::ostream & out, const Odometry &p) {
            return out << "(" << p.translation.x << 
                          ", " << p.translation.y << 
                          ", " << p.rotation << 
                          ")";
        }
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
