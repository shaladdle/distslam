import graphics as g
import numpy as np
from math import acos, cos, sin, pi, sqrt

def matdot(m1, m2):
    return np.dot([x for [x] in np.array(m1)]
                 ,[x for [x] in np.array(m2)]
                 )

def rotate_vector(center, vector, theta):
    rotmat = np.matrix([[cos(theta), -sin(theta)]
                       ,[sin(theta),  cos(theta)]
                       ])

    centvec = np.matrix([[center.x]
                        ,[center.y]
                        ])

    vector -= centvec

    rotvec = np.array(rotmat * vector)

    vector -= centvec

def rotate_pt(center, pt, theta):
    rotmat = np.matrix([[cos(theta), -sin(theta)]
                       ,[sin(theta),  cos(theta)]
                       ])

    centvec = np.matrix([[center.x]
                        ,[center.y]
                        ])

    vector = np.matrix([[pt.x]
                       ,[pt.y]
                       ])

    vector -= centvec

    rotvec = np.array(rotmat * vector)

    rotvec += centvec

    pt.x = rotvec[0][0]
    pt.y = rotvec[1][0]

class FourRectangle:
    def __init__(self, points):
        tl, bl, br, tr = points;
        self.lines = [ g.Line(tl,bl)
                     , g.Line(bl,br)
                     , g.Line(br,tr)
                     , g.Line(tr,tl)
                     ]
        self.front = g.Circle(g.Point((tr.x + br.x) / 2.0, (tr.y + br.y) / 2.0), 3.0)

    def draw(self, win):
        for l in self.lines:
            l.draw(win)
        self.front.color = "red"
        self.front.draw(win)

    def undraw(self):
        for l in self.lines:
            l.undraw()
        self.front.undraw()

def addP(a, b):
    return g.Point(a.x + b.x, a.y + b.y)

class Landmark:
    def __init__(self, win, ident, rad, center):
        self.ident = ident
        self.center = center
        self.rad = rad

        self.win = win
        self.circ = g.Circle(center, rad)
        self.circ.draw(win)

class Simulator:
    def __init__(self, win, start_pt, start_hdg, width, height):
        self.motion_noise = 0.2
        self.sense_noise = 0.01

        self.sense_max = 100
        self.sense_fov = pi
        self.fov_markers = []

        # Start the robot at some provided position, and start it with
        # a heading of 0 (should be pointing east)
        self.robot_pos = start_pt
        self.robot_hdg = start_hdg

        self.width = width
        self.height = height

        # Hang on to the window handle
        self.win = win

        # Create robot
        self.topc = g.Point(self.robot_pos.x - 10, self.robot_pos.y + 15)
        self.botc = g.Point(self.robot_pos.x + 10, self.robot_pos.y - 15)
        self.robotrect = g.Rectangle(self.topc, self.botc)
        self.robotrect.draw(self.win)

    def set_landmarks(self, landmarks):
        self.landmarks = landmarks

    def do_motors(self, u):
        u = np.array(u)
        noise = np.random.randn(3, 1)
        noise[2][0] /= 10
        if (u[0][0] == u[1][0] == 0):
            noise[0][0] = 0
            noise[1][0] = 0
        if (u[2][0] == 0):
            noise[2][0] = 0
            
        u += self.motion_noise * noise

        # math coordinates
        disp = np.linalg.norm(u[:2])
        math_hdg = -self.robot_hdg + u[2][0]
        math_dx = disp * cos(math_hdg)
        math_dy = disp * sin(math_hdg)
        math_x = self.robot_pos.x + math_dx
        math_y = self.height - self.robot_pos.y + math_dy

        rotmat = np.matrix([[cos(math_hdg), -sin(math_hdg)]
                           ,[sin(math_hdg),  cos(math_hdg)]
                           ])

        # the position of the robot in math frame
        math_pos = np.matrix([[math_x]
                             ,[math_y]])

        # compute the box points unrotated
        math_topc = np.matrix([[math_x - 10], [math_y + 15]])
        math_botc = np.matrix([[math_x + 10], [math_y - 15]])


        corners = [ np.matrix([[math_x - 10], [math_y + 15]])
                  , np.matrix([[math_x - 10], [math_y - 15]])
                  , np.matrix([[math_x + 10], [math_y - 15]])
                  , np.matrix([[math_x + 10], [math_y + 15]])
                  ]

        corners_local_frame =    (corner - math_pos for corner in corners)
        corners_local_rot =      (rotmat * corner   for corner in corners_local_frame)
        corners_rot_glob_frame = (corner + math_pos for corner in corners_local_rot)
        corners_arr_rot_glob   = (np.array(corner)  for corner in corners_rot_glob_frame)
        cpoints =                (g.Point(x, self.height - y) for [x], [y] in corners_arr_rot_glob)

        # redraw the box with the rotated box
        self.robotrect.undraw()
        self.robotrect = FourRectangle(cpoints);
        self.robotrect.draw(self.win)

        # update the robot position
        self.robot_pos = g.Point(math_x, self.height - math_y)
        self.robot_hdg = -math_hdg % (2 * pi)

    def sense(self):
        ret = {}
        rob_hdg_vec = np.matrix([[cos(self.robot_hdg)]
                                ,[sin(self.robot_hdg)]
                                ])

        rmat = np.matrix([[self.robot_pos.x]
                         ,[self.robot_pos.y]
                         ])

        print("rmat")
        print(rmat)

        for l in self.landmarks:
            larr = np.array([[l.center.x - self.robot_pos.x]
                            ,[l.center.y - self.robot_pos.y]
                            ])

            lmat = np.matrix(larr)

            loclmvec = np.matrix(lmat - rmat)

            if np.linalg.norm(loclmvec) < self.sense_max:
                print("vector\n" + str(loclmvec) + "\npasses test (norm = " + str(np.linalg.norm(loclmvec)) + ")")
                # first check angle
                dotprod = matdot(loclmvec, rob_hdg_vec)
                A = np.linalg.norm(loclmvec)
                B = np.linalg.norm(rob_hdg_vec)
                angle = abs(acos(dotprod / (A * B)))
                if angle < self.sense_fov / 2:
                    noise = np.random.randn(2)

                    # TODO also need to check for occlusion

                    ret[l.ident] = (l.center.x + noise[0], self.height - l.center.y + noise[1])
                    print("vector\n" + str(loclmvec) + "\npasses test (angle = " + str(angle) + ")")
                else:
                    print("vector\n" + str(loclmvec) + "\ndoes not pass test (angle = " + str(angle) + ")")
            else:
                print("vector\n" + str(loclmvec) + "\ndoes not pass test (norm = " + str(np.linalg.norm(loclmvec)) + ")")

        return ret

    def get_true_state(self):
        ret = []

        ret.append([self.robot_pos.x])
        ret.append([self.robot_pos.y])

        for l in self.landmarks:
            ret.append([l.center.x])
            ret.append([l.center.y])

        return np.matrix(ret)
