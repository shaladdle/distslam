import graphics as g
import numpy as np
import intersect
import vector
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
    def __init__(self, points, hdg, fov, sense_max):
        tl, bl, br, tr = points;
        self.lines = [ g.Line(tl,bl)
                     , g.Line(bl,br)
                     , g.Line(br,tr)
                     , g.Line(tr,tl)
                     ]
        self.eye = g.Point((tr.x + br.x) / 2.0, (tr.y + br.y) / 2.0)
        self.front = g.Circle(self.eye, 3.0)
        self.hdg = hdg
        self.fov = fov
        self.sight_range = sense_max
        self.sight_lines = []
        num_lines = 12
        self.drawn_lines = []

        # draw sight lines
        for i in range(num_lines):
            theta = (hdg - self.fov / 2.0) + (i * (self.fov / (num_lines - 1)))
            self.sight_lines.append(
                g.Line(self.eye,
                    g.Point(self.sight_range * cos(theta) + self.eye.x,
                        self.sight_range * sin(theta) + self.eye.y)))

    def draw(self, win, landmarks):
        for l in self.lines:
            l.draw(win)
        self.front.color = "red"
        self.front.draw(win)
        for l in self.sight_lines:
            # ray cast against landmarks
            min_t = 200
            dx = l.p2.x - l.p1.x
            dy = l.p2.y - l.p1.y
            ray = vector.Vector2(dx, dy)
            for c in landmarks:
                t = intersect.line_circle(l.p1, ray, c.center, c.rad)
                if t >= 0 and t <= 1 and t < min_t:
                    min_t = t
            #make line with new endpoint and draw it
            if min_t >= 0 and min_t <= 1:
                l = g.Line(l.p1, g.Point(l.p1.x + min_t * dx, l.p1.y + min_t * dy))
            l.draw(win)
            self.drawn_lines.append(l)

    def undraw(self):
        for l in self.lines:
            l.undraw()
        self.front.undraw()
        for l in self.drawn_lines:
            l.undraw()
        self.drawn_lines[:] = []

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
        self.sense_fov = 2 * pi / 3
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

    def do_motors(self, u, reverse):
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
        if reverse:
            disp = -disp
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

        corners_local_frame    = (corner - math_pos for corner in corners)
        corners_local_rot      = (rotmat * corner   for corner in corners_local_frame)
        corners_rot_glob_frame = (corner + math_pos for corner in corners_local_rot)
        corners_arr_rot_glob   = (np.array(corner)  for corner in corners_rot_glob_frame)
        cpoints                = (g.Point(x, self.height - y) for [x], [y] in corners_arr_rot_glob)

        # redraw the box with the rotated box
        self.robotrect.undraw()
        self.robotrect = FourRectangle(cpoints, self.robot_hdg, self.sense_fov, self.sense_max);
        self.robotrect.draw(self.win, self.landmarks)

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

        for l in self.landmarks:
            lglbmat = np.matrix([[l.center.x]
                                ,[l.center.y]
                                ])

            loclmvec = lglbmat - rmat 

            if np.linalg.norm(loclmvec) < self.sense_max:
                # first check angle
                dotprod = matdot(loclmvec, rob_hdg_vec)
                A = np.linalg.norm(loclmvec)
                B = np.linalg.norm(rob_hdg_vec)
                angle = abs(acos(dotprod / (A * B)))
                if angle < self.sense_fov / 2:
                    # add gaussian noise
                    loclmvec += self.sense_noise *  np.random.randn(2, 1)

                    # TODO also need to check for occlusion
                    ret[l.ident] = (loclmvec[0,0], -loclmvec[1,0])

        return ret

    def get_true_state(self):
        ret = []

        ret.append([self.robot_pos.x])
        ret.append([self.robot_pos.y])

        for l in self.landmarks:
            ret.append([l.center.x])
            ret.append([l.center.y])

        return np.matrix(ret)
