from typing import List, Iterable

import numpy
import pygame

from pygame import Vector3, Vector2
import math


def ortho_matrix(left, right, bottom, top, near_val, far_val):
    res = numpy.identity(4, dtype=numpy.float32)
    res.itemset((0, 0), float(2 / (right - left)))
    res.itemset((1, 1), float(2 / (top - bottom)))
    res.itemset((2, 2), float(-2 / (far_val - near_val)))

    t_x = -(right + left) / (right - left)
    t_y = -(top + bottom) / (top - bottom)
    t_z = -(far_val + near_val) / (far_val - near_val)
    res.itemset((0, 3), float(t_x))
    res.itemset((1, 3), float(t_y))
    res.itemset((2, 3), float(t_z))

    return res


def perspective_matrix(fovy, aspect, z_near, z_far):
    f = 1 / math.tan(fovy / 2)
    res = numpy.identity(4, dtype=numpy.float32)
    res.itemset((0, 0), f / aspect)
    res.itemset((1, 1), f)
    res.itemset((2, 2), (z_far + z_near) / (z_near - z_far))
    res.itemset((3, 2), (2 * z_far * z_near) / (z_near - z_far))
    res.itemset((2, 3), -1)
    res.itemset((3, 3), 0)
    return res


def get_matrix_looking_at(eye_xyz, target_xyz, up_vec):
    n = eye_xyz - target_xyz
    n.scale_to_length(1)
    u = up_vec.cross(n)
    v = n.cross(u)
    res = numpy.array([[u[0], u[1], u[2], (-u).dot(eye_xyz)],
                       [v[0], v[1], v[2], (-v).dot(eye_xyz)],
                       [n[0], n[1], n[2], (-n).dot(eye_xyz)],
                       [0, 0, 0, 1]], dtype=numpy.float32)
    return res


class Camera3D:

    def __init__(self):
        self.position = Vector3(0, 0, 0)
        self.direction: Vector3 = Vector3(0, 0, 1)
        self.up: Vector3 = Vector3(0, -1, 0)
        self.fov_degrees: float = 45  # vertical field of view

    def __repr__(self):
        return "{}(pos={}, dir={})".format(type(self).__name__, self.position, self.direction)

    def get_xform(self, surface_size):
        view_mat = get_matrix_looking_at(self.position, self.position + self.direction, self.up)
        proj_mat = perspective_matrix(self.fov_degrees / 180 * math.pi, surface_size[0] / surface_size[1], 0.5, 100000)
        return proj_mat @ view_mat

    def project_points_to_surface(self, screen_dims, points) -> List[Vector2]:
        camera_xform = self.get_xform(screen_dims)
        n = len(points)

        # coalesce all the points into a single numpy array
        point_list = numpy.ndarray((n, 4), dtype=numpy.float32)
        for i in range(n):
            pt = points[i]
            point_list[i] = (pt[0], pt[1], pt[2], 1)

        # transform the points through the camera's view matrix
        point_list = point_list.transpose()
        point_list = camera_xform.dot(point_list)
        point_list = point_list.transpose()

        res = []
        for i in range(n):
            w = point_list[i][3]

            if w > 0.001:
                x = screen_dims[0] * (0.5 + point_list[i][0] / w)
                y = screen_dims[1] * (0.5 + point_list[i][1] / w)
                res.append(Vector2(x, y))
            else:
                # means the point is behind the camera, and shouldn't be drawn
                res.append(None)
        return res

    def draw_line_3d(self, screen, p1: Vector3, p2: Vector3, color=(255, 255, 255), width=1):
        xformed_pts = self.project_points_to_surface(screen.get_size(), [p1, p2])
        if xformed_pts[0] is not None and xformed_pts[1] is not None:
            pygame.draw.line(screen, color, xformed_pts[0], xformed_pts[1], width=width)

    def draw_lines_3d(self, screen, lines):
        """
        lines: list of tuples (p1, p2, color, width)
        """
        all_pts = []
        for l in lines:
            all_pts.append(l[0])
            all_pts.append(l[1])
        all_xformed_pts = self.project_points_to_surface(screen.get_size(), all_pts)
        for i in range(0, len(all_xformed_pts) // 2):
            p1 = all_xformed_pts[i * 2]
            p2 = all_xformed_pts[i * 2 + 1]
            color = lines[i][2]
            width = lines[i][3]
            if p1 is not None and p2 is not None:
                pygame.draw.line(screen, color, p1, p2, width)
                

def gen_cube(angle, size, center, color):
    res = []
    pts = []
    for x in (-1, 1):
        for z in (-1, 1):
            xz = Vector2(x, z)
            xz = xz.rotate(angle)
            for y in (-1, 1):
                pts.append(Vector3(xz[0], y, xz[1]) * (size / 2) + center)

                pt = pts[-1]
                for n in pts[:len(pts)-1]:
                    if abs((pt - n).length() - size) <= 0.1:
                        res.append((pt, n, color, 1))
    return res


if __name__ == "__main__":
    # call it to see demo
    import sys

    pygame.init()

    screen = pygame.display.set_mode((600, 300), pygame.RESIZABLE)

    clock = pygame.time.Clock()

    camera = Camera3D()
    camera.position = Vector3(0, 10, -50)

    angle = 0
    lines = []

    import random

    cubes = []
    for _ in range(0, 10):
        angle = random.random() * 360
        speed = random.random() * 1
        size = 10 + random.random() * 30
        x = -100 + random.random() * 200
        z = 100 + random.random() * 40
        y = size / 2
        color = [random.randint(0, 255) for _ in range(3)]
        cubes.append([angle, speed, size, Vector3(x, y, z), color])

    while True:
        events = pygame.event.get()
        for e in events:
            if e.type == pygame.QUIT:
                sys.exit(0)
            if e.type == pygame.KEYDOWN:
                if e.key == pygame.K_ESCAPE:
                    sys.exit(0)
                elif e.key == pygame.K_i:
                    print("camera = " + str(camera))

        keys_held = pygame.key.get_pressed()
        if keys_held[pygame.K_LEFT] or keys_held[pygame.K_RIGHT]:
            xz = Vector2(camera.direction.x, camera.direction.z)
            xz = xz.rotate(1 * (1 if keys_held[pygame.K_LEFT] else -1))
            camera.direction.x = xz[0]
            camera.direction.z = xz[1]
            camera.direction.scale_to_length(1)

        if keys_held[pygame.K_UP] or keys_held[pygame.K_DOWN]:
            camera.direction.y += 0.01 * (1 if keys_held[pygame.K_UP] else -1)
            camera.direction.scale_to_length(1)

        ms = 1
        xz = Vector2(camera.position.x, camera.position.z)
        view_xz = Vector2(camera.direction.x, camera.direction.z)
        view_xz.scale_to_length(1)

        if keys_held[pygame.K_a]:
            xz = xz + ms * view_xz.rotate(90)
        if keys_held[pygame.K_d]:
            xz = xz + ms * view_xz.rotate(-90)
        if keys_held[pygame.K_w]:
            xz = xz + ms * view_xz
        if keys_held[pygame.K_s]:
            xz = xz + ms * view_xz.rotate(180)
        camera.position.x = xz[0]
        camera.position.z = xz[1]

        screen.fill((0, 0, 0))

        lines = []
        for c in cubes:
            c[0] += c[1]  # rotate
            lines.extend(gen_cube(c[0], c[2], c[3], c[4]))

        camera.draw_lines_3d(screen, lines)

        pygame.display.update()
        pygame.display.set_caption(str(int(clock.get_fps())))
        clock.tick(60)
