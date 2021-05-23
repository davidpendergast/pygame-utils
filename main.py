import numpy
import random
import math
import argparse

import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
import pygame
import pygame.gfxdraw

# Double Pendulum with Pygame + pyOpenGL
# by Ghast ~ https://github.com/davidpendergast

USE_GL = True  # if False, will use pure pygame

if USE_GL:
    import OpenGL as OpenGL
    import OpenGL.GL as GL
    import OpenGL.GLU as GLU

# Params (can be set via the command line)
N = 100000          # Number of pendulums
L = 20              # Length of pendulum arms
M = 5               # Mass of pendulum arms
G = 2 * 9.8         # Gravity
FPS = 60
ZOOM = 5

ML2 = M * L * L

# OpenGL-Only Params
COLOR_CHANNELS = 4  # must be 3 or 4
RAINBOW = True
OPACITY = 0.01


def get_initial_conds():
    theta1 = 3.1415 * (random.random() + 0.5)
    theta2 = random.random() * 6.283
    spread = (6.283 / 360) / 4
    return theta1, theta2, spread


class State:

    def __init__(self):
        theta1, theta2, spread = get_initial_conds()
        self.theta1 = numpy.linspace(theta1, theta1 + spread, N)
        self.theta2 = numpy.linspace(theta2, theta2 + spread, N)
        self.p1 = numpy.zeros(N)
        self.p2 = numpy.zeros(N)

        self.colors = [hsv_to_rgb(360 * i / N, 0.75, 1) for i in range(N)]
        self.colors.reverse()

        # arrays for temp storage (to avoid reallocating arrays constantly)
        self.sub = numpy.zeros(N)
        self.cos = numpy.zeros(N)
        self.cos2 = numpy.zeros(N)
        self.sin = numpy.zeros(N)
        self.temp1 = numpy.zeros(N)
        self.temp2 = numpy.zeros(N)
        self.temp3 = numpy.zeros(N)
        self.temp4 = numpy.zeros(N)
        self.denom = numpy.zeros(N)
        self.num = numpy.zeros(N)

        self.dtheta1 = numpy.zeros(N)
        self.dtheta2 = numpy.zeros(N)
        self.dp1 = numpy.zeros(N)
        self.dp2 = numpy.zeros(N)
        self.x1 = numpy.zeros(N, dtype=numpy.int16)
        self.y1 = numpy.zeros(N, dtype=numpy.int16)
        self.x2 = numpy.zeros(N, dtype=numpy.int16)
        self.y2 = numpy.zeros(N, dtype=numpy.int16)

        if USE_GL:
            self.vertex_data = numpy.zeros((N * 2 + 1) * 2, dtype=float)

            self.color_data = numpy.ones((N * 2 + 1) * COLOR_CHANNELS, dtype=float)
            if RAINBOW:
                for i in range(0, N * 2):
                    c = self.colors[i // 2]
                    self.color_data[i * COLOR_CHANNELS + 0] = c[0] / 256
                    self.color_data[i * COLOR_CHANNELS + 1] = c[1] / 256
                    self.color_data[i * COLOR_CHANNELS + 2] = c[2] / 256
            if COLOR_CHANNELS > 3:
                self.color_data[3::COLOR_CHANNELS] = OPACITY

            self.index_data = numpy.arange(0, N * 4, dtype=int)
            self.index_data[0::4] = N * 2  # center point is stored as the Nth vertex
            self.index_data[1::4] = numpy.arange(0, N * 2, 2, dtype=int)
            self.index_data[2::4] = numpy.arange(0, N * 2, 2, dtype=int)
            self.index_data[3::4] = numpy.arange(1, N * 2, 2, dtype=int)

    def euler_update(self, dt):
        numpy.subtract(self.theta1, self.theta2, out=self.sub)
        numpy.cos(self.sub, out=self.cos)
        numpy.square(self.cos, out=self.cos2)
        numpy.sin(self.sub, out=self.sin)

        self.calc_dtheta1(self.p1, self.p2, self.cos, self.cos2)
        self.calc_dtheta2(self.p1, self.p2, self.cos, self.cos2)
        self.calc_dp1(self.theta1, self.dtheta1, self.dtheta2, self.sin)
        self.calc_dp2(self.theta2, self.dtheta1, self.dtheta2, self.sin)

        self.theta1 = self.theta1 + dt * self.dtheta1
        self.theta2 = self.theta2 + dt * self.dtheta2
        self.p1 = self.p1 + dt * self.dp1
        self.p2 = self.p2 + dt * self.dp2

    def calc_dtheta1(self, p1, p2, cos, cos2):
        # self.dtheta1 = (6 / ML2) * (2 * p1 - 3 * cos * p2) / (16 - 9 * cos2)
        numpy.multiply(2, p1, out=self.temp1)
        numpy.multiply(3, cos, out=self.temp2)
        numpy.multiply(self.temp2, p2, out=self.temp2)
        numpy.subtract(self.temp1, self.temp2, out=self.num)
        numpy.multiply((6 / ML2), self.num, out=self.num)
        numpy.multiply(9, cos2, out=self.temp3)
        numpy.subtract(16, self.temp3, out=self.denom)
        numpy.divide(self.num, self.denom, out=self.dtheta1)

    def calc_dtheta2(self, p1, p2, cos, cos2):
        # self.dtheta2 = (6 / ML2) * (8 * p2 - 3 * cos * p1) / (16 - 9 * cos2)
        numpy.multiply(8, p2, out=self.temp1)
        numpy.multiply(3, cos, out=self.temp2)
        numpy.multiply(self.temp2, p1, out=self.temp2)
        numpy.subtract(self.temp1, self.temp2, out=self.num)
        numpy.multiply((6 / ML2), self.num, out=self.num)
        numpy.multiply(9, cos2, out=self.temp3)
        numpy.subtract(16, self.temp3, out=self.denom)
        numpy.divide(self.num, self.denom, out=self.dtheta2)

    def calc_dp1(self, theta1, dtheta1, dtheta2, sin):
        # self.dp1 = (-ML2 / 2) * (dtheta1 * dtheta2 * sin + 3 * G / L * numpy.sin(theta1))
        numpy.multiply(dtheta1, dtheta2, out=self.temp1)
        numpy.multiply(self.temp1, sin, out=self.temp1)
        numpy.sin(theta1, out=self.temp2)
        numpy.multiply(3 * G / L, self.temp2, out=self.temp2)
        numpy.add(self.temp1, self.temp2, out=self.temp3)
        numpy.multiply(-ML2 / 2, self.temp3, out=self.dp1)

    def calc_dp2(self, theta2, dtheta1, dtheta2, sin):
        # self.dp2 = (-ML2 / 2) * (-dtheta1 * dtheta2 * sin + G / L * numpy.sin(theta2))
        numpy.multiply(dtheta1, dtheta2, out=self.temp1)
        numpy.multiply(self.temp1, sin, out=self.temp1)
        numpy.sin(theta2, out=self.temp2)
        numpy.multiply(G / L, self.temp2, out=self.temp2)
        numpy.subtract(self.temp2, self.temp1, out=self.temp3)
        numpy.multiply(-ML2 / 2, self.temp3, out=self.dp2)

    def render_all(self, screen):
        x0 = screen.get_width() // 2
        y0 = screen.get_height() // 2

        # self.x1 = x0 + ZOOM * L * numpy.cos(self.theta1 + 3.1429 / 2)
        numpy.add(self.theta1, 3.1429 / 2, out=self.temp1)
        numpy.cos(self.temp1, out=self.temp1)
        numpy.multiply(ZOOM * L, self.temp1, out=self.temp1)
        numpy.add(x0, self.temp1, out=self.x1, casting='unsafe')

        # self.y1 = y0 + ZOOM * L * numpy.sin(self.theta1 + 3.1429 / 2)
        numpy.add(self.theta1, 3.1429 / 2, out=self.temp2)
        numpy.sin(self.temp2, out=self.temp2)
        numpy.multiply(ZOOM * L, self.temp2, out=self.temp2)
        numpy.add(y0, self.temp2, out=self.y1, casting='unsafe')

        # self.x2 = self.x1 + ZOOM * L * numpy.cos(self.theta2 + 3.1429 / 2)
        numpy.add(self.theta2, 3.1429 / 2, out=self.temp3)
        numpy.cos(self.temp3, out=self.temp3)
        numpy.multiply(ZOOM * L, self.temp3, out=self.temp3)
        numpy.add(self.x1, self.temp3, out=self.x2, casting='unsafe')

        # self.y2 = self.y1 + ZOOM * L * numpy.sin(self.theta2 + 3.1429 / 2)
        numpy.add(self.theta2, 3.1429 / 2, out=self.temp4)
        numpy.sin(self.temp4, out=self.temp4)
        numpy.multiply(ZOOM * L, self.temp4, out=self.temp4)
        numpy.add(self.y1, self.temp4, out=self.y2, casting='unsafe')

        if USE_GL:
            self.vertex_data[0:N*4:4] = self.x1
            self.vertex_data[1:N*4:4] = self.y1
            self.vertex_data[2:N*4:4] = self.x2
            self.vertex_data[3:N*4:4] = self.y2
            self.vertex_data[N*4] = x0
            self.vertex_data[N*4 + 1] = y0

            GL.glClear(GL.GL_COLOR_BUFFER_BIT)
            GL.glColor(1, 1, 0)

            GL.glEnableClientState(GL.GL_VERTEX_ARRAY)
            GL.glEnableClientState(GL.GL_COLOR_ARRAY)

            GL.glVertexPointer(2, GL.GL_FLOAT, 0, self.vertex_data)
            GL.glColorPointer(COLOR_CHANNELS, GL.GL_FLOAT, 0, self.color_data)
            GL.glDrawElements(GL.GL_LINES, len(self.index_data), GL.GL_UNSIGNED_INT, self.index_data);

            GL.glDisableClientState(GL.GL_VERTEX_ARRAY)
            GL.glDisableClientState(GL.GL_COLOR_ARRAY)

        else:
            screen.fill((0, 0, 0))
            # (-_-) don't think there's a good way to avoid this loop without gl...
            for i in range(0, N):
                pygame.gfxdraw.line(screen, x0, y0, self.x1[i], self.y1[i], self.colors[i])
                pygame.gfxdraw.line(screen, self.x1[i], self.y1[i], self.x2[i], self.y2[i], self.colors[i])


import cProfile
import pstats


class Profiler:

    def __init__(self):
        self.is_running = False
        self.pr = cProfile.Profile(builtins=False)

    def toggle(self):
        self.is_running = not self.is_running

        if not self.is_running:
            self.pr.disable()

            ps = pstats.Stats(self.pr)
            ps.strip_dirs()
            ps.sort_stats('cumulative')
            ps.print_stats(35)

        else:
            print("Started profiling...")
            self.pr.clear()
            self.pr.enable()


def initialize_display(size):
    pygame.init()

    if USE_GL:
        display = size
        flags = pygame.DOUBLEBUF | pygame.OPENGL
        screen = pygame.display.set_mode(display, flags)
        GL.glClearColor(0, 0, 0, 1)
        GL.glViewport(0, 0, display[0], display[1])
        GL.glOrtho(0.0, display[0], display[1], 0.0, 0.0, 1.0);
        if COLOR_CHANNELS > 3:
            GL.glEnable(GL.GL_BLEND)
            GL.glBlendFunc(GL.GL_SRC_ALPHA, GL.GL_ONE_MINUS_SRC_ALPHA);

        return screen
    else:
        return pygame.display.set_mode(size, 0, 8)


def start(size):
    screen = initialize_display(size)
    clock = pygame.time.Clock()

    state = State()
    profiler = Profiler()

    running = True
    ticks = 0
    while running:
        state.euler_update(1 / FPS)

        state.render_all(screen)

        pygame.display.flip()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    print("Reseting...")
                    state = State()
                elif event.key == pygame.K_p:
                    profiler.toggle()

        if ticks % 60 == 10:
            pygame.display.set_caption("Double Pendulum (FPS={:.1f}, N={})".format(clock.get_fps(), N))

        ticks += 1
        clock.tick(FPS)


def hsv_to_rgb(h, s, v):
    """
    :param h: 0 <= h < 360
    :param s: 0 <= s <= 1
    :param v: 0 <= v <= 1
    :return: (r, g, b) as floats
    """
    C = v * s
    X = C * (1 - abs((h / 60) % 2 - 1))
    m = v - C

    if h < 60:
        rgb_prime = (C, X, 0)
    elif h < 120:
        rgb_prime = (X, C, 0)
    elif h < 180:
        rgb_prime = (0, C, X)
    elif h < 240:
        rgb_prime = (0, X, C)
    elif h < 300:
        rgb_prime = (X, 0, C)
    else:
        rgb_prime = (C, 0, X)

    return (int(256 * rgb_prime[0] + m),
            int(256 * rgb_prime[1] + m),
            int(256 * rgb_prime[2] + m))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='A double pendulum simulation made with pygame, PyOpenGL, and numpy.')
    parser.add_argument('-n', type=int, metavar="int", default=1000, help=f'the number of pendulums (default {N})')
    parser.add_argument('--opacity', type=float,  metavar="float", default=0.1, help=f'the opacity of the pendulums (default {OPACITY})')
    parser.add_argument('--length', type=int,  metavar="int", default=L, help=f'the length of the pendulum arms (default {L})')
    parser.add_argument('--mass', type=int, metavar="float", default=M, help=f'the mass of the pendulum arms (default {M})')
    parser.add_argument('--fps', type=int, metavar="int", default=FPS, help=f'the target FPS for the simulation (default {FPS})')
    parser.add_argument('--zoom', type=int, metavar="int", default=ZOOM, help=f'the target FPS for the simulation (default {ZOOM})')
    parser.add_argument('--size', type=int, metavar="int", default=[800, 600], nargs=2, help='the window size for the simulation (default 600 800)')

    args = parser.parse_args()

    N = args.n
    OPACITY = args.opacity
    L = args.length
    M = args.mass
    FPS = args.fps
    ZOOM = args.zoom

    start(args.size)
