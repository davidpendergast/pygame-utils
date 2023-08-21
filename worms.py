import pygame

import math
import random

# "Infinite Worms" Screensaver Project by Ghast

WINDOW_TITLE = "Infinite Worms"
N_AGENTS = 48
GRID_DIMS = 64, 32
UPSCALE = 8
SCREEN_SIZE = 640, 320

WORM_SPEED = 30  # moves per sec
COLOR_OFFSET_VARIATION = 3  # secs
FPS = 60

SHRINK_RATE = 4
FADE_RATE = 12
BLUR_RADIUS = 1

HUE_SHIFT_RATE = 12  # degrees per sec

SATURATION_RANGE = (0.666, 1.0)
SATURATION_SHIFT_RATE = 60  # seconds per cycle

VALUE_SHIFT_RANGE = (0.666, 1.0)
VALUE_SHIFT_RATE = 16  # seconds per cycle


class Grid:

    def __init__(self, dims, agents):
        self.t = 0
        self.ticks = 0
        self.dims = dims
        self.surf = pygame.Surface(dims).convert_alpha()

        self.pheromones = pygame.Surface(dims)
        self.phero_fade = pygame.Surface(dims)
        self.phero_fade.fill("black")
        self.phero_fade.set_alpha(1)

        self.upscale = pygame.Surface((dims[0] * UPSCALE, dims[1] * UPSCALE))

        self.fade = pygame.Surface((dims[0] * UPSCALE, dims[1] * UPSCALE))
        self.fade.fill("black")
        self.fade.set_alpha(1)

        self.agents = agents

    def update(self, dt):
        self.surf.fill((0, 0, 0, 0))
        for a in self.agents:
            a.update(self, dt)
        self.pheromones.blit(self.phero_fade, (0, 0))

        if self.ticks % SHRINK_RATE == 0:
            shrink = pygame.transform.smoothscale(self.upscale, (self.upscale.get_width() - 4, self.upscale.get_height() - 2))
            self.upscale.fill("black")
            self.upscale.blit(shrink, (2, 1))

        if self.ticks % FADE_RATE == 0:
            self.upscale.blit(self.fade, (0, 0))
            self.upscale = pygame.transform.gaussian_blur(self.upscale, BLUR_RADIUS)

        ups = pygame.transform.scale(self.surf, self.upscale.get_size())
        self.upscale.blit(ups, (0, 0))

        self.t += dt
        self.ticks += 1

    def render(self, screen: pygame.Surface, mode='normal'):
        if mode == 'normal':
            screen.blit(pygame.transform.smoothscale(self.upscale, screen.get_size()), (0, 0))
        elif mode == 'pheromone':
            screen.blit(pygame.transform.scale(self.pheromones, screen.get_size()), (0, 0))
        else:
            screen.fill("black")


class Agent:

    def __init__(self, xy, period=1 / WORM_SPEED, offs=0.0):
        self.color = pygame.Color("red")
        self.hsva = self.color.hsva
        self.xy = xy
        self.period = period
        self.cooldown = period
        self.t = 0
        self.offs = offs

    def update(self, grid, dt):
        self.cooldown -= dt
        if self.cooldown <= 0:
            self.cooldown = self.period
            self.xy = self.calc_next_pos(grid)
            grid.pheromones.set_at(self.xy, self.color)
        grid.surf.set_at(self.xy, self.color)

        self.t += dt
        t = self.t + self.offs

        self.hsva = ((t * HUE_SHIFT_RATE) % 360,
                     SATURATION_RANGE[0] + (SATURATION_RANGE[1] - SATURATION_RANGE[0]) * (0.5 + math.cos(2 * math.pi * t / SATURATION_SHIFT_RATE) / 2),
                     VALUE_SHIFT_RANGE[0] + (VALUE_SHIFT_RANGE[1] - VALUE_SHIFT_RANGE[0]) * (0.5 + math.cos(2 * math.pi * t / VALUE_SHIFT_RATE) / 2),
                     1)
        self.color.hsva = int(self.hsva[0]), int(100 * self.hsva[1]), int(100 * self.hsva[2]), int(100 * self.hsva[3])

    def _evaluate(self, xy, grid):
        if xy[0] < 0 or xy[1] < 0 or xy[0] >= grid.dims[0] or xy[1] >= grid.dims[1]:
            return 1000  # don't wrap
        else:
            color = pygame.Color(grid.pheromones.get_at(xy))
            return color.hsva[2]

    def calc_next_pos(self, grid):
        neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        random.shuffle(neighbors)
        xys = [(self.xy[0] + n[0], self.xy[1] + n[1]) for n in neighbors]
        xys.sort(key=lambda n: self._evaluate(n, grid))
        return xys[0]


def main():
    pygame.init()

    screen = pygame.display.set_mode(SCREEN_SIZE, pygame.RESIZABLE)
    pygame.display.set_caption(f"{WINDOW_TITLE}")
    clock = pygame.time.Clock()
    dt = 0

    agents = [Agent((random.randint(0, GRID_DIMS[0] - 1), random.randint(0, GRID_DIMS[1] - 1)),
                    offs=i / N_AGENTS * COLOR_OFFSET_VARIATION) for i in range(N_AGENTS)]
    grid = Grid(GRID_DIMS, agents)

    running = True
    while running:
        for e in pygame.event.get():
            if e.type == pygame.QUIT or (e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE):
                running = False
            if e.type == pygame.KEYDOWN:
                if e.key == pygame.K_F4:
                    pygame.display.toggle_fullscreen()

        keys = pygame.key.get_pressed()
        mode = 'normal' if not keys[pygame.K_SPACE] else 'pheromone'

        grid.update(dt)

        screen.fill("black")
        grid.render(screen, mode=mode)

        pygame.display.flip()
        dt = clock.tick(FPS) / 1000

        if grid.ticks % 15 == 14:
            fps = clock.get_fps()
            pygame.display.set_caption(f"{WINDOW_TITLE} [FPS={fps:.1f}]")

main()

