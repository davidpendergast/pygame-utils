import os
import sys

import argparse
import math
import random
import enum

os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "1"
import pygame
import matplotlib.pyplot as plt


# Pygame Benchmarking Program
# by Ghast ~ github.com/davidpendergast


SCREEN_SIZE = 640, 480

ENT_SIZE = 64      # width & height of filled-in entities in pixels

SPAWN_RATE = 200   # entities per sec, increasing this will make the tests run faster but be less accurate
TICK_RATE = 0.10   # frequency of entity additions and fps measurements
STOP_AT_FPS = 45   # test cases stop when this FPS is reached

INCLUDE_SURFACES = True
INCLUDE_FILLED_SHAPES = True
INCLUDE_HOLLOW_SHAPES = True
INCLUDE_LINES = True

RAND_SEED = 27182818

CAPTION_REFRESH_PER_SEC = 3     # per second, how frequently to update the window title
LOW_FPS_GRACE_PERIOD = 0.5      # seconds, how long the FPS can stay under STOP_AT_FPS before ending the test case
PAUSE_BETWEEN_TESTS_TIME = 1    # seconds, how long to pause between tests
GRAPH_SMOOTHING_RADIUS = 1      # seconds, how much to smooth the graph (uses running average)
LOG_AXIS = True                 # whether to use log-scaling for the graph's y-axis

PX_PER_ENT = ENT_SIZE ** 2


class EntityType(enum.IntEnum):
    SURF_RGB = 0                # Surface with no per-pixel alpha
    SURF_RGB_WITH_ALPHA = 1     # Surface with no per-pixel alpha, but an alpha value < 255
    SURF_RGBA = 2               # Surface with per-pixel alpha.

    LINE = 3                    # A line drawn via pygame.draw.line
    RECT_FILLED = 4             # A rect drawn via pygame.draw.rect (with width = 0)
    RECT_HOLLOW = 5             # A rect drawn via pygame.draw.rect (with width > 0)
    CIRCLE_FILLED = 6           # A circle drawn via pygame.draw.circle (with width = 0)
    CIRCLE_HOLLOW = 7           # A circle drawn via pygame.draw.circle (with width > 0)


def _calc_avg_lengths(w, h, n=10000):
    """Finds the average length of a line between two random points in a (w x h) area using random sampling.
        returns: Average total length, x-length, and y-length
    """
    total_dx = 0
    total_dy = 0
    total_dist = 0

    # Fixed seed because this can affect the test quite a lot. For example, if hollow circles
    # use a thickness of 6 pixels instead of 5 that's a 20% increase.
    rand = random.Random(x=12345)

    for _ in range(n):
        p1 = rand.randint(0, w - 1), rand.randint(0, h - 1)
        p2 = rand.randint(0, w - 1), rand.randint(0, h - 1)
        dx = p1[0] - p2[0]
        dy = p1[1] - p2[1]
        total_dx += abs(dx)
        total_dy += abs(dy)
        total_dist += math.sqrt(dx * dx + dy * dy)
    return total_dist / n, total_dx / n, total_dy / n


AVG_LENGTH, AVG_WIDTH, AVG_HEIGHT = _calc_avg_lengths(*SCREEN_SIZE)

# Want to define the geometric entities such that on average they'll have an area of about PX_PER_ENT pixels.
# Note that these are approximate (particularly the circle one, which assumes the relationship between radius
# and pixels changed is linear (it's actually quadratic)).
LINE_WIDTH = max(1, round(PX_PER_ENT / AVG_LENGTH))
HOLLOW_RECT_WIDTH = max(1, round(PX_PER_ENT / (2 * (AVG_WIDTH + AVG_HEIGHT))))
HOLLOW_CIRCLE_RADIUS_MULT = 0.25  # radius = mult * dist between two random points
HOLLOW_CIRCLE_WIDTH = max(1, round((AVG_LENGTH * HOLLOW_CIRCLE_RADIUS_MULT)
                                   - math.sqrt(max(0, (AVG_LENGTH * HOLLOW_CIRCLE_RADIUS_MULT) ** 2
                                                   - PX_PER_ENT / math.pi))))
FILLED_CIRCLE_RADIUS = max(1, round(math.sqrt(PX_PER_ENT / math.pi)))


def _print_info():
    print(f"\nStarting new simulation:\n"
          f"  Screen size =   {SCREEN_SIZE} px\n"
          f"  Entity size =   {ENT_SIZE} x {ENT_SIZE} px\n"
          f"  Minimum FPS =   {STOP_AT_FPS} fps\n"
          f"  Spawn Rate =    {SPAWN_RATE} ents/sec\n"
          f"  Tick Rate =     {TICK_RATE} sec")

    print(f"\nAverage number of pixels changed per render:")
    print(f"  Filled Rect:    {PX_PER_ENT:.2f} = {ENT_SIZE}**2")
    print(f"  Filled Circle:  {math.pi * FILLED_CIRCLE_RADIUS**2:.2f} = pi * {FILLED_CIRCLE_RADIUS} ** 2")
    print(f"  Line:           {AVG_LENGTH * LINE_WIDTH:.2f} = {AVG_LENGTH:.2f} * {LINE_WIDTH:.2f}")
    print(f"  Hollow Rect:    {2 * (AVG_WIDTH + AVG_HEIGHT) * HOLLOW_RECT_WIDTH:.2f}"
          f" = 2 * ({AVG_WIDTH:.2f} + {AVG_HEIGHT:.2f}) * {HOLLOW_RECT_WIDTH:.2f}")
    avg_hollow_circle_area = math.pi * ((AVG_LENGTH * HOLLOW_CIRCLE_RADIUS_MULT)**2
                                        - (AVG_LENGTH*HOLLOW_CIRCLE_RADIUS_MULT - HOLLOW_CIRCLE_WIDTH)**2)
    print(f"  Hollow Circle:  {avg_hollow_circle_area:.2f}"
          f" = pi * (({AVG_LENGTH:.2f})**2 - ({AVG_LENGTH:.2f}"
          f" - {HOLLOW_CIRCLE_WIDTH:.2f})**2) (approx.)")


class EntityFactory:

    def __init__(self, seed=RAND_SEED):
        self.rand = random.Random(x=seed)

    def get_next(self):
        return (random.randint(0, 4096),  # determines Entity type
                random.randint(0, 4096),  # determines (x1, y1)
                random.randint(0, 4096),  # determines (x2, y2)
                random.randint(0, 4096))  # determines color and/or opacity


class Renderer:

    def __init__(self, screen, n_pts=503, ent_types=tuple(e for e in EntityType), seed=RAND_SEED):
        self.screen = screen
        self.entities = []
        self.ent_types = ent_types
        self.random = random.Random(x=seed)

        # Instantiate the invisible points that float around the screen.
        # These points determine the locations of entities. Note that they're not
        # updated one-by-one each frame -- instead, their positions are derived
        # on-the-fly using their initial position, velocity, and the time.
        w, h = self.screen.get_size()
        self.pts = [pygame.Vector2(self.random.randint(0, w-1),
                                   self.random.randint(0, h-1)) for _ in range(n_pts)]
        self.vels = [pygame.Vector2(0, 1) for _ in range(n_pts)]
        for v in self.vels:
            v.rotate_ip(random.random() * 360.0)
            v.scale_to_length(random.randint(30, 50))

        self.t = 0  # current time
        self.colors = ["red", "green", "blue", "yellow", "cyan", "magenta", "orange"]

        # Precompute the surfaces that entities may need.
        self.rgb_surfs = []
        self.rgba_surfs = []
        self.rgb_surfs_with_alpha = []
        for idx, c in enumerate(self.colors):
            rgb_surf = pygame.Surface((ENT_SIZE, ENT_SIZE))
            rgb_surf.fill(c)
            self.rgb_surfs.append(rgb_surf)

            opacity = (idx + 1) / len(self.colors)

            rgba_surf = pygame.Surface((ENT_SIZE, ENT_SIZE), flags=pygame.SRCALPHA)
            rgb = tuple(i for i in pygame.color.Color(c))
            rgba_surf.fill((*rgb[:3], int(opacity * 255)))
            self.rgba_surfs.append(rgba_surf)

            rgb_surf_with_alpha = pygame.Surface((ENT_SIZE, ENT_SIZE))
            rgb_surf_with_alpha.fill(c)
            rgb_surf_with_alpha.set_alpha(int(opacity * 255))
            self.rgb_surfs_with_alpha.append(rgb_surf_with_alpha)

        # Bit hacky, but it's important to look up each entity's render method quickly.
        self.render_methods = [None] * len(EntityType)
        for e in EntityType:
            self.render_methods[e] = getattr(self, f"render_{e.name}")

    def update(self, dt):
        self.t += dt

    def get_point(self, n1, n2):
        base_pt = self.pts[n1 % len(self.pts)] + self.pts[n2 % len(self.pts)]
        vel = self.vels[n2 % len(self.vels)] + self.vels[n1 % len(self.vels)]
        pt = base_pt + vel * self.t
        x = int(pt.x) % self.screen.get_width()
        y = int(pt.y) % self.screen.get_height()
        return x, y

    def render(self, bg_color=(0, 0, 0)):
        self.screen.fill(bg_color)
        for ent in self.entities:
            ent_type = self.ent_types[ent[0] % len(self.ent_types)]
            p1 = self.get_point(ent[0], ent[1])
            p2 = self.get_point(ent[0], ent[2])
            color_idx = ent[3]

            self.render_methods[ent_type](p1, p2, color_idx)  # do actual rendering

    def render_SURF_RGB(self, p1, p2, color_idx):
        surf = self.rgb_surfs[color_idx % len(self.rgb_surfs)]
        self.screen.blit(surf, (p1[0] - ENT_SIZE // 2, p1[1] - ENT_SIZE // 2))

    def render_SURF_RGBA(self, p1, p2, color_idx):
        surf = self.rgba_surfs[color_idx % len(self.rgba_surfs)]
        self.screen.blit(surf, (p1[0] - ENT_SIZE // 2, p1[1] - ENT_SIZE // 2))

    def render_SURF_RGB_WITH_ALPHA(self, p1, p2, color_idx):
        surf = self.rgb_surfs_with_alpha[color_idx % len(self.rgb_surfs_with_alpha)]
        self.screen.blit(surf, (p1[0] - ENT_SIZE // 2, p1[1] - ENT_SIZE // 2))

    def render_LINE(self, p1, p2, color_idx):
        c = self.colors[color_idx % len(self.colors)]
        pygame.draw.line(self.screen, c, p1, p2, width=LINE_WIDTH)

    def render_RECT_HOLLOW(self, p1, p2, color_idx):
        c = self.colors[color_idx % len(self.colors)]
        pygame.draw.rect(self.screen, c, (p1[0], p1[1], p2[0] - p1[0], p2[1] - p1[1]),
                         width=HOLLOW_RECT_WIDTH)

    def render_RECT_FILLED(self, p1, p2, color_idx):
        c = self.colors[color_idx % len(self.colors)]
        pygame.draw.rect(self.screen, c, (p1[0]  - ENT_SIZE // 2, p1[1] - ENT_SIZE // 2, ENT_SIZE, ENT_SIZE))

    def render_CIRCLE_HOLLOW(self, p1, p2, color_idx):
        c = self.colors[color_idx % len(self.colors)]
        r = (abs(p2[0] - p1[0]) + abs(p2[1] - p1[0])) * HOLLOW_CIRCLE_RADIUS_MULT
        pygame.draw.circle(self.screen, c, p1, r, width=HOLLOW_CIRCLE_WIDTH)

    def render_CIRCLE_FILLED(self, p1, p2, color_idx):
        c = self.colors[color_idx % len(self.colors)]
        pygame.draw.circle(self.screen, c, p1, FILLED_CIRCLE_RADIUS)


def start_plot(title, subtitle=None):
    if subtitle is not None:
        plt.suptitle(title, fontsize=16)
        plt.title(subtitle, fontsize=10, y=1)
    else:
        plt.title(title)
    plt.xlabel('Entities')
    plt.ylabel('FPS')
    if LOG_AXIS:
        plt.yscale('log')
    yticks = [15, 30, 45, 60, 120, 144, 240]
    plt.yticks(yticks, [str(yt) for yt in yticks])


def get_x_and_y(data, smooth_radius=0):
    x = []
    y = []
    for cnt in data:
        x.append(cnt)
        y.append(data[cnt])
    if smooth_radius > 0 and len(x) > 0:
        y = smooth_data(x, y, smooth_radius)
    return x, y


def add_to_plot(data, label, show_t60=False, smooth_radius=0):
    x, y = get_x_and_y(data, smooth_radius=smooth_radius)

    if show_t60:
        t60 = solve_for_t(x, y, 60)
        if t60 is not None:
            plt.axhline(y=60, xmin=0, xmax=t60, color='red', linestyle='dotted', linewidth=1)
            plt.axvline(x=t60, color='red', linestyle='dotted', linewidth=1)

    if "SURF" in label:
        linestyle = "solid"
        if "ALPHA" in label or "RGBA" in label:
            linewidth = 1.5
        else:
            linewidth = 2

    elif "CIRCLE" in label or "RECT" in label or "LINE" in label:
        linestyle = "dashed"
        if "HOLLOW" in label:
            linewidth = 1
        else:
            linewidth = 1.5
    else:
        linestyle = "dashdot"
        linewidth = 2

    plt.plot(x, y, label=label, linestyle=linestyle, linewidth=linewidth)


def finish_plot():
    plt.legend()
    plt.get_current_fig_manager().set_window_title('Benchmark Results')
    plt.show()


def solve_for_t(x, y, target_y, or_else=None):
    res = None
    for i in range(0, len(x) - 1):
        if y[i] > target_y >= y[i + 1]:
            a = (target_y - y[i + 1]) / (y[i] - y[i + 1])
            res = x[i + 1] + a * (x[i] - x[i + 1])
            break
    return res if res is not None else or_else


def smooth_data(x, y, box_radius):
    y_smoothed = []
    for i in range(len(x)):
        y_sum, n_pts = y[i], 1
        bs = min([box_radius, abs(x[i] - x[0]), abs(x[-1] - x[i])])

        i2 = i - 1
        while i2 >= 0 and abs(x[i2] - x[i]) <= bs:
            y_sum += y[i2]
            n_pts += 1
            i2 -= 1

        i2 = i + 1
        while i2 < len(x) and abs(x[i2] - x[i]) <= bs:
            y_sum += y[i2]
            n_pts += 1
            i2 += 1

        y_smoothed.append(y_sum / n_pts)
    return y_smoothed


class TestCase:

    def __init__(self, name, caption_title, screen, ent_types=tuple(e for e in EntityType), seed=RAND_SEED):
        self.name = name
        self.caption_title = caption_title
        self.screen = screen
        self.ent_types = ent_types

        self.seed = seed
        self.factory = None
        self.renderer = None
        self.clock = None

    def start(self, pause=2):
        pygame.display.set_caption(f"{self.caption_title}".replace("#", "N=0"))
        self.factory = EntityFactory(seed=self.seed)
        self.renderer = Renderer(self.screen, ent_types=self.ent_types, seed=self.seed)
        self.renderer.t = -pause

        self.clock = pygame.time.Clock()
        last_t_above_stopping_point = 0
        dt = 0

        results = {}

        running = True
        while running:
            for evt in pygame.event.get():
                if evt.type == pygame.QUIT:
                    raise ValueError("Quit")

            self.renderer.update(dt)
            self.renderer.render()
            pygame.display.flip()

            dt = self.clock.tick() / 1000.0

            n = max(0, int(int(self.renderer.t / TICK_RATE) * TICK_RATE * SPAWN_RATE))
            if len(self.renderer.entities) < n:
                last_n = len(self.renderer.entities)
                if last_n > 0:
                    results[last_n] = self.clock.get_fps()
                    if results[last_n] >= STOP_AT_FPS:
                        last_t_above_stopping_point = self.renderer.t
                    elif self.renderer.t > last_t_above_stopping_point + LOW_FPS_GRACE_PERIOD:
                        running = False
                while len(self.renderer.entities) < n:
                    self.renderer.entities.append(self.factory.get_next())

            # update caption periodically
            cap_refresh = CAPTION_REFRESH_PER_SEC
            if int((self.renderer.t + dt) * cap_refresh) > int(self.renderer.t * cap_refresh):
                pygame.display.set_caption(self.caption_title.replace("#", f"N={n}, FPS={self.clock.get_fps():.2f}"))

        return results


def _print_result(case_num, test, res, fps_to_display=(144, 120, 60, STOP_AT_FPS)):
    print(f"\n{case_num}. {test.name} Results:")
    x, y = get_x_and_y(res, smooth_radius=GRAPH_SMOOTHING_RADIUS * SPAWN_RATE)
    for fps in reversed(sorted(f for f in set(fps_to_display) if f >= STOP_AT_FPS)):
        t = solve_for_t(x, y, fps, or_else=0)
        print(f"  {fps:>3} FPS: {round(t):>4} entities")


def _build_test_cases(screen):
    ents_to_test = []
    if INCLUDE_SURFACES:
        ents_to_test.extend([EntityType.SURF_RGB, EntityType.SURF_RGBA, EntityType.SURF_RGB_WITH_ALPHA])
    if INCLUDE_FILLED_SHAPES:
        ents_to_test.extend([EntityType.RECT_FILLED, EntityType.CIRCLE_FILLED])
    if INCLUDE_LINES:
        ents_to_test.append(EntityType.LINE)
    if INCLUDE_HOLLOW_SHAPES:
        ents_to_test.extend([EntityType.RECT_HOLLOW, EntityType.CIRCLE_HOLLOW])

    test_cases = []
    if len(ents_to_test) == 0:
        raise ValueError("Nothing to test, all cases are disabled.")
    elif len(ents_to_test) == 1:
        e = ents_to_test[0]
        test_cases.append(TestCase(e.name, f"{e.name} (#, CASE=1/1)", screen, (e,)))
    else:
        n_cases = 1 + len(ents_to_test)
        test_cases = [TestCase("ALL", f"ALL (#, CASE=1/{n_cases})", screen, ents_to_test)]
        for e_idx, e in enumerate(ents_to_test):
            test_cases.append(TestCase(e.name, f"{e.name} (#, CASE={2 + e_idx}/{n_cases})", screen, (e,)))
    return test_cases


def _run():
    pygame.init()
    screen = pygame.display.set_mode(SCREEN_SIZE)

    _print_info()
    test_cases = _build_test_cases(screen)

    all_results = {}
    try:
        for test in test_cases:
            res = test.start(pause=PAUSE_BETWEEN_TESTS_TIME)
            all_results[test.name] = res
            _print_result(test_cases.index(test) + 1, test, res)
    except ValueError as err:
        if str(err) == "Quit":
            msg = "no results to show" if len(all_results) == 0 else "showing partial results"
            print(f"\nBenchmark was cancelled before completion ({msg})")
            pass  # show partial results (if possible) when you quit
        else:
            raise err

    pygame.display.quit()

    # display the plot
    if len(all_results) > 0:
        print("\nDisplaying plot...")
        pg_ver = pygame.version.ver
        sdl_ver = ".".join(str(v) for v in pygame.version.SDL)
        py_ver = ".".join(str(v) for v in sys.version_info[:3])
        start_plot(f"FPS vs. Entities ({ENT_SIZE}x{ENT_SIZE})",
                   subtitle=f"pygame {pg_ver} (SDL {sdl_ver}, Python {py_ver})")
        for test_name in all_results:
            add_to_plot(all_results[test_name], test_name, show_t60=(test_name == "ALL"),
                        smooth_radius=GRAPH_SMOOTHING_RADIUS * SPAWN_RATE)
        finish_plot()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='A benchmarking program for pygame rendering.')
    parser.add_argument('--size', type=int, metavar="int", default=SCREEN_SIZE, nargs=2, help=f'the window size (default {SCREEN_SIZE[0]} {SCREEN_SIZE[1]})')
    parser.add_argument('--entity-size', type=int, metavar="int", default=ENT_SIZE, help=f'the size of entities in both dimensions (default {ENT_SIZE} px)')

    parser.add_argument('--skip-surfaces', dest='surfaces', action='store_false', default=True, help=f'if used, will skip tests for RGB, RGBA, and RGB (with alpha) surfaces')
    parser.add_argument('--skip-filled', dest='filled', action='store_false', default=True, help=f'if used, will skip tests for pygame.draw.rect and circle with width = 0')
    parser.add_argument('--skip-hollow', dest='hollow', action='store_false', default=True, help=f'if used, will skip tests for pygame.draw.rect and circle with width > 0')
    parser.add_argument('--skip-lines', dest='lines', action='store_false', default=True, help=f'if used, will skip tests for pygame.draw.line')

    parser.add_argument('--spawn-rate', type=int, metavar="int", default=SPAWN_RATE, help=f'number of entities to spawn per second (default {SPAWN_RATE}, smaller = slower and more accurate)')
    parser.add_argument('--tick-rate', type=float, metavar="float", default=TICK_RATE, help=f'how frequently to sample the FPS and add new entities (in seconds, default {TICK_RATE})')
    parser.add_argument('--fps-thresh', type=int, metavar="int", default=STOP_AT_FPS, help=f'the FPS at which a test case should stop (default {STOP_AT_FPS})')

    parser.add_argument('--smooth', type=float, metavar="float", default=GRAPH_SMOOTHING_RADIUS, help=f'how much to smooth the graph, in seconds (default {GRAPH_SMOOTHING_RADIUS}, or use 0 for none)')
    parser.add_argument('--no-log-axis', dest='log_axis', action='store_false', default=LOG_AXIS, help=f'if used, will disable log-scaling on the graph\'s y-axis')

    parser.add_argument('--seed', type=int, metavar="int", default=RAND_SEED, help=f'random seed (default={RAND_SEED}, use 0 to generate one)')

    args = parser.parse_args()

    SCREEN_SIZE = args.size
    ENT_SIZE = args.entity_size

    INCLUDE_SURFACES = args.surfaces
    INCLUDE_FILLED_SHAPES = args.filled
    INCLUDE_HOLLOW_SHAPES = args.hollow
    INCLUDE_LINES = args.lines

    SPAWN_RATE = args.spawn_rate
    TICK_RATE = args.tick_rate
    STOP_AT_FPS = args.fps_thresh
    GRAPH_SMOOTHING_RADIUS = args.smooth
    LOG_AXIS = args.log_axis
    RAND_SEED = args.seed if args.seed > 0 else None

    _run()
