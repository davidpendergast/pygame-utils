import pygame
import matplotlib.pyplot as plt

import math
import random
import enum
import sys


# Pygame Benchmarking Program
# by Ghast (@Ghast_NEOH)


SCREEN_SIZE = 640, 480

ENT_SIZE = 64      # width & height of filled-in entities in pixels

SPAWN_RATE = 200   # entities per sec, increasing this will make the tests run faster but be less accurate
TICK_RATE = 0.10   # frequency of entity additions and fps measurements
STOP_AT_FPS = 45   # test cases stop when this FPS is reached

CAPTION_REFRESH_PER_SEC = 3     # per second
LOW_FPS_GRACE_PERIOD = 0.5      # seconds
PAUSE_BETWEEN_TESTS_TIME = 1    # seconds
GRAPH_SMOOTHING_RADIUS = 1      # seconds

PX_PER_ENT = ENT_SIZE ** 2


class EntityType(enum.IntEnum):
    SURF_RGB = 0                # Surface with no per-pixel alpha
    SURF_RGB_WITH_ALPHA = 1     # Surface with no per-pixel alpha, but an alpha value < 255
    SURF_RGBA = 2               # Surface with per-pixel alpha.

    LINE = 3                    # A line drawn via pygame.draw.line
    RECT_FILLED = 4             # A rect drawn via pygame.draw.rect (with width = 0)
    RECT_HOLLOW = 5             # A rect drawn via pygame.draw.rect (with width > 0)
    CIRCLE_FILLED = 6           # A circle drawn via pygame.draw.cricle (with width = 0)
    CIRCLE_HOLLOW = 7           # A circle drawn via pygame.draw.cricle (with width > 0)


def _calc_avg_lengths(w, h, n=10000):
    total_dx = 0
    total_dy = 0
    total_dist = 0
    for _ in range(n):
        p1 = random.randint(0, w - 1), random.randint(0, h - 1)
        p2 = random.randint(0, w - 1), random.randint(0, h - 1)
        dx = p1[0] - p2[0]
        dy = p1[1] - p2[1]
        total_dx += abs(dx)
        total_dy += abs(dy)
        total_dist += math.sqrt(dx * dx + dy * dy)
    return total_dist / n, total_dx / n, total_dy / n


# For the current screen size, these are the average distances between two random points.
AVG_LENGTH, AVG_WIDTH, AVG_HEIGHT = _calc_avg_lengths(*SCREEN_SIZE)

# Want to define the geometric entities such that on average they'll have an area of
# about PX_PER_ENT pixels. Note that these are approximate (particularly the circle one,
# which assumes the relationship between radius and pixels changed is linear (it's actually quadratic)).
LINE_THICKNESS = max(1, round(PX_PER_ENT / AVG_LENGTH))
HOLLOW_RECT_THICKNESS = max(1, round(PX_PER_ENT / (2 * (AVG_WIDTH + AVG_HEIGHT))))
HOLLOW_CIRCLE_RADIUS_MULT = 0.25
HOLLOW_CIRCLE_THICKNESS = max(1, round((AVG_LENGTH * HOLLOW_CIRCLE_RADIUS_MULT)
                                       - math.sqrt(max(0, (AVG_LENGTH * HOLLOW_CIRCLE_RADIUS_MULT) ** 2
                                                       - PX_PER_ENT / math.pi))))
FILLED_CIRCLE_RADIUS = max(1, round(math.sqrt(PX_PER_ENT / math.pi)))


def _print_info():
    print(f"\nStarting new simulation:\n"
          f"  Screen size =   {SCREEN_SIZE}\n"
          f"  Entity size =   {ENT_SIZE}x{ENT_SIZE}\n"
          f"  Minimum FPS =   {STOP_AT_FPS}")
    print(f"\nAverage number of pixels changed per render:")
    print(f"  Filled Rect:    {PX_PER_ENT:.2f} = {ENT_SIZE}**2")
    print(f"  Filled Circle:  {math.pi * FILLED_CIRCLE_RADIUS**2:.2f} = pi * {FILLED_CIRCLE_RADIUS} ** 2")
    print(f"  Line:           {AVG_LENGTH * LINE_THICKNESS:.2f} = {AVG_LENGTH:.2f} * {LINE_THICKNESS:.2f}")
    print(f"  Hollow Rect:    {2 * (AVG_WIDTH + AVG_HEIGHT) * HOLLOW_RECT_THICKNESS:.2f}"
          f" = 2 * ({AVG_WIDTH:.2f} + {AVG_HEIGHT:.2f}) * {HOLLOW_RECT_THICKNESS:.2f}")
    avg_hollow_circle_area = math.pi * ((AVG_LENGTH * HOLLOW_CIRCLE_RADIUS_MULT)**2
                                        - (AVG_LENGTH*HOLLOW_CIRCLE_RADIUS_MULT - HOLLOW_CIRCLE_THICKNESS)**2)
    print(f"  Hollow Circle:  {avg_hollow_circle_area:.2f}"
          f" = pi * (({AVG_LENGTH:.2f})**2 - ({AVG_LENGTH:.2f}"
          f" - {HOLLOW_CIRCLE_THICKNESS:.2f})**2) (approx.)")


class EntityFactory:

    def __init__(self, seed=27182818):
        self.rand = random.Random(x=seed)

    def get_next(self):
        return (random.randint(0, 4096),  # determines Entity type
                random.randint(0, 4096),  # determines (x1, y1)
                random.randint(0, 4096),  # determines (x2, y2)
                random.randint(0, 4096))  # determines color and/or opacity


class Renderer:

    def __init__(self, screen, n_pts=512, ent_types=tuple(e for e in EntityType), seed=27182818):
        self.screen = screen
        self.entities = []
        self.ent_types = ent_types
        self.random = random.Random(x=seed)

        # Instantiate the invisible points that "float" around the screen.
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
        self.colors = ["red", "green", "blue", "yellow", "cyan", "magenta"]

        # Pre-compute the surfaces that entities may need.
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

    def get_pt(self, x):
        base_pt = self.pts[x % len(self.pts)]
        vel = self.vels[x % len(self.vels)]
        pt = base_pt + vel * self.t
        x = int(pt.x) % self.screen.get_width()
        y = int(pt.y) % self.screen.get_height()
        return x, y

    def render(self, bg_color=(0, 0, 0)):
        self.screen.fill(bg_color)
        for ent in self.entities:
            # extract entity info from tuple
            ent_type = self.ent_types[ent[0] % len(self.ent_types)]
            p1 = self.get_pt(ent[1])
            p2 = self.get_pt(ent[2])
            color_idx = ent[3]

            self.render_methods[ent_type](p1, p2, color_idx)  # hack activated

    def render_SURF_RGB(self, p1, p2, color_idx):
        surf = self.rgb_surfs[color_idx % len(self.rgb_surfs)]
        self.screen.blit(surf, p1)

    def render_SURF_RGBA(self, p1, p2, color_idx):
        surf = self.rgba_surfs[color_idx % len(self.rgba_surfs)]
        self.screen.blit(surf, p1)

    def render_SURF_RGB_WITH_ALPHA(self, p1, p2, color_idx):
        surf = self.rgb_surfs_with_alpha[color_idx % len(self.rgb_surfs_with_alpha)]
        self.screen.blit(surf, p1)

    def render_LINE(self, p1, p2, color_idx):
        c = self.colors[color_idx % len(self.colors)]
        pygame.draw.line(screen, c, p1, p2, width=LINE_THICKNESS)

    def render_RECT_HOLLOW(self, p1, p2, color_idx):
        c = self.colors[color_idx % len(self.colors)]
        pygame.draw.rect(screen, c, (p1[0], p1[1], p2[0] - p1[0], p2[1] - p1[1]), width=HOLLOW_RECT_THICKNESS)

    def render_RECT_FILLED(self, p1, p2, color_idx):
        c = self.colors[color_idx % len(self.colors)]
        pygame.draw.rect(screen, c, (p1[0], p1[1], ENT_SIZE, ENT_SIZE))

    def render_CIRCLE_HOLLOW(self, p1, p2, color_idx):
        c = self.colors[color_idx % len(self.colors)]
        r = (abs(p2[0] - p1[0]) + abs(p2[1] - p1[0])) * HOLLOW_CIRCLE_RADIUS_MULT
        pygame.draw.circle(screen, c, p1, r, width=HOLLOW_CIRCLE_THICKNESS)

    def render_CIRCLE_FILLED(self, p1, p2, color_idx):
        c = self.colors[color_idx % len(self.colors)]
        pygame.draw.circle(screen, c, p1, FILLED_CIRCLE_RADIUS)


def start_plot(title, subtitle=None):
    if subtitle is not None:
        plt.suptitle(title, fontsize=16)
        plt.title(subtitle, fontsize=10, y=1)
    else:
        plt.title(title)
    plt.xlabel('Entities')
    plt.ylabel('FPS')
    plt.yscale('log')
    yticks = [15, 30, 45, 60, 120, 144, 240]
    plt.yticks(yticks, [str(yt) for yt in yticks])


def finish_plot():
    plt.legend()
    plt.get_current_fig_manager().set_window_title('Benchmark Results')
    plt.show()


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

    def __init__(self, name, caption_title, screen, ent_types=tuple(e for e in EntityType), seed=271828):
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
    for fps in reversed(sorted(set(fps_to_display))):
        t = solve_for_t(x, y, fps, or_else=float('NaN'))
        print(f"  {fps:>3} FPS: {int(t):>4} entities")


if __name__ == "__main__":
    pygame.init()
    screen = pygame.display.set_mode(SCREEN_SIZE)

    _print_info()

    n_cases = 1 + len(EntityType)
    test_cases = [TestCase("ALL", f"ALL (#, CASE=1/{n_cases})",  screen)]
    for e in EntityType:
        test_cases.append(TestCase(e.name, f"{e.name} (#, CASE={2 + int(e)}/{n_cases})", screen, (e,)))

    all_results = {}
    try:
        for test in test_cases:
            res = test.start(pause=PAUSE_BETWEEN_TESTS_TIME)
            all_results[test.name] = res
            _print_result(test_cases.index(test) + 1, test, res)
    except ValueError as err:
        if str(err) == "Quit":
            print("\nWARN: Benchmark was cancelled before completion")
            pass  # show partial results even if you quit
        else:
            raise err

    pygame.display.quit()

    # display the plots
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
