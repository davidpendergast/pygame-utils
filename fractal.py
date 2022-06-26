import asyncio
import pygame
import numpy

W, H = 600, 450

FPS = 60
TOTAL_ITERATIONS_PER_FRAME = 100_000
MAX_ITERS_PER_CELL_PER_FRAME = 250

SHOW_BOX = True          # B to toggle
SIMPLE_COLORING = False  # C to toggle

FLAGS = 0  # | pygame.SCALED

ABS_LIMIT = 2


class State:

    def __init__(self, view_rect=(-2, -2 * (H / W), 4, 4 * (H / W))):
        self.colors = numpy.zeros((W, H), numpy.int32)

        self.rect = view_rect
        self.iter_counts = numpy.zeros((W, H))

        self.last_iters_0 = 1
        self.last_iters_1 = 0

        self.min_escaped_iter = 0
        self.max_escaped_iter = 1

        yy, xx = numpy.meshgrid([self.rect[1] + i * self.rect[3] / H for i in range(0, H)],
                                [self.rect[0] + i * self.rect[2] / W for i in range(0, W)])

        self.c = xx - 1j * yy
        self.z = numpy.zeros((W, H), dtype=complex)

        self.cells_to_compute = numpy.abs(self.z) < ABS_LIMIT

    def update(self, total_iterations=TOTAL_ITERATIONS_PER_FRAME):
        cells = self.cells_to_compute
        n_cells = numpy.count_nonzero(cells)

        if n_cells > 0:
            iters_per_cell = min(max(1, int(total_iterations / n_cells)), MAX_ITERS_PER_CELL_PER_FRAME)
            for i in range(iters_per_cell):
                self.step(cells)

        if numpy.any(~cells):
            self.max_escaped_iter = numpy.max(self.iter_counts[~cells])
            self.min_escaped_iter = numpy.min(self.iter_counts[~cells])
            if self.max_escaped_iter == self.min_escaped_iter:
                self.max_escaped_iter += 1

        self.update_colors()

    def get_deepest_iteration(self):
        return int(numpy.max(self.iter_counts))

    def get_pretty_rect(self) -> str:
        return "[{:.2f}, {:.2f}, {:.4f}, {:.4f}]".format(*self.rect)

    def step(self, cells):
        self.z[cells] *= self.z[cells]
        self.z[cells] += self.c[cells]
        self.iter_counts[cells] += 1

        cells[cells] = numpy.abs(self.z[cells]) < ABS_LIMIT

    def update_colors(self):
        map_to_rainbow(self.iter_counts, self.colors, self.min_escaped_iter, self.max_escaped_iter)

        in_set = self.cells_to_compute
        self.colors[in_set] = 0

    def draw(self, screen):
        pygame.surfarray.blit_array(screen, self.colors)


def map_to_rainbow(in_array, out_array, low, high):
    if not SIMPLE_COLORING:
        # start at blue and rotate backwards
        # temp = (240 - numpy.round((in_array - low) * (360 / (high - low)))) % 360
        temp = in_array - low
        numpy.multiply(temp, 360 / (high - low), out=temp)
        numpy.round(temp, out=temp)
        numpy.subtract(240, temp, out=temp)
        numpy.remainder(temp, 360, out=temp)

        hues_to_rgb(temp, out_array)
    else:
        out_array[:] = numpy.round((in_array - low) / (high - low) * 63) * 0x020401


def hues_to_rgb(h, out):
    X = numpy.zeros(h.shape, dtype=numpy.int32)
    numpy.multiply(255, 1 - numpy.abs((h / 60) % 2 - 1), out=X, casting='unsafe')

    r = numpy.zeros(h.shape, dtype=numpy.int32)
    g = numpy.zeros(h.shape, dtype=numpy.int32)
    b = numpy.zeros(h.shape, dtype=numpy.int32)

    h_lt_60 = h < 60
    h_bt_60_120 = (60 <= h) & (h < 120)
    h_bt_120_180 = (120 <= h) & (h < 180)
    h_bt_180_240 = (180 <= h) & (h < 240)
    h_bt_240_300 = (240 <= h) & (h < 300)
    h_gt_300 = 300 <= h

    r[h_lt_60] = 255
    r[h_bt_60_120] = X[h_bt_60_120]
    # r[h_bt_120_180] = 0  no-ops
    # r[h_bt_180_240] = 0
    r[h_bt_240_300] = X[h_bt_240_300]
    r[h_gt_300] = 255

    g[h_lt_60] = X[h_lt_60]
    g[h_bt_60_120] = 255
    g[h_bt_120_180] = 255
    g[h_bt_180_240] = X[h_bt_180_240]
    # g[h_bt_240_300] = 0
    # g[h_gt_300] = 0

    # b[h_lt_60] = 0
    # b[h_bt_60_120] = 0
    b[h_bt_120_180] = X[h_bt_120_180]
    b[h_bt_180_240] = 255
    b[h_bt_240_300] = 255
    b[h_gt_300] = X[h_gt_300]

    out[:] = r * 0x010000 + g * 0x000100 + b * 0x000001


async def main():
    global W, H, FPS, TOTAL_ITERATIONS_PER_FRAME, MAX_ITERS_PER_CELL_PER_FRAME, SHOW_BOX, SIMPLE_COLORING, FLAGS, ABS_LIMIT
    pygame.init()
    screen = pygame.display.set_mode((W, H), flags=FLAGS)
    state = State()
    clock = pygame.time.Clock()

    running = True
    last_update_time = pygame.time.get_ticks()

    while running:
        current_time = pygame.time.get_ticks()
        for e in pygame.event.get():
            if e.type == pygame.QUIT or (e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE):
                running = False
            elif e.type == pygame.KEYDOWN:
                if e.key == pygame.K_r:
                    print("Resetting.")
                    state = State()
                elif e.key == pygame.K_b:
                    SHOW_BOX = not SHOW_BOX
                elif e.key == pygame.K_c:
                    SIMPLE_COLORING = not SIMPLE_COLORING
            elif e.type == pygame.MOUSEBUTTONDOWN:
                r = state.rect
                xy = (r[0] + e.pos[0] / W * r[2], r[1] + e.pos[1] / H * r[3])

                zoom_change = None
                if e.button == 1:
                    zoom_change = 2.0
                elif e.button == 3:
                    zoom_change = 0.5

                if zoom_change is not None:
                    new_rect = (xy[0] - r[2] / (2 * zoom_change),
                                xy[1] - r[3] / (2 * zoom_change),
                                r[2] / zoom_change,
                                r[3] / zoom_change)
                    print("Zooming to: {}".format(new_rect))
                    state = State(view_rect=new_rect)
        state.update()
        state.draw(screen)

        mouse_xy = pygame.mouse.get_pos()
        if SHOW_BOX and 0 < mouse_xy[0] < W - 1 and 0 < mouse_xy[1] < H - 1:
            rect = pygame.Rect(mouse_xy[0] - W // 4, mouse_xy[1] - H // 4, W // 2, H // 2)
            pygame.draw.rect(screen, (255, 255, 255), rect, width=1)

        pygame.display.flip()

        if current_time // 250 > last_update_time // 250:
            pygame.display.set_caption("Fractal (FPS={:.1f}, SIZE={}, ITERS={}, VIEW={})".format(
                clock.get_fps(), (W, H), state.get_deepest_iteration(), state.get_pretty_rect()))

        last_update_time = current_time
        clock.tick(FPS)
        await asyncio.sleep(0)


if __name__ == "__main__":
    asyncio.run(main())
