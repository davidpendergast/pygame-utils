import pygame
import numpy
import asyncio


# Conway's Game of Life (using pygame + numpy)
# by Ghast

W, H = 1000, 600
FPS = 60
RATIO = 0.5  # Portion of cells that start out alive.

FLAGS = 0 | pygame.SCALED

class State:
    def __init__(self):
        self.grid = numpy.random.randint(0, 100, (W, H), numpy.int16)
        self.grid[self.grid <= RATIO * 100] = 1
        self.grid[self.grid > RATIO * 100] = 0

        self.neighbor_counts = numpy.zeros((W, H), numpy.int16)

    def step(self):
        self.neighbor_counts[:] = 0
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if (dx, dy) != (0, 0):
                    shifted = numpy.roll(self.grid, (dx, dy), (0, 1))
                    numpy.add(self.neighbor_counts, shifted, out=self.neighbor_counts)

        alive = self.grid > 0
        two = self.neighbor_counts == 2
        three = self.neighbor_counts == 3

        self.grid[:] = 0  # zero out the current grid

        # lots of unnecessary copying here, but it's so elegant
        self.grid[(alive & (two | three)) | ((~alive) & three)] = 1

    def draw(self, screen, color=0xFFFFFF):
        pygame.surfarray.blit_array(screen, self.grid * color)

async def main():
    global FPS, W, H, RATIO, FLAGS
    pygame.init()
    screen = pygame.display.set_mode((W, H), flags=FLAGS)
    state = State()
    clock = pygame.time.Clock()

    running = True
    last_update_time = pygame.time.get_ticks()

    while running:
        current_time = pygame.time.get_ticks()
        dt = (current_time - last_update_time) / 1000
        for e in pygame.event.get():
            if e.type == pygame.QUIT or (e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE):
                running = False
            elif e.type == pygame.KEYDOWN:
                if e.key == pygame.K_r:
                    print("Reseting.")
                    state = State()
                elif e.key == pygame.K_RIGHT:
                    FPS += 10
                    print("Increased target FPS to: {}".format(FPS))
                elif e.key == pygame.K_LEFT:
                    FPS = max(10, FPS - 10)
                    print("Decreased target FPS to: {}".format(FPS))
        state.step()
        state.draw(screen, color=0xAAFFAA)

        pygame.display.flip()

        if current_time // 1000 > last_update_time // 1000:
            pygame.display.set_caption("Game of Life (FPS={:.1f}, TARGET_FPS={}, SIZE={})".format(clock.get_fps(), FPS, (W, H)))

        last_update_time = current_time
        clock.tick(FPS)
        await asyncio.sleep(0)

if __name__ == "__main__":
    asyncio.run( main() )
