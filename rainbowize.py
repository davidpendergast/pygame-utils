import pygame
import math
import numpy


def apply(sprite: pygame.Surface, prog: int, period=20, strength=0.66, bands=0.6) -> pygame.Surface:
    x = numpy.linspace(0, 1, sprite.get_width())
    y = numpy.linspace(0, 1, sprite.get_height())
    gradient = numpy.outer(x, y) * bands

    red_mult = numpy.sin(math.pi * 2 * (gradient + prog / period)) * 0.5 + 0.5
    green_mult = numpy.sin(math.pi * 2 * (gradient + prog / period + 0.25)) * 0.5 + 0.5
    blue_mult = numpy.sin(math.pi * 2 * (gradient + prog / period + 0.5)) * 0.5 + 0.5

    sprite_copy = sprite.copy()

    red_pixels = pygame.surfarray.pixels_red(sprite_copy)
    red_pixels[:] = (red_pixels * (1 - strength) + red_pixels * red_mult * strength).astype(dtype='uint16')

    green_pixels = pygame.surfarray.pixels_green(sprite_copy)
    green_pixels[:] = (green_pixels * (1 - strength) + green_pixels * green_mult * strength).astype(dtype='uint16')

    blue_pixels = pygame.surfarray.pixels_blue(sprite_copy)
    blue_pixels[:] = (blue_pixels * (1 - strength) + blue_pixels * blue_mult * strength).astype(dtype='uint16')

    return sprite_copy


if __name__ == "__main__":
    pygame.init()
    pygame.display.set_mode((640, 480))
    pygame.display.set_caption("rainbowize.py")

    frog = pygame.image.load("data/frog.png").convert_alpha()
    w, h = frog.get_size()
    frog = pygame.transform.scale(frog, (w * 3, h * 3))

    clock = pygame.time.Clock()
    i = 0

    while True:
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                raise SystemExit()

        screen = pygame.display.get_surface()
        screen.fill((0, 0, 0))

        frog_rainbowized = apply(frog, i)
        screen.blit(frog, (100, 300))
        screen.blit(frog_rainbowized, (200, 300))

        i += 1
        pygame.display.flip()
        clock.tick(60)

