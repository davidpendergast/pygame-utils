import pygame
import numpy


def apply_lut(source: pygame.Surface, lut: pygame.Surface, idx: int) -> pygame.Surface:
    """Generates a new copy of an image with a color lookup table applied.
    source: Base image.
    lut: The "lookup table" surface. The first column of pixels should match the colors in the base image, and
      subsequent columns should contain different mappings for those colors.
    idx: Which column of the lookup table to use. If 0, colors will not be changed (since that's the "key column").
    """
    # return _basic_pygame_lut(source, lut, idx)
    return _numpy_lut_iterative(source, lut, idx)


def _numpy_lut_iterative(source: pygame.Surface, lut: pygame.Surface, idx: int):
    if (source.get_flags() & pygame.SRCALPHA) != (lut.get_flags() & pygame.SRCALPHA):
        msg = ("source", "lut") if (source.get_flags() & pygame.SRCALPHA) else ("lut", "source")
        raise ValueError(f"Source and LUT images must have matching pixel formats ({msg[0]} has per-pixel alpha "
                         f"and {msg[1]} doesn't).")
    res = source.copy()
    res_array = pygame.surfarray.pixels2d(res)
    orig_array = pygame.surfarray.pixels2d(source)
    lut_array = pygame.surfarray.pixels2d(lut)

    # one array operation per mapping in the LUT... can we do better?
    for y in range(lut.get_height()):
        res_array[orig_array == lut_array[0, y]] = lut_array[idx, y]

    return res


def _basic_pygame_lut(source: pygame.Surface, lut: pygame.Surface, idx: int):
    """Works, but slow"""
    res = source.copy()

    table = {}
    for y in range(lut.get_height()):
        table[tuple(lut.get_at((0, y)))] = lut.get_at((idx, y))

    for y in range(res.get_height()):
        for x in range(res.get_width()):
            c = tuple(res.get_at((x, y)))
            if c in table:
                res.set_at((x, y), table[c])

    return res


if __name__ == "__main__":
    pygame.init()

    FPS = 60

    screen = pygame.display.set_mode((640, 480), flags=pygame.RESIZABLE)
    clock = pygame.time.Clock()

    BASE_IMG = pygame.image.load("data/frog_src.png").convert()
    BASE_IMG.set_colorkey(BASE_IMG.get_at((0, 0)))

    LUT_IMG = pygame.image.load("data/frog_lut.png").convert()
    LUT_IDX = 1
    NUM_LUTS = LUT_IMG.get_width()

    running = True
    last_update_time = pygame.time.get_ticks()

    def _draw_in_rect(surf, img, rect: pygame.Rect):
        if img.get_height() / img.get_width() >= rect[3] / rect[2]:
            xformed = pygame.transform.scale(img, (img.get_width() * rect[3] / img.get_height(), rect[3]))
        else:
            xformed = pygame.transform.scale(img, (rect[2], img.get_height() * rect[2] / img.get_width()))
        blit_rect = xformed.get_rect(center=rect.center)
        surf.blit(xformed, blit_rect)
        return blit_rect

    while running:
        current_time = pygame.time.get_ticks()
        dt = (current_time - last_update_time) / 1000
        pygame.time.get_ticks()
        for e in pygame.event.get():
            if e.type == pygame.QUIT or (e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE):
                running = False
            elif e.type == pygame.KEYDOWN:
                if e.key == pygame.K_RIGHT:
                    LUT_IDX += 1
                elif e.key == pygame.K_LEFT:
                    LUT_IDX -= 1

        LUT_IDX %= NUM_LUTS
        RESULT_IMG = apply_lut(BASE_IMG, LUT_IMG, LUT_IDX)

        screen = pygame.display.get_surface()
        W, H = screen.get_size()
        screen.fill((66, 66, 66))

        to_draw = [BASE_IMG, LUT_IMG, RESULT_IMG]

        for i, img in enumerate(to_draw):
            rect = pygame.Rect(i * W / 3, 0, int(W / 3), H)
            blit_rect = _draw_in_rect(screen, img, rect)
            if img == LUT_IMG:
                selection_rect = pygame.Rect(blit_rect.x + LUT_IDX * blit_rect.width / NUM_LUTS,
                                             blit_rect.y,
                                             blit_rect.width / NUM_LUTS,
                                             blit_rect.height)
                pygame.draw.rect(screen, (255, 0, 0), selection_rect, width=2)

        pygame.display.flip()

        if current_time // 1000 > last_update_time // 1000:
            pygame.display.set_caption(
                "LUT Demo (FPS={:.1f}, TARGET_FPS={})".format(clock.get_fps(), FPS))

        last_update_time = current_time
        clock.tick(FPS)
