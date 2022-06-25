import pygame

try:
    import numpy  # required for pygame.surfarray stuff
    _has_numpy = True
except ImportError:
    _has_numpy = False


_CACHE = {}


def apply_lut(source: pygame.Surface, lut: pygame.Surface, idx: int, cache=True) -> pygame.Surface:
    """Generates a new copy of a surface with a color lookup table applied.
    source: Base image.
    lut: The "lookup table" surface. The first column of pixels should match the colors in the base image, and
      subsequent columns should contain different re-mappings for those colors.
    idx: Which column of the lookup table to use. If 0, colors will not be changed (since that's the "key column").
    cache: Whether to cache the result for quick lookup later. Note that this assumes the pixel values in the source
      and lut surfaces won't change between calls.
    """
    # this keying system assumes the source and lut's pixel values are static
    cache_key = (id(source), id(lut), idx) if cache else None

    if cache_key in _CACHE:
        return _CACHE[cache_key]
    else:
        if _has_numpy:
            res = _numpy_lut_iterative(source, lut, idx)
        else:
            res = _basic_pygame_lut(source, lut, idx)

        if cache_key is not None:
            _CACHE[cache_key] = res
        return res


def _numpy_lut_iterative(source: pygame.Surface, lut: pygame.Surface, idx: int):
    # translucency isn't supported in lut
    if lut.get_flags() & pygame.SRCALPHA:
        lut = lut.convert()

    if source.get_flags() & pygame.SRCALPHA:
        # preserve source's alpha channel if it has one
        orig_alpha_array = pygame.surfarray.pixels_alpha(source)
        source = source.convert()
    else:
        orig_alpha_array = None

    res = source.copy()
    res_array = pygame.surfarray.pixels2d(res)

    source_array = pygame.surfarray.pixels2d(source)
    lut_array = pygame.surfarray.pixels2d(lut)

    # do the actual color swapping
    for y in range(lut.get_height()):
        res_array[source_array == lut_array[0, y]] = lut_array[idx, y]

    # restore alpha channel, if necessary
    if orig_alpha_array is not None:
        res = res.convert_alpha()
        res_alpha = pygame.surfarray.pixels_alpha(res)
        res_alpha[:] = orig_alpha_array

    return res


def _basic_pygame_lut(source: pygame.Surface, lut: pygame.Surface, idx: int):
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

    BASE_IMG: pygame.Surface = None
    LUT_IMG: pygame.Surface = None
    LUT_IDX = 1
    NUM_LUTS = 2

    def load_sample_images():
        global BASE_IMG, LUT_IMG, LUT_IDX, NUM_LUTS
        BASE_IMG = pygame.image.load("data/frog_src.png").convert()
        BASE_IMG.set_colorkey(BASE_IMG.get_at((0, 0)))

        LUT_IMG = pygame.image.load("data/frog_lut.png").convert()
        NUM_LUTS = LUT_IMG.get_width()
        LUT_IDX = min(LUT_IDX, NUM_LUTS - 1)

    load_sample_images()

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
                elif e.key == pygame.K_r:
                    print("INFO: reloading images...")
                    load_sample_images()

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
