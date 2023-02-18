
import math
import numpy

import pygame
import pygame._sdl2 as sdl2


def apply_rainbow(surface: pygame.Surface, offset=0., strength=0.666, bands=2.) -> pygame.Surface:
    """Adds a rainbow effect to an image.

        Note that this returns a new surface and does not modify the original.

        Args:
            surface: The original image.
            offset: A value from 0 to 1 that applies a color shift to the rainbow. Changing this parameter
                    over time will create an animated looping effect. Values outside the interval [0, 1) will
                    act as though they're modulo 1.
            strength: A value from 0 to 1 that determines how strongly the rainbow's color should appear.
            bands: A value greater than 0 that determines how many color bands should be rendered (in other words,
                    how "thin" the rainbow should appear).
    """
    x = numpy.linspace(0, 1, surface.get_width())
    y = numpy.linspace(0, 1, surface.get_height())
    gradient = numpy.outer(x, y) * bands

    red_mult = numpy.sin(math.pi * 2 * (gradient + offset)) * 0.5 + 0.5
    green_mult = numpy.sin(math.pi * 2 * (gradient + offset + 0.25)) * 0.5 + 0.5
    blue_mult = numpy.sin(math.pi * 2 * (gradient + offset + 0.5)) * 0.5 + 0.5

    res = surface.copy()

    red_pixels = pygame.surfarray.pixels_red(res)
    red_pixels[:] = (red_pixels * (1 - strength) + red_pixels * red_mult * strength).astype(dtype='uint16')

    green_pixels = pygame.surfarray.pixels_green(res)
    green_pixels[:] = (green_pixels * (1 - strength) + green_pixels * green_mult * strength).astype(dtype='uint16')

    blue_pixels = pygame.surfarray.pixels_blue(res)
    blue_pixels[:] = (blue_pixels * (1 - strength) + blue_pixels * blue_mult * strength).astype(dtype='uint16')

    return res


def make_fancy_scaled_display(
        size,
        scale_factor=0.,
        extra_flags=0,
        outer_fill_color=None) -> pygame.Surface:
    """Creates a SCALED pygame display with a custom scale factor and background color.

        Args:
            size: The base resolution of the display surface.
            extra_flags: Extra flags (aside from SCALED) to give the display, e.g. RESIZABLE.
            scale_factor: The initial scaling factor for the window.
                    For example, if the display's base size is 64x32 and this arg is 5, the window will be 320x160
                    in the physical display. If this arg is 0 or less, the window will use the default SCALED behavior
                    of filling as much space as the computer's display will allow.
                    Non-integer values greater than 1 can be used here too. Positive values less than 1 will act like 1.
            outer_fill_color: When the display surface can't cleanly fill the physical window with an integer scale
                    factor, a solid color is used to fill the empty space. This arg lets you set that color (it's black
                    by default).
        Returns:
            The display surface.
    """

    # create the display in "hidden" mode, because it isn't properly sized yet
    res = pygame.display.set_mode(size, pygame.SCALED | extra_flags | pygame.HIDDEN)
    window = sdl2.Window.from_display_module()

    # due to a bug, we *cannot* let this Window object get GC'd
    # https://github.com/pygame-community/pygame-ce/issues/1889
    globals()["sdl2_Window_ref"] = window  # so store it somewhere safe...

    # resize the window to achieve the desired scale factor
    if scale_factor > 0:
        initial_scale_factor = max(scale_factor, 1)  # scale must be >= 1
        window.size = (int(size[0] * initial_scale_factor),
                       int(size[1] * initial_scale_factor))
        window.position = sdl2.WINDOWPOS_CENTERED  # recenter it too

    # set the out-of-bounds color
    if outer_fill_color is not None:
        renderer = sdl2.Renderer.from_window(window)
        renderer.draw_color = pygame.Color(outer_fill_color)

    # show the window (unless they wanted it hidden)
    if not (pygame.HIDDEN & extra_flags):
        window.show()

    return res


if __name__ == "__main__":
    pygame.init()

    fill_color = (92, 64, 92)

    screen = make_fancy_scaled_display(
        (256, 128),
        extra_flags=pygame.RESIZABLE,
        scale_factor=3,
        outer_fill_color=fill_color
    )

    pygame.display.set_caption("rainbowize.py")

    frog_img = pygame.image.load("data/frog.png").convert_alpha()
    frog_img = pygame.transform.scale(frog_img, (frog_img.get_width() * 3,
                                                 frog_img.get_height() * 3))

    clock = pygame.time.Clock()

    while True:
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                raise SystemExit()
            if e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE:
                raise SystemExit()

        screen.fill(fill_color)

        # loop the animation 1.5 times per second
        elapsed_time_ms = pygame.time.get_ticks()
        animation_period_ms = 666

        rainbow_frog_img = apply_rainbow(
            frog_img,
            offset=elapsed_time_ms / animation_period_ms,
            bands=1.2
        )

        scr_w, scr_h = screen.get_size()
        screen.blit(frog_img, (scr_w / 4 - frog_img.get_width() / 2, scr_h / 2 - frog_img.get_height() / 2))
        screen.blit(rainbow_frog_img, (3 * scr_w / 4 - rainbow_frog_img.get_width() / 2,
                                       scr_h / 2 - rainbow_frog_img.get_height() / 2))

        pygame.display.flip()
        clock.tick(60)
