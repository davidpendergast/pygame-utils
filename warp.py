import typing

import pygame
import numpy
import cv2


def warp(surf: pygame.Surface,
         warp_pts,
         smooth=True,
         out: pygame.Surface = None) -> typing.Tuple[pygame.Surface, pygame.Rect]:
    """Stretches a pygame surface to fill a quad using cv2's perspective warp.

        Args:
            surf: The surface to transform.
            warp_pts: A list of four xy coordinates representing the polygon to fill.
                Points should be specified in clockwise order starting from the top left.
            smooth: Whether to use linear interpolation for the image transformation.
                If false, nearest neighbor will be used.
            out: An optional surface to use for the final output. If None or not
                the correct size, a new surface will be made instead.

        Returns:
            [0]: A Surface containing the warped image.
            [1]: A Rect describing where to blit the output surface to make its coordinates
                match the input coordinates.
    """
    if len(warp_pts) != 4:
        raise ValueError("warp_pts must contain four points")

    w, h = surf.get_size()
    is_alpha = surf.get_flags() & pygame.SRCALPHA

    # XXX throughout this method we need to swap x and y coordinates
    # when we pass stuff between pygame and cv2. I'm not sure why .-.
    src_corners = numpy.float32([(0, 0), (0, w), (h, w), (h, 0)])
    quad = [tuple(reversed(p)) for p in warp_pts]

    # find the bounding box of warp points
    # (this gives the size and position of the final output surface).
    min_x, max_x = float('inf'), -float('inf')
    min_y, max_y = float('inf'), -float('inf')
    for p in quad:
        min_x, max_x = min(min_x, p[0]), max(max_x, p[0])
        min_y, max_y = min(min_y, p[1]), max(max_y, p[1])
    warp_bounding_box = pygame.Rect(int(min_x), int(min_y),
                                    int(max_x - min_x),
                                    int(max_y - min_y))

    shifted_quad = [(p[0] - min_x, p[1] - min_y) for p in quad]
    dst_corners = numpy.float32(shifted_quad)

    mat = cv2.getPerspectiveTransform(src_corners, dst_corners)

    orig_rgb = pygame.surfarray.pixels3d(surf)

    flags = cv2.INTER_LINEAR if smooth else cv2.INTER_NEAREST
    out_rgb = cv2.warpPerspective(orig_rgb, mat, warp_bounding_box.size, flags=flags)

    if out is None or out.get_size() != out_rgb.shape[0:2]:
        out = pygame.Surface(out_rgb.shape[0:2], pygame.SRCALPHA if is_alpha else 0)

    pygame.surfarray.blit_array(out, out_rgb)

    if is_alpha:
        orig_alpha = pygame.surfarray.pixels_alpha(surf)
        out_alpha = cv2.warpPerspective(orig_alpha, mat, warp_bounding_box.size, flags=flags)
        alpha_px = pygame.surfarray.pixels_alpha(out)
        alpha_px[:] = out_alpha
    else:
        out.set_colorkey(surf.get_colorkey())

    # XXX swap x and y once again...
    return out, pygame.Rect(warp_bounding_box.y, warp_bounding_box.x,
                            warp_bounding_box.h, warp_bounding_box.w)


if __name__ == "__main__":
    pygame.init()

    screen = pygame.display.set_mode((640, 480))
    pygame.display.set_caption("warp.py")

    frog_img = pygame.image.load("data/frog.png").convert_alpha()
    frog_img = pygame.transform.scale(frog_img, (frog_img.get_width() * 5,
                                                 frog_img.get_height() * 5))
    default_rect = frog_img.get_rect(center=screen.get_rect().center)
    warped_frog_img = None

    corners = [default_rect.topleft, default_rect.topright,
               default_rect.bottomright, default_rect.bottomleft]
    held_corner_idx = -1

    automatic_demo_mode = True
    t = 0

    clock = pygame.time.Clock()
    font = pygame.font.SysFont("", 24)

    while True:
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                raise SystemExit()
            elif e.type == pygame.KEYDOWN:
                if e.key == pygame.K_ESCAPE:
                    raise SystemExit()
                elif e.key == pygame.K_RETURN:
                    # [Enter] = toggle demo mode
                    automatic_demo_mode = not automatic_demo_mode
                elif e.key == pygame.K_r:
                    # [R] = reset corners
                    corners = [default_rect.topleft, default_rect.topright,
                               default_rect.bottomright, default_rect.bottomleft]
            elif e.type == pygame.MOUSEBUTTONDOWN:
                if e.button == 1:
                    # [LMB] = move a corner to a new position and start dragging it
                    automatic_demo_mode = False  # enter manual mode

                    # find the point closest to the click
                    epos_vector = pygame.Vector2(e.pos)
                    best_idx, best_dist = 0, float('inf')
                    for idx, c in enumerate(corners):
                        c_dist = epos_vector.distance_to(c)
                        if c_dist < best_dist:
                            best_idx = idx
                            best_dist = c_dist
                    held_corner_idx = best_idx  # indicate we're dragging that point
                    corners[best_idx] = e.pos   # move the point to the click location
            elif e.type == pygame.MOUSEBUTTONUP:
                if e.button == 1:
                    held_corner_idx = -1  # release the point we're dragging

        keys = pygame.key.get_pressed()

        # move the 'held point' to the mouse's location
        mouse_pos = pygame.mouse.get_pos()
        if mouse_pos is not None and held_corner_idx >= 0:
            corners[held_corner_idx] = mouse_pos

        # make the points oscillate in circles if we're in 'demo mode'
        if automatic_demo_mode:
            perturbs = [pygame.Vector2() for _ in range(4)]
            for idx, pert in enumerate(perturbs):
                pert.from_polar(((idx + 1) * 15, (5 - idx) * 30 * t))  # circular motion
            pts_to_use = [pert + pygame.Vector2(c) for c, pert in zip(corners, perturbs)]
        else:
            pts_to_use = corners

        screen.fill((40, 45, 50))

        # generate the warped image
        warped_frog_img, warped_pos = warp(
            frog_img,
            pts_to_use,
            smooth=not keys[pygame.K_SPACE],  # toggle smoothing while [Space] is held
            out=warped_frog_img)

        # draw green border around the warped image
        pygame.draw.rect(
            screen, "limegreen",
            warped_frog_img.get_rect(topleft=warped_pos.topleft), width=1)
        border_text = font.render(f"pos=({warped_pos.x}, {warped_pos.y})", True, "lime")
        screen.blit(border_text, (warped_pos.x, warped_pos.y - border_text.get_height() - 2))

        # draw red warp guidelines
        pygame.draw.line(screen, "red2", pts_to_use[0], pts_to_use[2], width=1)
        pygame.draw.line(screen, "red2", pts_to_use[1], pts_to_use[3], width=1)
        for i in range(len(pts_to_use)):
            pygame.draw.line(screen, "red2", pts_to_use[i], pts_to_use[(i + 1) % 4], width=2)

        # draw labels on warp points
        for i, pt in enumerate(pts_to_use):
            text_img = font.render(f"p{i}=({int(pt[0])}, {int(pt[1])})", True, "red")
            scr_pos = (pt[0] + 2, pt[1]) if i in (1, 2) else (pt[0] - text_img.get_width() - 2, pt[1])
            screen.blit(text_img, scr_pos)

        # blit actual warped image
        screen.blit(warped_frog_img, warped_pos)

        pygame.display.flip()
        t += clock.tick(60) / 1000.0
