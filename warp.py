import pygame
import numpy
import cv2


def bounding_box(pts):
    min_x = float('inf')
    max_x = -float('inf')
    min_y = float('inf')
    max_y = -float('inf')
    for p in pts:
        min_x = min(min_x, p[0])
        min_y = min(min_y, p[1])
        max_x = max(max_x, p[0])
        max_y = max(max_y, p[1])
    return pygame.Rect(min_x, min_y, max_x - min_x, max_y - min_y)


def warp(surf: pygame.Surface, corners):
    w, h = surf.get_size()
    src_corners = numpy.float32([(0, 0), (0, h), (w, h), (w, 0)])

    corners = [tuple(reversed(c)) for c in corners]  # no idea why this is necessary...

    dest_rect = bounding_box(corners)
    corners = [(c[0] - dest_rect.x, c[1] - dest_rect.y) for c in corners]

    dst_corners = numpy.float32(corners)
    mat = cv2.getPerspectiveTransform(src_corners, dst_corners)

    buf = pygame.surfarray.array3d(surf)
    buf2 = pygame.surfarray.array_alpha(surf)

    out = cv2.warpPerspective(buf, mat, dest_rect.size)
    out_alpha = cv2.warpPerspective(buf2, mat, dest_rect.size)

    res = pygame.Surface(out.shape[0:2], pygame.SRCALPHA)
    pygame.surfarray.blit_array(res, out)

    alpha_px = pygame.surfarray.pixels_alpha(res)
    alpha_px[:] = out_alpha

    return res, pygame.Rect(dest_rect.y, dest_rect.x, dest_rect.h, dest_rect.w)


if __name__ == "__main__":
    pygame.init()

    bg_color = pygame.Color(92, 64, 92)
    outer_bg_color = bg_color.lerp("black", 0.25)

    screen = pygame.display.set_mode((640, 480))

    pygame.display.set_caption("warp.py")

    frog_img = pygame.image.load("data/frog.png").convert_alpha()
    frog_img = pygame.transform.scale(frog_img, (frog_img.get_width() * 3,
                                                 frog_img.get_height() * 3))

    default_rect = frog_img.get_rect(center=screen.get_rect().center)

    corners = [default_rect.topleft, default_rect.topright,
               default_rect.bottomright, default_rect.bottomleft]

    held_corner_idx = -1

    clock = pygame.time.Clock()

    t = 0

    while True:
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                raise SystemExit()
            elif e.type == pygame.KEYDOWN:
                if e.key == pygame.K_ESCAPE:
                    raise SystemExit()
            elif e.type == pygame.MOUSEBUTTONDOWN:
                if e.button == 1:
                    epos_vector = pygame.Vector2(e.pos)
                    best_idx = 0
                    for idx, c in enumerate(corners):
                        if epos_vector.distance_to(c) < epos_vector.distance_to(corners[best_idx]):
                            best_idx = idx
                    held_corner_idx = best_idx
                    corners[best_idx] = e.pos
            elif e.type == pygame.MOUSEBUTTONUP:
                if e.button == 1:
                    held_corner_idx = -1

        screen.fill(bg_color)

        mouse_pos = pygame.mouse.get_pos()
        if mouse_pos is not None and held_corner_idx >= 0:
            corners[held_corner_idx] = mouse_pos

        # draw guidelines (red)
        pygame.draw.line(screen, (255, 0, 0), corners[0], corners[2])
        pygame.draw.line(screen, (255, 0, 0), corners[1], corners[3])
        for i in range(len(corners)):
            pygame.draw.line(screen, (255, 0, 0), corners[i], corners[(i + 1) % len(corners)])

        # draw actual warped image
        warped_frog_img, warped_pos = warp(frog_img, corners)
        screen.blit(warped_frog_img, warped_pos)

        # draw warped image border (green)
        pygame.draw.rect(screen, (0, 255, 0), warped_frog_img.get_rect(topleft=warped_pos.topleft), width=1)

        pygame.display.flip()
        clock.tick(60)
        t += 1
