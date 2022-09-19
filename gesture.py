import pygame
import kivy.gesture


class InProgressGesture:

    def __init__(self):
        self.complete_paths = []
        self.active_path = None

    def is_active(self):
        return self.active_path is not None

    def is_empty(self):
        return len(self.complete_paths) == 0 and (self.active_path is None or len(self.active_path) == 0)

    def add_point(self, pt):
        if self.active_path is None:
            raise ValueError("there's no active path")
        if len(self.active_path) > 0 and self.active_path[-1] == pt:
            return  # don't add dupes
        self.active_path.append(pt)

    def start_path(self, pt):
        if self.active_path is None:
            self.active_path = []
        if len(self.active_path) > 0:
            print("WARN: started a path while one was active; ending it.")
            self.end_path(pt=pt)
            self.active_path = []
        self.active_path.append(pt)

    def end_path(self, pt=None):
        if self.active_path is None:
            raise ValueError("there's no active path")
        if pt is not None:
            self.add_point(pt)
        if len(self.active_path) > 0:
            self.complete_paths.append(self.active_path)
        self.active_path = None

    def build(self, name) -> 'SavedGesture':
        if self.is_active():
            self.end_path()
        return SavedGesture(name, self.complete_paths)

    def _draw_path(self, path, screen, color, width, offset=(0, 0)):
        for idx in range(len(path) - 1):
            p1 = (path[idx][0] + offset[0], path[idx][1] + offset[1])
            p2 = (path[idx + 1][0] + offset[0], path[idx + 1][1] + offset[1])
            pygame.draw.line(screen, color, p1, p2, width=width)

    def draw(self, screen, complete_color, active_color, width=3):
        for path in self.complete_paths:
            self._draw_path(path, screen, complete_color, width)
        if self.active_path is not None:
            self._draw_path(self.active_path, screen, active_color, width)


class SavedGesture:

    def __init__(self, name, paths, norm_rect=(0.0, 0.0, 1.0, 1.0)):
        self.name = name
        self.norm_rect = norm_rect
        self.paths = SavedGesture._normalize(paths, norm_rect=self.norm_rect)

        self.kivy_gesture = kivy.gesture.Gesture()
        for path in paths:
            self.kivy_gesture.add_stroke(path)
        self.kivy_gesture.normalize()

    def __repr__(self):
        return self.name

    def get_score(self, other: 'SavedGesture'):
        return self.kivy_gesture.get_score(other.kivy_gesture)

    def draw(self, surf, rect, color, width=3):
        for path in self.paths:
            for idx in range(len(path) - 1):
                p1_raw = path[idx]
                p1 = (round(rect[0] + (p1_raw[0] - self.norm_rect[0]) * rect[2] / self.norm_rect[2]),
                      round(rect[1] + (p1_raw[1] - self.norm_rect[1]) * rect[3] / self.norm_rect[3]))
                p2_raw = path[idx + 1]
                p2 = (round(rect[0] + (p2_raw[0] - self.norm_rect[0]) * rect[2] / self.norm_rect[2]),
                      round(rect[1] + (p2_raw[1] - self.norm_rect[1]) * rect[3] / self.norm_rect[3]))
                pygame.draw.line(surf, color, p1, p2, width=width)

    @staticmethod
    def _normalize(paths, norm_rect=(0.0, 0.0, 1.0, 1.0)):
        min_x = float('inf')
        max_x = -float('inf')
        min_y = float('inf')
        max_y = -float('inf')

        for path in paths:
            for pt in path:
                min_x = min(pt[0], min_x)
                max_x = max(pt[0], max_x)
                min_y = min(pt[1], min_y)
                max_y = max(pt[1], max_y)

        if min_x == float('inf'):
            return []

        if min_x == max_x:
            min_x -= 1.0
            max_x += 1.0

        if min_y == max_y:
            min_y -= 1.0
            max_y += 1.0

        res = []
        for path in paths:
            norm_path = []
            for pt in path:
                ax = (pt[0] - min_x) / (max_x - min_x)
                ay = (pt[1] - min_y) / (max_y - min_y)
                norm_path.append((norm_rect[0] + ax * norm_rect[2],
                                  norm_rect[1] + ay * norm_rect[3]))
            res.append(norm_path)
        return res


def interpolate_color(c1, c2, a):
    r = int(c1[0] + a * (c2[0] - c1[0]))
    g = int(c1[1] + a * (c2[1] - c1[1]))
    b = int(c1[2] + a * (c2[2] - c1[2]))
    return (r, g, b)


def start():
    pygame.init()
    screen = pygame.display.set_mode((640, 480))
    clock = pygame.time.Clock()

    saved_gestures = []
    gesture = InProgressGesture()

    font = pygame.font.Font(pygame.font.get_default_font(), 16)

    print("Controls:\n"
          "  [Mouse] to draw\n"
          "  [Enter] to save gesture\n"
          "  [Escape] to clear screen\n"
          "  [Backspace] to delete saved gesture")

    running = True
    while running:
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                running = False
            elif e.type == pygame.MOUSEBUTTONDOWN:
                if e.button == 1:
                    gesture.start_path(e.pos)
            elif e.type == pygame.MOUSEBUTTONUP:
                if e.button == 1 and gesture.is_active():
                    gesture.end_path(e.pos)
            elif e.type == pygame.MOUSEMOTION:
                if gesture.is_active():
                    gesture.add_point(e.pos)
            elif e.type == pygame.KEYDOWN:
                if e.key == pygame.K_RETURN:
                    name = f"Gesture {len(saved_gestures)}"
                    gest = gesture.build(name)
                    saved_gestures.append(gest)
                    print(f"Saved Gesture: {name}")
                    gesture = InProgressGesture()
                elif e.key == pygame.K_BACKSPACE:
                    if len(saved_gestures) > 0:
                        saved_gestures.pop(-1)
                elif e.key == pygame.K_ESCAPE:
                    gesture = InProgressGesture()

        screen.fill((255, 255, 255))

        temp_saved_gesture = gesture.build("temp") if (not gesture.is_active() and not gesture.is_empty()) else None

        bad_color = (255, 128, 128)
        good_color = (128, 255, 128)

        cols = 8
        cell_size = (screen.get_width() // cols, screen.get_width() // cols + font.get_height())
        for i in range(len(saved_gestures)):
            rect = (int((i % cols) * cell_size[0]), int(i // cols) * cell_size[1],
                    cell_size[0], cell_size[1] - font.get_height())
            if temp_saved_gesture is not None:
                score = saved_gestures[i].get_score(temp_saved_gesture)
                adj_score = min(1, max(0, (score + 1) / 2))
                bg_color = interpolate_color(bad_color, good_color, adj_score)
                pygame.draw.rect(screen, bg_color, rect)
                score_surf = font.render(f"{score:.3f}", True, (0, 0, 0))
                screen.blit(score_surf, (int(rect[0] + rect[2] / 2 - score_surf.get_width() / 2), rect[1] + rect[3]))
            else:
                pygame.draw.rect(screen, (0, 0, 0), rect, width=1)
            saved_gestures[i].draw(screen, rect, (0, 0, 0))

        gesture.draw(screen, (0, 0, 0), (255, 50, 50))
        pygame.display.flip()
        clock.tick(60)


if __name__ == "__main__":
    start()