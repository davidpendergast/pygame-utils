import math

import pygame
import typing
import random
import heapq


class Utils:

    @staticmethod
    def bound(val: float, lower: float, higher: float):
        if val < lower:
            return lower
        elif val > higher:
            return higher
        else:
            return val


class Entity:

    def __init__(self, rect: pygame.Rect, solid=True, color="red"):
        self.size = rect.size

        self.pos = pygame.Vector2(rect.x, rect.y)
        self.vel = pygame.Vector2(0, 0)

        self.color = color
        self.solid = solid

        self.level = None

    def get_rect(self) -> pygame.Rect:
        return pygame.Rect(self.pos.x, self.pos.y, self.size[0], self.size[1])

    def remove_self_from_level(self):
        if self.level is not None:
            self.level.remove_entity(self)

    def update_physics(self, dt: float):
        self.pos += dt * self.vel

    def update(self, dt: float):
        self.update_physics(dt)

    def draw(self, surface: pygame.Surface, offset=(0, 0)):
        pygame.draw.rect(surface, self.color, self.get_rect().move(*offset))


class Actor(Entity):

    def __init__(self, rect: pygame.Rect, color="green"):
        super().__init__(rect, color=color)

        self.max_vel = (64, 640)  # pixels per second
        self.jump_height = 18     # pixels
        self.gravity = 200        # pixels per second^2

        self.is_grounded = False
        self.wants_to_jump = False

    def act(self, keys_pressed_this_frame: typing.Set[int]):
        pass

    def get_jump_speed(self):
        # https://openstax.org/books/university-physics-volume-1/pages/4-3-projectile-motion
        return math.sqrt(2 * self.gravity * self.jump_height)

    def try_to_jump(self):
        self.wants_to_jump = True

    def update(self, dt: float):
        if self.wants_to_jump: # and self.is_grounded:
            self.vel.y = -self.get_jump_speed()  # remember negatives are UP on the y-axis
        self.wants_to_jump = False

        super().update(dt)

    def update_physics(self, dt: float):
        self.vel.y += self.gravity * dt  # apply gravity

        # restrict velocity to its acceptable range
        self.vel.x = Utils.bound(self.vel.x, -self.max_vel[0], self.max_vel[0])
        self.vel.y = Utils.bound(self.vel.y, -self.max_vel[1], self.max_vel[1])

        super().update_physics(dt)


class Player(Actor):

    def __init__(self, rect: pygame.Rect, color="blue"):
        super().__init__(rect, color=color)

    def handle_keypress(self, key_id: int):
        if key_id in (pygame.K_SPACE, pygame.K_w, pygame.K_UP):
            self.try_to_jump()

    def act(self, keys_pressed_this_frame: typing.Set[int]):
        for k in (pygame.K_SPACE, pygame.K_w, pygame.K_UP):
            if k in keys_pressed_this_frame:
                self.try_to_jump()
                break

    def update(self, dt: float):
        pressed_keys = pygame.key.get_pressed()

        walk_dir = 0
        if pressed_keys[pygame.K_a] or pressed_keys[pygame.K_LEFT]:
            walk_dir -= 1
        if pressed_keys[pygame.K_d] or pressed_keys[pygame.K_RIGHT]:
            walk_dir += 1

        self.vel.x = walk_dir * self.max_vel[0]

        super().update(dt)


class Tile:

    def __init__(self, solid=True, color="white"):
        self.solid = solid
        self.color = color

    def get_rect(self, grid_xy: typing.Tuple[int, int],
                 size: typing.Tuple[int, int]) -> pygame.Rect:
        return pygame.Rect(grid_xy[0] * size[0], grid_xy[1] * size[1], size[0], size[1])

    def draw(self, surface: pygame.Surface,
             grid_xy: typing.Tuple[int, int],
             size: typing.Tuple[int, int],
             offset=(0, 0)):
        my_rect = self.get_rect(grid_xy, size)
        pygame.draw.rect(surface, self.color, my_rect.move(*offset))


class Level:

    def __init__(self, cell_size=(16, 16)):
        self.cell_size = cell_size

        self.tiles: typing.Dict[typing.Tuple[int, int], Tile] = {}
        self.entities: typing.List[Entity] = []

    def set_tile(self, grid_xy: typing.Tuple[int, int], tile: typing.Optional[Tile]):
        if tile is not None:
            self.tiles[grid_xy] = tile
        else:
            if grid_xy in self.tiles:
                del self.tiles[grid_xy]

    def add_entity(self, entity: Entity):
        entity.level = self
        self.entities.append(entity)

    def remove_entity(self, entity: Entity):
        entity.level = None
        self.entities.remove(entity)

    def all_actors(self) -> typing.Generator[Actor, None, None]:
        for entity in self.entities:
            if isinstance(entity, Actor):
                yield entity

    def get_player(self) -> typing.Optional[Player]:
        for actor in self.all_actors():
            if isinstance(actor, Player):
                return actor
        return None

    def all_grid_coords_in_rect(self, rect: pygame.Rect) -> typing.Generator[typing.Tuple[int, int], None, None]:
        min_xy = (rect.x // self.cell_size[0], rect.y // self.cell_size[0])
        max_xy = (rect.right // self.cell_size[1], rect.bottom // self.cell_size[1])

        for grid_x in range(min_xy[0], max_xy[0] + 1):
            for grid_y in range(min_xy[1], max_xy[1] + 1):
                yield grid_x, grid_y

    def all_entities_in_rect(self, rect: pygame.Rect) -> typing.Generator[Entity, None, None]:
        # TODO this could be improved
        for entity in self.entities:
            if entity.get_rect().colliderect(rect):
                yield entity

    def draw_tiles(self, surface: pygame.Surface, offset=(0, 0)):
        area_to_draw = surface.get_rect(topleft=offset)
        for grid_xy in self.all_grid_coords_in_rect(area_to_draw):
            if grid_xy in self.tiles:
                self.tiles[grid_xy].draw(surface, grid_xy, self.cell_size, offset=offset)

    def draw_entities(self, surface: pygame.Surface, offset=(0, 0)):
        area_to_draw = surface.get_rect(topleft=offset)
        for entity in self.all_entities_in_rect(area_to_draw):
            entity.draw(surface, offset=offset)

    def update_all(self, dt: float, keys_pressed_this_frame: typing.Set[int]):
        for entity in self.entities:
            if isinstance(entity, Actor):
                entity.act(keys_pressed_this_frame)
            entity.update(dt)

        self.resolve_collisions(max_displacement=max(self.cell_size))

    def is_overlapping_solid_tile(self, rect):
        for grid_xy in self.all_grid_coords_in_rect(rect):
            if grid_xy in self.tiles and self.tiles[grid_xy].solid:
                return True
        return False

    def resolve_collisions(self, max_displacement=float('inf')):
        solid_entities = [entity for entity in self.entities if entity.solid]
        max_displacement_sq = max_displacement ** 2

        for entity in solid_entities:
            rect = entity.get_rect()
            if not self.is_overlapping_solid_tile(rect):
                # it's not colliding with anything, easy
                shift = (0, 0)
            else:
                shift = None

                candidate_shifts = [(0, 0, 0)]
                seen_shifts = set(candidate_shifts)
                while len(candidate_shifts) > 0:
                    cur_shift = heapq.heappop(candidate_shifts)
                    new_rect = rect.move(*cur_shift[1:])

                    if not self.is_overlapping_solid_tile(new_rect):
                        shift = cur_shift[1:]
                        break
                    else:
                        for xy in ((0, -1), (-1, 0), (1, 0), (0, 1)):
                            next_shift_x = cur_shift[1] + xy[0]
                            next_shift_y = cur_shift[2] + xy[1]
                            next_shift = ((next_shift_x ** 2 + next_shift_y ** 2), next_shift_x, next_shift_y)
                            if next_shift not in seen_shifts and next_shift[0] <= max_displacement_sq:
                                seen_shifts.add(next_shift)
                                heapq.heappush(candidate_shifts, next_shift)

            if shift is None:
                print(f"INFO: {entity} was crushed!")
                entity.dead = True
            else:
                if shift[0] != 0:
                    entity.pos.x = rect.x + shift[0]
                    if entity.vel.x * shift[0] < 0:
                        entity.vel.x = 0
                if shift[1] != 0:
                    if entity.vel.y * shift[1] < 0:
                        entity.vel.y = 0
                    entity.pos.y = rect.y + shift[1]

            if isinstance(entity, Actor) and self.is_overlapping_solid_tile(rect.move(0, 1)):
                entity.is_grounded = True


def make_demo_level(dims=(32, 16), density=0.2, cell_size = (16, 16)):
    level = Level(cell_size=cell_size)
    colors = ["darkslategray1", "darkslategray2", "darkslategray3"]
    tiles = [Tile(solid=True, color=c) for c in colors]

    for x in range(dims[0]):
        for y in range(dims[1]):
            if random.random() < density:
                level.set_tile((x, y), random.choice(tiles))

    player_rect = pygame.Rect(0, 0, 8, 12)
    player_rect.center = (dims[0] * cell_size[0] // 2, dims[1] * cell_size[1] // 2)
    player = Player(player_rect)
    level.add_entity(player)

    # clear some space for the player
    for grid_xy in level.all_grid_coords_in_rect(player_rect.inflate(*cell_size)):
        level.set_tile(grid_xy, None)

    return level


if __name__ == "__main__":
    pygame.init()
    pygame.display.set_mode((640, 480), flags=pygame.RESIZABLE)

    clock = pygame.time.Clock()
    dt = 0
    running = True
    FPS = 60

    level = make_demo_level()
    camera = [0, 0]

    while running:
        keys_pressed_this_frame = set()

        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                running = False
            elif e.type == pygame.KEYDOWN:
                keys_pressed_this_frame.add(e.key)
                if e.key == pygame.K_r:
                    print("INFO: reset level")
                    level = make_demo_level()

        level.update_all(dt, keys_pressed_this_frame)

        screen = pygame.display.get_surface()
        screen.fill("black")

        level.draw_tiles(screen, offset=camera)
        level.draw_entities(screen, offset=camera)

        pygame.display.flip()
        dt = clock.tick(FPS) / 1000.0
        dt = min(4 / FPS, dt)  # set upper limit on dt

