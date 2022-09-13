import math

import pygame
import typing
import random
import heapq

# TODO [ ] camera that tracks player smoothly
# TODO [ ] items that can be picked up
# TODO [ ] bullets from player
# TODO [ ] menus...?
# TODO [ ] enemies
# TODO [ ] death
# TODO [ ] finishing level -> generating next one


class Utils:

    @staticmethod
    def bound(val: float, lower: float, higher: float):
        if val < lower:
            return lower
        elif val > higher:
            return higher
        else:
            return val


class Inputs:
    """A singleton class that keeps track of player inputs.
        The idea is that the game loop keeps these sets up to date as it processes
        the event queue, and then any places in the codebase that care about inputs
        can query them here.
    """
    KEYS_PRESSED_THIS_FRAME = set()
    KEYS_RELEASED_THIS_FRAME = set()
    KEYS_HELD = set()

    @staticmethod
    def prepare_for_next_frame():
        """This should only be called from the game loop."""
        Inputs.KEYS_RELEASED_THIS_FRAME.clear()
        Inputs.KEYS_PRESSED_THIS_FRAME.clear()

    @staticmethod
    def handle_key_down(key_id):
        """This should only be called from the game loop."""
        Inputs.KEYS_PRESSED_THIS_FRAME.add(key_id)
        Inputs.KEYS_HELD.add(key_id)

    @staticmethod
    def handle_key_up(key_id):
        """This should only be called from the game loop."""
        Inputs.KEYS_RELEASED_THIS_FRAME.add(key_id)
        if key_id in Inputs.KEYS_HELD:
            Inputs.KEYS_HELD.remove(key_id)

    @staticmethod
    def is_held(key_ids: typing.Union[int, typing.Iterable[int]]):
        """Queries whether a key or any single key from a collection of keys is currently held down."""
        if isinstance(key_ids, int):
            return key_ids in Inputs.KEYS_HELD
        else:
            return any(k in Inputs.KEYS_HELD for k in key_ids)

    @staticmethod
    def was_pressed_this_frame(key_ids: typing.Union[int, typing.Iterable[int]]):
        """Queries whether a key or any single key from a collection of keys was pressed this frame."""
        if isinstance(key_ids, int):
            return key_ids in Inputs.KEYS_PRESSED_THIS_FRAME
        else:
            return any(k in Inputs.KEYS_PRESSED_THIS_FRAME for k in key_ids)


class Entity:
    """An object in the level."""

    def __init__(self, rect: pygame.Rect, solid=True, color="red"):
        self.size = rect.size
        self.solid = solid
        self.color = color

        self.pos = pygame.Vector2(rect.x, rect.y)
        self.vel = pygame.Vector2(0, 0)

        self.max_vel = (float('inf'), float('inf'))  # pixels per second
        self.gravity = 0                             # pixels per second^2

        # status variables
        self.dead = False
        self.is_grounded = False

        self.level = None

    def get_rect(self) -> pygame.Rect:
        """Returns an integer rectangle representing this entity's position and size."""
        return pygame.Rect(self.pos.x, self.pos.y, self.size[0], self.size[1])

    def handle_collision(self, others: typing.Collection['Entity']):
        pass

    def handle_death(self):
        self.remove_self_from_level()

    def remove_self_from_level(self):
        if self.level is not None:
            self.level.remove_entity(self)

    def update(self, dt: float):
        self.update_physics(dt)

    def update_physics(self, dt: float):
        # apply gravity
        self.vel.y += dt * self.gravity

        # restrict velocity to its acceptable range
        self.vel.x = Utils.bound(self.vel.x, -self.max_vel[0], self.max_vel[0])
        self.vel.y = Utils.bound(self.vel.y, -self.max_vel[1], self.max_vel[1])

        # apply velocity
        self.pos += dt * self.vel

    def draw(self, surface: pygame.Surface, offset=(0, 0)):
        pygame.draw.rect(surface, self.color, self.get_rect().move(*offset))

    def __repr__(self):
        return f"{type(self).__name__}({self.get_rect()})"


class Player(Entity):

    JUMP_KEYS = (pygame.K_SPACE, pygame.K_w, pygame.K_UP)
    MOVE_LEFT_KEYS = (pygame.K_LEFT, pygame.K_a)
    MOVE_RIGHT_KEYS = (pygame.K_RIGHT, pygame.K_d)

    def __init__(self, rect: pygame.Rect, color="blue"):
        super().__init__(rect, color=color)

        # these values feel good when cells are 32x32
        self.jump_height = 80       # pixels
        self.max_vel = (112, 512)   # pixels / sec
        self.gravity = 480          # pixels / sec^2

        # how much faster the player should fall when jump key is released
        self.fastfall_gravity_multiplier = 1.25

    def calc_jump_speed(self):
        """returns: the player's initial speed when it jumps."""
        return math.sqrt(2 * self.gravity * self.jump_height)

    def update(self, dt: float):
        # handle jumping
        if self.is_grounded and Inputs.was_pressed_this_frame(Player.JUMP_KEYS):
            self.vel.y = -self.calc_jump_speed()

        # handle walking and drifting horizontally
        walk_dir = 0
        if Inputs.is_held(Player.MOVE_LEFT_KEYS):
            walk_dir -= 1
        if Inputs.is_held(Player.MOVE_RIGHT_KEYS):
            walk_dir += 1
        self.vel.x = walk_dir * self.max_vel[0]

        super().update(dt)

    def update_physics(self, dt: float):
        # if we're moving upward but not holding jump, add some extra gravity.
        # this gives the player more control over the jump's height.
        if self.vel.y < 0 and not Inputs.is_held(Player.JUMP_KEYS):
            self.vel.y += self.gravity * dt * self.fastfall_gravity_multiplier

        super().update_physics(dt)


class Tile:
    """A cell-based piece of level geometry."""

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

    def __init__(self, cell_size=(32, 32)):
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
        self.remove_entities([entity])

    def remove_entities(self, entity_list: typing.List[Entity]):
        ids_to_remove = set()
        for entity in entity_list:
            entity.level = None
            ids_to_remove.add(id(entity))
        self.entities = [entity for entity in self.entities if id(entity) not in ids_to_remove]

    def get_player(self) -> typing.Optional[Player]:
        for entity in self.entities:
            if isinstance(entity, Player):
                return entity
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

    def draw_all(self, surface: pygame.Surface, offset=(0, 0)):
        self.draw_tiles(surface, offset=offset)
        self.draw_entities(surface, offset=offset)

    def draw_tiles(self, surface: pygame.Surface, offset=(0, 0)):
        area_to_draw = surface.get_rect(topleft=offset)
        for grid_xy in self.all_grid_coords_in_rect(area_to_draw):
            if grid_xy in self.tiles:
                self.tiles[grid_xy].draw(surface, grid_xy, self.cell_size, offset=offset)

    def draw_entities(self, surface: pygame.Surface, offset=(0, 0)):
        area_to_draw = surface.get_rect(topleft=offset)
        for entity in self.all_entities_in_rect(area_to_draw):
            entity.draw(surface, offset=offset)

    def update_all(self, dt: float):
        for entity in self.entities:
            entity.update(dt)
        self.resolve_collisions(max_displacement=max(self.cell_size))

    def is_overlapping_solid_tile(self, rect):
        for grid_xy in self.all_grid_coords_in_rect(rect):
            if grid_xy in self.tiles and self.tiles[grid_xy].solid:
                return True
        return False

    def resolve_collisions(self, max_displacement=float('inf')):
        """Resolves entity-tile collisions and entity-entity overlaps.
            This is where the magic happens.
        """
        # handling entity-to-tile collisions
        # this logic is roughly O(N) w.r.t the number of entities in the level.
        for entity in self.entities:
            if entity.dead or not entity.solid:
                # dead and non-solid entities don't collide with tiles.
                continue

            rect = entity.get_rect()
            if not self.is_overlapping_solid_tile(rect):
                # it's not colliding with anything, easy
                best_shift = (0, 0)
            else:
                # entity is colliding with the tile grid. so try to find the nearest valid position
                # it can be shifted to (within max_displacement). if no position exists, the entity
                # is marked dead.
                best_shift = None
                candidate_shifts = [(0, 0, 0)]  # items are (total_distance, x_shift_dist, y_shift_dist)
                seen_shifts = set(candidate_shifts)

                while len(candidate_shifts) > 0:
                    # the purpose of the heapq stuff is to ensure we're always checking
                    # the candidate position that's closest to the original position.
                    cur_shift = heapq.heappop(candidate_shifts)
                    new_rect = rect.move(*cur_shift[1:])

                    if not self.is_overlapping_solid_tile(new_rect):
                        # we found a valid position
                        best_shift = cur_shift[1:]
                        break
                    else:
                        # add neighbors of the position we just checked to the queue
                        for xy in ((0, -1), (-1, 0), (1, 0), (0, 1)):
                            next_shift_x = cur_shift[1] + xy[0]
                            next_shift_y = cur_shift[2] + xy[1]
                            next_shift = ((next_shift_x ** 2 + next_shift_y ** 2), next_shift_x, next_shift_y)
                            if next_shift not in seen_shifts and next_shift[0] <= max_displacement ** 2:
                                seen_shifts.add(next_shift)
                                heapq.heappush(candidate_shifts, next_shift)

            if best_shift is None:
                # we couldn't find a valid position; entity got crushed!
                entity.dead = True
            else:
                # shift the entity to the position we found, and update its velocity if necessary.
                # (e.g. if it was moving down, and got shifted up, that means it collided from below,
                # so set its y-velocity to 0).
                if best_shift[0] != 0:
                    entity.pos.x = rect.x + best_shift[0]
                    if entity.vel.x * best_shift[0] < 0:
                        entity.vel.x = 0
                if best_shift[1] != 0:
                    if entity.vel.y * best_shift[1] < 0:
                        entity.vel.y = 0
                    entity.pos.y = rect.y + best_shift[1]

            entity.is_grounded = self.is_overlapping_solid_tile(rect.move(0, 1))

        # handling entity-to-entity overlaps.
        # this logic is roughly O(N) w.r.t to the number of entities in the level.
        entity_grid = {}
        for entity in self.entities:
            if not entity.dead:
                for grid_xy in self.all_grid_coords_in_rect(entity.get_rect()):
                    if grid_xy not in entity_grid:
                        entity_grid[grid_xy] = []
                    entity_grid[grid_xy].append(entity)

        for entity in self.entities:
            if not entity.dead:
                collided_with = {}
                for grid_xy in self.all_grid_coords_in_rect(entity.get_rect()):
                    if grid_xy in entity_grid:
                        for other in entity_grid[grid_xy]:
                            if other is not entity and id(other) not in collided_with and not other.dead:
                                if entity.get_rect().colliderect(other.get_rect()):
                                    # found an overlap!
                                    collided_with[id(other)] = other
                if len(collided_with) > 0:
                    entity.handle_collision(collided_with.values())

        # remove dead entities
        dead_entities = list(filter(lambda ent: ent.dead, self.entities))
        for entity in dead_entities:
            entity.handle_death()
        self.remove_entities(dead_entities)


def make_demo_level(dims=(20, 15), density=0.2, cell_size=(32, 32)):
    level = Level(cell_size=cell_size)
    colors = ["darkslategray1", "darkslategray2", "darkslategray3"]
    tiles = [Tile(solid=True, color=c) for c in colors]

    for x in range(dims[0]):
        for y in range(dims[1]):
            if random.random() < density:
                level.set_tile((x, y), random.choice(tiles))

    player_rect = pygame.Rect(0, 0, int(cell_size[0] * 0.5), int(cell_size[1] * 0.8))
    player_rect.center = (dims[0] * cell_size[0] // 2, dims[1] * cell_size[1] // 2)
    player = Player(player_rect)

    level.add_entity(player)

    # clear some space for the player
    for grid_xy in level.all_grid_coords_in_rect(player_rect.inflate(*cell_size)):
        level.set_tile(grid_xy, None)

    return level


def draw_translucent_rect(surface: pygame.Surface, color, rect: pygame.Rect):
    new_surf = pygame.Surface(rect.size)
    new_surf.fill(color)
    if len(color) >= 4:
        new_surf.set_alpha(color[3])
    surface.blit(new_surf, rect.topleft)


def draw_translucent_circle(surface, color, center, radius, width=0):
    new_surf = pygame.Surface((int(radius * 2), int(radius * 2)))

    # if shape's color isn't black, use black as the colorkey (otherwise use white)
    if color[0:3] != (0, 0, 0):
        new_surf.set_colorkey((0, 0, 0))
    else:
        new_surf.fill((255, 255, 255))
        new_surf.set_colorkey((255, 255, 255))

    pygame.draw.circle(new_surf, color, (radius, radius), radius, width=width)
    if len(color) >= 4:
        new_surf.set_alpha(color[3])
    surface.blit(new_surf, (center[0] - radius, center[1] - radius))


if __name__ == "__main__":
    pygame.init()
    pygame.display.set_mode((640, 480), flags=pygame.RESIZABLE)

    clock = pygame.time.Clock()
    FPS = 60

    level = make_demo_level()
    camera = [0, 0]

    dt = 0
    running = True

    while running:
        # input handling
        Inputs.prepare_for_next_frame()
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                running = False
            elif e.type == pygame.KEYDOWN:
                Inputs.handle_key_down(e.key)
            elif e.type == pygame.KEYUP:
                Inputs.handle_key_up(e.key)

            # debug keybinds
            if e.type == pygame.KEYDOWN:
                if e.key == pygame.K_r:
                    # press 'R' to reset level
                    level = make_demo_level()

        # game logic
        level.update_all(dt)

        # rendering logic
        screen = pygame.display.get_surface()
        screen.fill("black")
        level.draw_all(screen, offset=camera)

        draw_translucent_rect(screen, (125, 255, 100, 75), pygame.Rect(100, 200, 300, 200))
        draw_translucent_circle(screen, (0, 0, 0, 120), (450, 200), 75)
        pygame.display.flip()

        pygame.display.set_caption(f"Platformer Demo (FPS={clock.get_fps():.1f})")
        dt = clock.tick(FPS) / 1000.0
        dt = min(4 / FPS, dt)  # set upper limit on dt

