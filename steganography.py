import math
import random

import pygame


def write_data_to_image(dest: pygame.Surface, ascii_data: str,
                        bit_depth=2, allow_resize=True,
                        append_noise=True) -> pygame.Surface:
    if chr(0) in ascii_data:
        raise ValueError("cannot store data containing NULL bytes.")
    if bit_depth not in (1, 2, 4, 8):
        raise ValueError(f"bit_depth must be 1, 2, 4, or 8, instead got: {bit_depth}")
    total_bits = len(ascii_data) * 8  # total bits to store

    pixels_needed = math.ceil(total_bits / (3 * 8 * bit_depth)) + 1
    if pixels_needed > dest.get_width() * dest.get_height():
        if allow_resize:
            mult = int(pixels_needed / (dest.get_width() * dest.get_height()))
            surf = pygame.transform.scale(dest, (dest.get_width() * mult, dest.get_height() * mult))
        else:
            raise ValueError(f"dest doesn't have enough pixels ({dest.get_width() * dest.get_height()}) to fit "
                             f"this data ({pixels_needed}). ")
    else:
        surf = dest.copy()

    # write header data
    r, g, b, *_ = surf.get_at((0, 0))
    r = set_bit(r, 0, (bit_depth // 4) % 2)
    g = set_bit(g, 0, (bit_depth // 2) % 2)
    b = set_bit(b, 0, (bit_depth // 1) % 2)
    surf.set_at((0, 0), (r, g, b))

    for bit_idx_in_data in range((surf.get_width() * surf.get_height() - 1) * 3 * bit_depth):
        if bit_idx_in_data < len(ascii_data) * 8:
            # writing actual data
            char_data = ord(ascii_data[bit_idx_in_data // 8])
        elif len(ascii_data) * 8 <= bit_idx_in_data < (len(ascii_data) + 1) * 8:
            char_data = 0  # indicates the end of the data stream
        elif append_noise:
            # add some random bits from the data stream
            char_data = ord(random.choice(ascii_data))
        else:
            break

        bit_val_to_write = get_bit(char_data, bit_idx_in_data % 8)
        px_idx = 1 + bit_idx_in_data // (bit_depth * 3)
        px_xy = (px_idx % surf.get_width(), px_idx // surf.get_width())
        color_at_px = list(surf.get_at(px_xy))

        color_channel_idx = (bit_idx_in_data // bit_depth) % 3
        bit_idx_in_color = bit_idx_in_data % bit_depth
        color_at_px[color_channel_idx] = set_bit(color_at_px[color_channel_idx], bit_idx_in_color, bit_val_to_write)
        surf.set_at(px_xy, color_at_px)

    return surf


def read_data_from_image(surf: pygame.Surface) -> str:
    # read header data
    r, g, b, *_ = surf.get_at((0, 0))
    bit_depth = get_bit(r, 0) * 4 + get_bit(g, 0) * 2 + get_bit(b, 0)
    print("bit_depth =", bit_depth)

    data = []
    cur_word = 0

    for bit_idx_in_data in range((surf.get_width() * surf.get_height() - 1) * 3 * bit_depth):
        px_idx = 1 + bit_idx_in_data // (bit_depth * 3)
        px_xy = (px_idx % surf.get_width(), px_idx // surf.get_width())
        color_at_px = list(surf.get_at(px_xy))

        color_channel_idx = (bit_idx_in_data // bit_depth) % 3
        bit_idx_in_color = bit_idx_in_data % bit_depth
        bit_val = get_bit(color_at_px[color_channel_idx], bit_idx_in_color)
        cur_word = set_bit(cur_word, bit_idx_in_data % 8, bit_val)
        if bit_idx_in_data % 8 == 7:
            if cur_word == 0:
                break  # found the terminal character
            else:
                data.append(chr(cur_word))
                cur_word = 0

    return "".join(data)


# yoinked from https://stackoverflow.com/questions/12173774/how-to-modify-bits-in-an-integer
def set_bit(v, index, x) -> int:
    """Set the index:th bit of v to 1 if x is truthy, else to 0, and return the new value."""
    mask = 1 << index   # Compute mask, an integer with just bit 'index' set.
    v &= ~mask          # Clear the bit indicated by the mask (if x is False)
    if x:
        v |= mask       # If x was True, set the bit indicated by the mask.
    return v            # Return the result, we're done.


def get_bit(v, index):
    mask = 1 << index
    v &= mask
    return v >> index


if __name__ == "__main__":
    img = pygame.image.load("data/splash.png")

    import json
    with open("data/arrival.json") as f:
        input_data_as_json = json.load(f)

    input_data = json.dumps(input_data_as_json, ensure_ascii=True)

    new_surf = write_data_to_image(img, input_data, bit_depth=4)

    output_data = read_data_from_image(new_surf)
    print("input_data: ", input_data)
    print("output_data:", output_data)

    pygame.image.save(new_surf, "data/splash_output.png")


