import math
import typing

import pygame
import numpy


def write_text_to_surface(text_data: str, input_surface: pygame.Surface,
                          bit_depth_range: typing.Union[typing.Tuple[int, int], int] = (1, 4),
                          end_str: str = chr(0),
                          resize_mode: str = 'smooth') -> pygame.Surface:
    """Writes ascii data into the pixel values of a surface (RGB only).

        This function returns a new copy of the image, leaving the original unmodified. Lower-ordered bits of each color
        channel are consumed first, to preserve the image as best as possible. A lower bit depth will give better image
        quality at the cost of worse data compression, and vice-versa. Noise is added after the end of the data section
        to avoid creating an obvious boundary.

        Args:
            text_data: The ascii data to write.
            input_surface: The surface to write the data into.
            bit_depth_range: The range of bits to use for data storage, per byte of image data. An optimal value
              will be selected from this range based on the image's size and the amount of ascii data. Bounds must be
              between 1 and 8. If a single int is provided, that bit depth will be used.
            end_str: Indicator for where the data ends. This can be any string or character that doesn't appear
              in the data string. By default, it's the NULL character (0x00000000).
            resize_mode: This parameter controls how the function behaves when the image isn't large enough
              to fit the data. Accepted values: None, "smooth", or "integer". If None, the image will not be resized,
              and the function will throw an error if the data doesn't fit. If "smooth", the image will be scaled up
              smoothly in each axis. If "integer", the image will be scaled up by an integer multiplier in each axis.

        Returns:
            A copy of the input surface with the text data encoded into its pixel values.

        Raises:
            ValueError: If any parameters are invalid, or resize_mode is None and the data is too large to be stored.
    """
    if isinstance(bit_depth_range, int):
        bit_depth_range = (bit_depth_range, bit_depth_range)
    if not (1 <= bit_depth_range[0] <= bit_depth_range[1] <= 8):
        raise ValueError(f"Illegal bit_depth_range: {bit_depth_range}")

    if end_str in text_data:
        raise ValueError(f"text_data cannot contain end_str (found at index: {text_data.index(end_str)})")
    text_data += end_str

    bytes_in_img = input_surface.get_width() * input_surface.get_height() * 3
    if bytes_in_img == 0:
        raise ValueError("Cannot write text to empty surface.")

    # find an optimal bit depth within the specified bounds.
    header_data_size = 3
    optimal_bit_depth = math.ceil(len(text_data) * 8 / (bytes_in_img - header_data_size))
    bit_depth = max(bit_depth_range[0], min(optimal_bit_depth, bit_depth_range[1]))

    data_array = _str_to_flat_array(text_data, bits_per_index=bit_depth)
    img_bytes_needed = header_data_size + data_array.size
    img_bytes_in_input = input_surface.get_width() * input_surface.get_height() * 3

    # resize the input surface (if necessary) so it's large enough to hold the data.
    if img_bytes_in_input <= img_bytes_needed:
        if resize_mode == 'integer':
            mult = math.ceil(math.sqrt(img_bytes_needed / img_bytes_in_input))
            new_dims = (input_surface.get_width() * mult,
                        input_surface.get_height() * mult)
        elif resize_mode == 'smooth':
            mult = math.sqrt(img_bytes_needed / img_bytes_in_input)
            new_dims = (math.ceil(input_surface.get_width() * mult),
                        math.ceil(input_surface.get_height() * mult))
        else:
            raise ValueError(f"The surface is too small to contain {len(text_data)} bytes of text "
                             f"with a bit_depth of {bit_depth}.")
        output_surface = pygame.transform.scale(input_surface, new_dims)
    else:
        output_surface = input_surface.copy()

    img_bytes_in_output = output_surface.get_width() * output_surface.get_height() * 3
    if img_bytes_needed < img_bytes_in_output:
        end_idx = header_data_size + data_array.size
        data_array = numpy.pad(data_array, (header_data_size, img_bytes_in_output - (data_array.size + header_data_size)),
                               'constant', constant_values=(0, 0))
        # fill the rest of the image with noise, to avoid creating a visible boundary.
        data_array[end_idx:] = numpy.random.randint(2 ** bit_depth, size=data_array.size - end_idx)

    # write 1 bit of 'header data' into the first 3 bytes of the image, to indicate the bit depth of the data section.
    first_px_rgb = list(output_surface.get_at((0, 0)))
    first_px_rgb[0] = _set_bit(first_px_rgb[0], 0, (bit_depth // 1) % 2)
    first_px_rgb[1] = _set_bit(first_px_rgb[1], 0, (bit_depth // 2) % 2)
    first_px_rgb[2] = _set_bit(first_px_rgb[2], 0, (bit_depth // 4) % 2)

    colors = [
        pygame.surfarray.pixels_red(output_surface),
        pygame.surfarray.pixels_green(output_surface),
        pygame.surfarray.pixels_blue(output_surface)
    ]

    # finally, write the actual data.
    for c in range(3):
        colors[c] &= 255 - ((1 << bit_depth) - 1)  # e.g. 0x11110000, where # of zeros = bit_depth
        colors[c] |= data_array[c::3].reshape(colors[c].shape)

    # write header data (first 3 bytes = RGB channels of the 1st pixel in the image).
    output_surface.set_at((0, 0), first_px_rgb)

    return output_surface


def read_text_from_surface(surface: pygame.Surface, end_str=chr(0)) -> str:
    """Extracts the ascii data that was written into a surface by write_text_to_surface(...).
        surface: The surface.
        end_str: Indicator for where the data ends. Must match the string that was used when writing the data.
    """
    # first, read the header data to find the bit_depth
    first_px_rgb = surface.get_at((0, 0))
    bit_depth = first_px_rgb[0] % 2 + (first_px_rgb[1] % 2) * 2 + (first_px_rgb[2] % 2) * 4
    if not (1 <= bit_depth <= 8):
        raise ValueError(f"Illegal bit_depth: {bit_depth}")
    header_data_size = 3

    colors = [
        pygame.surfarray.pixels_red(surface),
        pygame.surfarray.pixels_green(surface),
        pygame.surfarray.pixels_blue(surface)
    ]

    raw_data = numpy.array([0] * (surface.get_width() * surface.get_height() * 3), dtype="uint8")
    mask = (1 << bit_depth) - 1  # e.g. 0x00001111, where # of 1s = bit_depth
    for c in range(3):
        raw_data[c::3] = (colors[c] & mask).reshape(raw_data.size // 3)

    return _flat_array_to_str(raw_data[header_data_size:], bits_per_index=bit_depth, end_str=end_str)


def save_text_as_image_file(text_data: str, input_surface: pygame.Surface, filepath: str,
                            bit_depth_range=(1, 4), end_str=chr(0), resize_mode='smooth'):
    to_save = write_text_to_surface(text_data, input_surface,
                                    bit_depth_range=bit_depth_range,
                                    end_str=end_str,
                                    resize_mode=resize_mode)
    pygame.image.save(to_save, filepath)


def load_text_from_image_file(filepath: str, end_str=chr(0)) -> str:
    img = pygame.image.load(filepath)
    return read_text_from_surface(img, end_str=end_str)


def _str_to_flat_array(data: str, bits_per_index=2):
    """
    Example input: "abc" with bits_per_index=2
    raw_bytes: [01100001 01100010 01100011] (aka 97 98 99)
    raw_bits:  [1 0 0 0 0 1 1 0 0 1 0 0 0 1 1 0 1 1 0 0 0 1 1 0]
    res:       [01 00 10 01 10 00 10 01 11 00 10 01]
    """
    raw_bytes = numpy.array(bytearray(data, 'utf-8'), dtype='uint8')
    raw_bits = numpy.array([0] * 8 * raw_bytes.size, dtype='uint8')
    for i in range(8):
        raw_bits[i:i + 8 * raw_bytes.size:8] = (raw_bytes & (1 << i)) >> i
    if raw_bits.size % bits_per_index > 0:
        # pad end with 0s if bits_per_index doesn't cleanly divide raw_bits
        raw_bits = numpy.pad(raw_bits, (0, bits_per_index - (raw_bits.size % bits_per_index)),
                             'constant', constant_values=0)
    res = numpy.array([0] * math.ceil(raw_bits.size / bits_per_index), dtype='uint8')
    for j in range(bits_per_index):
        res |= raw_bits[j::bits_per_index] << j
    return res


def _flat_array_to_str(arr, bits_per_index=2, end_str=chr(0)) -> str:
    """Reverse of _str_to_flat_array"""
    raw_bits = numpy.array([0] * (arr.size * bits_per_index), dtype='uint8')
    for j in range(bits_per_index):
        raw_bits[j::bits_per_index] = (arr & (1 << j)) >> j

    overflow = raw_bits.size % 8
    if overflow > 0:
        raw_bits = numpy.resize(raw_bits, (raw_bits.size - overflow,))

    raw_bytes = numpy.array([0] * (raw_bits.size // 8), dtype='uint8')
    for i in range(8):
        raw_bytes |= raw_bits[i::8] << i

    as_bytes = raw_bytes.tobytes()
    if end_str.encode("utf-8") in as_bytes:
        return as_bytes[0:as_bytes.index(end_str.encode("utf-8"))].decode("utf-8")
    else:
        return as_bytes.decode("utf-8")


# yoinked from https://stackoverflow.com/questions/12173774/how-to-modify-bits-in-an-integer
def _set_bit(v, index, x) -> int:
    """Set the index:th bit of v to 1 if x is truthy, else to 0, and return the new value."""
    mask = 1 << index   # Compute mask, an integer with just bit 'index' set.
    v &= ~mask          # Clear the bit indicated by the mask (if x is False)
    if x:
        v |= mask       # If x was True, set the bit indicated by the mask.
    return v            # Return the result, we're done.


if __name__ == "__main__":
    input_filename = "data/splash.png"
    output_filename = "data/splash_output.png"
    _end_str = "~END~"
    img = pygame.image.load(input_filename)

    import json
    with open("data/arrival.json") as f:
        input_data_as_json = json.load(f)

    input_data = json.dumps(input_data_as_json, ensure_ascii=True)
    input_data = input_data * 10

    new_surf = write_text_to_surface(input_data, img, bit_depth_range=(1, 5), end_str=_end_str, resize_mode='integer')
    pygame.image.save(new_surf, output_filename)

    output_data_nosave = read_text_from_surface(new_surf, end_str=_end_str)
    output_data_from_img = load_text_from_image_file(output_filename, end_str=_end_str)

    print("input_data:", input_data)
    print("output_data_nosave:", output_data_nosave)
    print("output_data_from_img:", output_data_from_img)
    print(f"input_data == output_data_nosave = {input_data == output_data_nosave}")
    print(f"input_data == output_data_from_img = {input_data == output_data_from_img}")
