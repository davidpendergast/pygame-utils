import pygame
import imageio.v3 as iio
import typing

import numpy


def load_gif(filename) -> typing.Tuple[typing.List[pygame.Surface],
                                       typing.Dict[str, typing.Any]]:
    """Loads the image data and metadata from a gif file.

        Args:
            filename: The gif file to load.

        Returns:
            A tuple containing two items:
                output[0]: A list of Surfaces corresponding to the frames of the gif.
                output[1]: A dict containing metadata about the gif. The keys are strings and will always be present:
                    'file': (str) The file the gif was loaded from (matches the filename input).
                    'width': (int) The width of the image frames in pixels.
                    'height': (int) The height of the image frames in pixels.
                    'dims': (int, int) The dimensions of the image frames (width, height) in pixels.
                    'length': (int) The number of frames in the gif.
                    'duration': (int) The duration of each frame in the gif, in milliseconds.
    """
    # read gif metadata
    meta = iio.immeta(f"{filename}", extension=".gif", index=None)

    # ndarray with (num_frames, height, width, channel)
    gif = iio.imread(f"{filename}", extension=".gif", index=None)
    gif = numpy.transpose(gif, axes=(0, 2, 1, 3))  # flip x and y axes

    return [pygame.surfarray.make_surface(gif[i]) for i in range(gif.shape[0])], {
        'file': filename,
        'width': gif.shape[1],
        'height': gif.shape[2],
        'size': (gif.shape[1], gif.shape[2]),
        'length': gif.shape[0],
        'duration': int(meta['duration']) if 'duration' in meta else 0,
        'loop': int(meta['loop']) if 'loop' in meta else 0
    }


if __name__ == "__main__":
    FILE = "data/MOSHED-doctor.gif"  # your gif goes here
    TEXT_SIZE = 32

    pygame.init()
    pygame.display.set_mode((640, 480), pygame.RESIZABLE)
    pygame.display.set_caption("gifreader.py")

    frames, metadata = load_gif(FILE)
    frm_duration = max(metadata['duration'], 1)

    font = pygame.font.Font(None, TEXT_SIZE)

    clock = pygame.time.Clock()
    elapsed_time = 0

    while True:
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                raise SystemExit()

        screen = pygame.display.get_surface()
        screen.fill((0, 0, 0))
        screen_size = screen.get_size()

        idx = (elapsed_time // frm_duration) % len(frames)

        offs = (screen_size[0] // 2 - frames[idx].get_width() // 2,
                screen_size[1] // 2 - frames[idx].get_height() // 2)
        screen.blit(frames[idx], offs)

        if TEXT_SIZE > 0:  # render metadata
            x, y = 0, 0
            for key in metadata:
                screen.blit(font.render(f"{key}: {metadata[key]}", True, (255, 255, 255), (0, 0, 0, 0)), (x, y))
                y += font.get_height()
            screen.blit(font.render(f"frame: {idx}", True, (255, 255, 255), (0, 0, 0, 0)), (x, y))
            y += font.get_height()

        pygame.display.flip()
        elapsed_time += clock.tick(60)
