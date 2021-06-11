import pygame
import numpy
import cv2

import math
import time
import os
import errno

class Video:
    """A video playback utility for pygame.

    Requires:
        pygame  (pip install pygame)
        numpy   (pip install numpy)
        cv2     (pip install opencv-python)

    This class streams the image data from a video file and provides it as a pygame Surface. There are methods that
    control video playback, such as play, pause, and set_frame, as well as methods that provide information about the
    video itself, like the dimensions and the FPS.

    The video playback tries to play in "real time", meaning that if you call play, and then call get_surface 5 seconds
    later, you'll get the video frame at the 5 second mark.

    However, this class is also lazy, meaning that video data is only processed when get_surface is called. There is
    no asynchronous background processing, and it doesn't cache any video data in memory besides the current frame.
    You don't need to advance frames manually or update the video each frame to keep it "going".

    This class does not provide sound from video files, only the images.

    Example Usage:

        import pygame
        import video

        pygame.init()

        screen = pygame.display.set_mode((1080, 720))
        clock = pygame.time.Clock()

        vid = video.Video("your_file_goes_here.mp4")
        vid.play()

        while True:
            for e in pygame.event.get():
                if e.type == pygame.QUIT:
                    raise SystemExit
            screen.fill((0, 0, 0))
            screen.blit(vid.get_surface(), (0, 0))

            pygame.display.flip()
            clock.tick(60)
    """

    class _PlaybackState:
        def __init__(self, start_frame, playing, t):
            self.frame = start_frame
            self.playing = playing
            self.t = t

    def __init__(self, filename, fps=0):
        """Inits a new Video.

        The video will be paused initially.

        filename: The path to the video file.
        fps: The playback framerate for the video. If 0, the native FPS will be used.
        """
        if not os.path.isfile(filename):
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), filename)

        self._filename = filename
        self._vid = cv2.VideoCapture(filename)
        self._vid_frame = 0  # the frame the next _vid.read() call will give.

        self._frame_count = int(self._vid.get(cv2.CAP_PROP_FRAME_COUNT))
        self._frame_width = int(self._vid.get(cv2.CAP_PROP_FRAME_WIDTH))
        self._frame_height = int(self._vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self._fps = int(self._vid.get(cv2.CAP_PROP_FPS)) if fps <= 0 else fps

        self._buf = pygame.Surface((self._frame_width, self._frame_height))
        self._cur_frame = -1  # the frame that was last drawn to _buf

        self._playback_state = Video._PlaybackState(0, False, time.time())
        self._final_frame = float('inf')

    def __repr__(self):
        return f"{type(self).__name__}({self._filename})"

    def play(self, loops=0, maxtime=0):
        """Begins video playback.

        loops: How many times playback should loop. If 0, it will repeat forever.
        maxtime: How long in seconds the video should play. If 0, it will play forever.
        """
        if self.is_paused():
            self._playback_state.playing = True
            self._playback_state.t = time.time()
        self.frame = self._playback_state.frame % self.get_frame_count()  # go back to first loop
        self._final_frame = self._calc_final_frame(loops, maxtime)

    def set_frame(self, n):
        """Jumps to a specific frame."""
        self._playback_state.frame = n
        self._playback_state.t = time.time()

    def pause(self):
        """Pauses the video."""
        if not self.is_paused():
            self._playback_state.frame = self.get_current_frame(wrapped=False)
            self._playback_state.playing = False

    def is_paused(self):
        """Whether the video is paused.

        Note that if a video has finished (e.g. finished looping), it will still be considered unpaused.
        """
        return not self._playback_state.playing

    def is_finished(self) -> bool:
        """Whether the termination condition passed into play has been reached."""
        return self.get_current_frame() >= self._final_frame

    def get_surface(self) -> pygame.Surface:
        """Returns the video's image data for the current frame.

        This is where the bulk of this class's work is performed. This method calculates the current frame (based on the
        current time and other factors), and if the frame has changed since the last call to this method, it reads
        video data from the file and blits it onto a surface, which is returned.

        If the buffer surface is already up-to-date, this method returns it instantly.
        If the video has ended, (i.e. the termination condition passed into play() has been reached), a blank surface is returned.
        """
        cur_frame = self.get_current_frame(wrapped=False)
        if cur_frame >= self._final_frame:
            self._buf.fill((0, 0, 0))  # video is over, you get a black screen
            return self._buf
        else:
            self._draw_frame_to_surface_if_necessary(cur_frame)
            return self._buf

    def _draw_frame_to_surface_if_necessary(self, frame_n):
        frame_n %= self._frame_count
        if self._cur_frame == frame_n:
            return  # this frame is already drawn

        if self._vid_frame > frame_n:
            # we have to loop back to the beginning
            self._vid_frame = 0
            self._vid = cv2.VideoCapture(self._filename)

        success, frame_data = None, None
        for _ in range(frame_n - self._vid_frame + 1):
            success, next_frame_data = self._vid.read()
            if not success:
                # sometimes CAP_PROP_FRAME_COUNT will be straight-up wrong, indicating more or less frames
                # than there actually are. In that case we just... skip or freeze the final frames, I guess?
                continue
            else:
                frame_data = next_frame_data
                self._vid_frame += 1

        if frame_data is not None:
            pygame.surfarray.blit_array(self._buf, numpy.flip(numpy.rot90(frame_data[::-1])))
        self._cur_frame = self._vid_frame - 1

    def get_width(self) -> int:
        """The width of the video in pixels."""
        return self._frame_width

    def get_height(self) -> int:
        """The height of the video in pixels."""
        return self._frame_height

    def get_size(self) -> (int, int):
        """The dimensions of the video in pixels."""
        return self.get_width(), self.get_height()

    def get_current_frame(self, wrapped=True) -> int:
        """The current frame number.

        wrapped: if True, the result will be less than frame_count.
                 if False, the result may be greater or equal to frame_count, indicating the video has looped.
        """
        if self.is_paused():
            return self._playback_state.frame
        else:
            cur_time = time.time()
            start_time = self._playback_state.t
            elapsed_frames = int(self.get_fps() * (cur_time - start_time))
            cur_frame = self._playback_state.frame + elapsed_frames

            return cur_frame % self._frame_count if wrapped else cur_frame

    def get_fps(self) -> int:
        """The frames per second at which the video will play."""
        return self._fps

    def get_frame_count(self) -> int:
        """The number of frames in the video."""
        return self._frame_count

    def get_duration(self) -> float:
        """The duration of the video in seconds."""
        return self._frame_count / self._fps

    def _calc_final_frame(self, loops, maxtime):
        res = float('inf')
        if loops > 0:
            res = min(res, loops * self.get_frame_count())
        if maxtime > 0:
            res = min(res, math.ceil(self.get_fps() * maxtime))
        return res
