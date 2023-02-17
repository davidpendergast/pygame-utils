# A collection of pygame experiments and utilities.

All of these are free to use & modify, with or without attribution. Every top-level python file is a standalone project. Most of these require additional libraires. Use this to install every dependency, or just install the ones you need based on the imports at the top of the file you want to use.
```
pip install -r requirements.txt
```

# life ([life.py](life.py))
An efficient game of life simulation using pygame and numpy.

![life_demo.gif](screenshots/life_demo.gif?raw=true "Life Demo")

After installing dependencies, run with `python life.py` 

# colorswirl ([colorswirl.py](colorswirl.py))
A colorful celluar automata effect using pygame and numpy.

![colorswirl.gif](screenshots/colorswirl.gif?raw=true "Colorswirl Demo")

After installing dependencies, run with `python colorswirl.py` 

# fractal ([fractal.py](fractal.py))
An implementation of the mandlebrot set, which lets you zoom in and out.

![fractal_zoom.gif](screenshots/fractal_zoom.gif?raw=true "Fractal Demo")

After installing dependencies, run with `python fractal.py` 

# rainbowize ([rainbowize.py](rainbowize.py))
A function that applies a "rainbow effect" to a single surface.

![rainbowize.gif](screenshots/rainbowize.gif?raw=true "Rainbowize Demo")

After installing dependencies, run with `python rainbowize.py` to see a demo. 

Or import it into your own project and call `rainbowize.apply_rainbow(my_surface)`.

This program also demonstrates how to set up a `pygame.SCALED` display with a custom initial scaling factor and outer fill color (see function `make_fancy_scaled_display`).

# lut ([lut.py](lut.py))
A function that transforms the colors of a surface using a lookup table (aka a "LUT").

![lut.gif](screenshots/lut.gif?raw=true "LUT Demo")

After installing dependencies, run with `python lut.py` to see a demo.

Or import it into your own project and call `lut.apply_lut(source_surface, lut_surface, idx)`.

If `numpy` isn't available, the function will fall back to a pure pygame routine (which is slower but produces the same result). The function also has an optional built-in caching system, and handles per-pixel alpha in a reasonable way.

# video ([video.py](video.py))
A video playback utility for pygame using numpy and cv2.

![video_demo.gif](screenshots/video_demo.gif?raw=true "Video Demo")

See documentation in class for usage instructions.

# steganography ([steganography.py](steganography.py))
A module that writes text data into the pixel values of images.

![stega_demo.png](screenshots/stega_demo.png?raw=true "Steganography Demo")

After installing dependencies, import the module into your project and call its methods. See docstrings for detailed usage instructions and information about advanced settings (e.g. bit depth, image resizing).

### Example of writing a message into a surface.
```
output_surface = steganography.write_text_to_surface("secret message", my_surface)
message = steganography.read_text_from_surface(output_surface)
print(message)  # prints "secret message"
```

### Example of saving a message to a PNG.
```
steganography.save_text_as_image_file("secret message", my_surface, "path/to/file.png")
message = steganography.load_text_from_image_file("path/to/file.png")
print(message)  # prints "secret message"
```

# gifreader ([gifreader.py](gifreader.py))
A utility function that loads a gif file's images and metadata. 

Metadata includes things like frame durations and image dimensions. See documentation on the function for usage instructions.

Requires `imageio` (`pip install imageio`) and `numpy`

### Example of loading and displaying a GIF
![gifreader_demo.gif](screenshots/gifreader_demo.gif?raw=true "gifreader_demo")

# benchmark ([benchmark.py](benchmark.py))
A program that benchmarks pygame's rendering and displays the results in a graph.

Requires `matplotlib` (`pip install matplotlib`).

### Instructions
Use `python benchmark.py` to run the tests with default settings, or `python benchmark.py --help` to see all settings. 

If you close the window while the test is running, the results from cases that have already completed will still be shown.

### Sample Results
![benchmark_results.png](screenshots/benchmark_results.png?raw=true "benchmark_results")

A plot that shows the relationship between FPS and the number of entities being rendered. Each line is a separate test case.
```
ALL           = Entities of all types rendered together
SURF_RGB      = Surfaces with no translucency
SURF_RGBA     = Surfaces with per-pixel translucency
SURF_RGB_WITH_ALPHA = Surfaces with full-surface translucency
RECT_FILLED   = pygame.draw.rect with width = 0 (i.e. filled)
CIRCLE_FILLED = pygame.draw.circle with width = 0 (i.e. filled)
LINE          = pygame.draw.line
RECT_HOLLOW   = pygame.draw.rect with width > 0
CIRCLE_HOLLOW = pygame.draw.circle with width > 0
```
Note that non-rectangular entities are scaled up and/or assigned widths so that drawing them will roughly change the same number of pixels as blitting a surface. This seemed like the most sensible way to compare rendering speeds.

# double-pendulum ([pendulum.py](pendulum.py))
An efficient double pendulum simulation using pygame, numpy, and OpenGL.

### Demo (N=1000)
![n=1000animated.gif](screenshots/n=1000animated.gif?raw=true "n=1000 animated")

### Instructions
After installing dependencies, use this to run the default program: <br>
```
python pendulum.py
``` 

To see the program's optional arguments, use:
```
python pendulum.py -h
```

A command like this can be used to make a "realistic-looking" 3-pendulum simulation:
```
python pendulum.py -n 3 --opacity 1.0 --size 400 400 --length 5 --zoom 20 --spread 1.3
```

While the simulation is running, the following actions can be used via keybindings: <br>
* \[r\] to restart the simlution with a new starting angle <br>
* \[Esc\] to quit the program <br>
* \[p\] to enable / disable profiling <br>

### N=1000
![n=1000.PNG](screenshots/n=1000.PNG?raw=true "n=1000")

### N=10,000
![n=10000.PNG](screenshots/n=10000.PNG?raw=true "n=10000")

### N=100,000
![n=100000.PNG](screenshots/n=100000.PNG?raw=true "n=100000")
