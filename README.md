# A collection of pygame experiments and utilities.

All of these are free to use & modify, with or without attribution. Every top-level python file is a standalone project. Most of these require additional libraires. Use this to install every dependency, or just install the ones you need based on the imports at the top of the file you want to use.
```
pip install -r requirements.txt
```

# life ([life.py](life.py))
An efficient game of life simulation using pygame and numpy.

![life_demo.gif](screenshots/life_demo.gif?raw=true "Life Demo")

After installing dependencies, run with `python life.py` 

# colorswirl([colorswirl.py](colorswirl.py))
A colorful celluar automata effect using pygame and numpy.

![colorswirl.gif](screenshots/colorswirl.gif?raw=true "Colorswirl Demo")

After installing dependencies, run with `python colorswirl.py` 

# fractal ([fractal.py](fractal.py))
An implementation of the mandlebrot set, which lets you zoom in and out.

![fractal_zoom.gif](screenshots/fractal_zoom.gif?raw=true "Fractal Demo")

After installing dependencies, run with `python fractal.py` 

# rainbowize ([rainbowize.py](rainbowize.py))
A function that applies a "rainbow effect" to a single pygame Surface.

![rainbowize.gif](screenshots/rainbowize.gif?raw=true "Rainbowize Demo")

After installing dependencies, run with `python rainbowize.py` to see a demo. 

Or import it into your own project and call `rainbowize.apply(my_surface, i)`.

# video ([video.py](video.py))
A video playback utility for pygame using numpy and cv2.

![video_demo.gif](screenshots/video_demo.gif?raw=true "Video Demo")

See documentation in class for usage instructions.

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
