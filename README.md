# A collection of pygame experiments and utilities.

Most of these require additional libraires. Use this to install every dependency, or just install the ones you need based on the imports at the top of the file you want to use.
```
pip install -r requirements.txt
``` 

# video (video.py)
A video playback utility for pygame using numpy and cv2.

![video_demo.gif](screenshots/n=video_demo.gif?raw=true "Video Demo")

See documentation in class for usage instructions.

# life (life.py)
An efficient game of life simulation using pygame and numpy.

![life_demo.gif](screenshots/n=life_demo.gif?raw=true "Life Demo")

Run with `python life.py` 

# double-pendulum (pendulum.py)
An efficient double pendulum simulation using pygame, numpy, and OpenGL.

### Demo (N=1000)
![n=1000animated.gif](screenshots/n=1000animated.gif?raw=true "n=1000 animated")

### Instructions
After installing the dependencies, use this to run the default program: <br>
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
