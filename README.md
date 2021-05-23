# double-pendulum
An efficient double pendulum simulation using pygame, numpy, and OpenGL.

### Demo (N=1000)
![n=1000animated.gif](screenshots/n=1000animated.gif?raw=true "n=1000 animated")

### Instructions
Use the following to install dependencies and run the default program: <br>
```
pip install -r requirements.txt
python main.py
``` 

To see the program's optional arguments, use:
```
python main.py -h
```

A command like this can be used to make a "realistic-looking" 3-pendulum simulation:
```
python main.py -n 3 --opacity 1.0 --size 400 400 --length 5 --zoom 20 --spread 1.3
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
