# League of Legends Arena

An immitation based arena environment for League of Legends


## Installation

1. Install the dependencies for Windows.

> pip install keyboard
> pip install pytesseract

2. Install [OpenAI Gym](https://github.com/openai/gym) and its dependencies.
 ```
pip install gym
```

3. Download and install `gym-lol`:

 ```
git clone https://github.com/wau/gym-lol.git
cd gym-lol
python setup.py install
```

## Settings

The gym will copy the resolution of the game unless otherwise specified. It is recommended that you run the game at the lowest possible resolution (1280x768).