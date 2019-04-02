# ** TO BUILD (windows) **
1. run cmake-gui and set 64 bit msvc compiler
2. cmake CMakeCache.txt -DCMAKE_WINDOWS_EXPORT_ALL_SYMBOLS=TRUE 
3. copy build/Release/interf.dll to libs/

# ** TO INSTALL **
pip3 install -e gym-interf


# ** TO PLAY **
```python
import gym
import gym_interf
from gym.utils.play import play

env = gym.make('interf-v0')
play(env=env, zoom=2, fps=30)

```



