### 1.1 环境

./envs/tpycolab/tenv.py

具体的环境构建是在key_to_door.py中，这里的pycolab 是一种通用的构建方法，我们可以尝试自己根据自己的需求来构建环境.

我们的目标是找到一个能够验证图结构方法适用的环境。

./envs/tpycolab/
需要一个新的环境来验证通用性，参照tvt中的另外一个叫active_visual_match

./envs/tAtari
尝试使用封装的Atari游戏的环境，参照baseline完成，验证有效性