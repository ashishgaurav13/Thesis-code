#!/bin/bash

echo $1

# python3 -B fs_without_icl.py -env gridworld -seed $1 -save_dir save/gridworld/$1
python3 -B fs_with_icl.py -env gridworld -seed $1 -save_dir save/gridworld/$1
# python3 -B fs_without_icl.py -env cartpole -seed $1 -save_dir save/cartpole/$1
python3 -B fs_with_icl.py -env cartpole -seed $1 -save_dir save/cartpole/$1
# python3 -B fs_without_icl.py -env highd -reduced -seed $1 -save_dir save/highds/$1
python3 -B fs_with_icl.py -env highd -reduced -seed $1 -save_dir save/highds/$1
# python3 -B fs_without_icl.py -env highd -seed $1 -save_dir save/highd/$1
# python3 -B fs_with_icl.py -env highd -seed $1 -save_dir save/highd/$1
# python3 -B fs_without_icl.py -env ant -seed $1 -save_dir save/ant/$1
# python3 -B fs_with_icl.py -env ant -seed $1 -save_dir save/ant/$1
# python3 -B fs_without_icl.py -env hc -seed $1 -save_dir save/hc/$1
# python3 -B fs_with_icl.py -env hc -seed $1 -save_dir save/hc/$1