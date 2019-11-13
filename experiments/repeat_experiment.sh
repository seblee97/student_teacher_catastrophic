# kill sessions already running
tmux kill-session -t 0
tmux kill-session -t 10
tmux kill-session -t 15
tmux kill-session -t 20
tmux kill-session -t 25

tmux kill-session -t tensorboard

# start tmux sessions one for each experiment
tmux new -s 0 -d
tmux new -s 10 -d
tmux new -s 15 -d
tmux new -s 20 -d
tmux new -s 25 -d

# initialise virtual environments
tmux send-keys -t 0 "source ~/envs/cata/bin/activate" C-m
tmux send-keys -t 10 "source ~/envs/cata/bin/activate" C-m
tmux send-keys -t 15 "source ~/envs/cata/bin/activate" C-m
tmux send-keys -t 20 "source ~/envs/cata/bin/activate" C-m
tmux send-keys -t 25 "source ~/envs/cata/bin/activate" C-m

# send keys to run experiments
tmux send-keys -t 0 "python main.py --lc continual --tc overlapping --sc threshold --ts 100000 --v 0 --nl relu --to '[0, 0]' -s 0" C-m
tmux send-keys -t 10 "python main.py --lc continual --tc overlapping --sc threshold --ts 100000 --v 0 --nl relu --to '[0, 0]' -s 10" C-m
tmux send-keys -t 15 "python main.py --lc continual --tc overlapping --sc threshold --ts 100000 --v 0 --nl relu --to '[0, 0]' -s 15" C-m
tmux send-keys -t 20 "python main.py --lc continual --tc overlapping --sc threshold --ts 100000 --v 0 --nl relu --to '[0, 0]' -s 20" C-m
tmux send-keys -t 25 "python main.py --lc continual --tc overlapping --sc threshold --ts 100000 --v 0 --nl relu --to '[0, 0]' -s 25" C-m

# start tmux session for tensorboard, launch tensorboard
tmux new -s tensorboard -d
tmux send-keys -t "tensorboard" "source ~/envs/cata/bin/activate" C-m
tmux send-keys -t "tensorboard" "tensorboard --logdir results/" C-m
