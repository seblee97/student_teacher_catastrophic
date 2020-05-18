# kill sessions already running
tmux kill-session -t run_1
tmux kill-session -t run_2
tmux kill-session -t run_3
tmux kill-session -t run_4
tmux kill-session -t run_5

# start tmux sessions one for each experiment
tmux new -s run_1 -d
tmux new -s run_2 -d
tmux new -s run_3 -d
tmux new -s run_4 -d
tmux new -s run_5 -d

# initialise virtual environments
tmux send-keys -t run_1 "source ~/envs/cata/bin/activate" C-m
tmux send-keys -t run_2 "source ~/envs/cata/bin/activate" C-m
tmux send-keys -t run_3 "source ~/envs/cata/bin/activate" C-m
tmux send-keys -t run_4 "source ~/envs/cata/bin/activate" C-m
tmux send-keys -t run_5 "source ~/envs/cata/bin/activate" C-m

# send keys to run experiments
tmux send-keys -t run_1 "python main.py --en run1 --to '[25, 0]'" C-m
tmux send-keys -t run_2 "python main.py --en run2 --to '[50, 0]'" C-m
tmux send-keys -t run_3 "python main.py --en run3 --to '[75, 0]'" C-m
tmux send-keys -t run_4 "python main.py --en run4 --to '[100, 0]'" C-m

# start tmux session for tensorboard, launch tensorboard
# tmux new -s tensorboard -d
# tmux send-keys -t "tensorboard" "source ~/envs/cata/bin/activate" C-m
# tmux send-keys -t "tensorboard" "tensorboard --logdir results/" C-m