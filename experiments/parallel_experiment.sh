# kill sessions already running
tmux kill-session -t meta_independent
tmux kill-session -t meta_noisy
tmux kill-session -t meta_drifting
tmux kill-session -t meta_overlapping
tmux kill-session -t continual_independent
tmux kill-session -t continual_noisy
tmux kill-session -t continual_drifting
tmux kill-session -t continual_overlapping

# start tmux sessions one for each experiment
tmux new -s meta_independent -d
tmux new -s meta_noisy -d
tmux new -s meta_drifting -d
tmux new -s meta_overlapping -d
tmux new -s continual_independent -d
tmux new -s continual_noisy -d
tmux new -s continual_drifting -d
tmux new -s continual_overlapping -d

# initialise virtual environments
tmux send-keys -t meta_independent "source ~/envs/cata/bin/activate" C-m
tmux send-keys -t meta_noisy "source ~/envs/cata/bin/activate" C-m
tmux send-keys -t meta_drifting "source ~/envs/cata/bin/activate" C-m
tmux send-keys -t meta_overlapping "source ~/envs/cata/bin/activate" C-m
tmux send-keys -t continual_independent "source ~/envs/cata/bin/activate" C-m
tmux send-keys -t continual_noisy "source ~/envs/cata/bin/activate" C-m
tmux send-keys -t continual_drifting "source ~/envs/cata/bin/activate" C-m
tmux send-keys -t continual_overlapping "source ~/envs/cata/bin/activate" C-m

# send keys to run experiments
tmux send-keys -t meta_independent "python main.py --lc meta --tc independent" C-m
tmux send-keys -t meta_noisy "python main.py --lc meta -tc noisy" C-m
tmux send-keys -t meta_drifting "python main.py --lc meta --tc drifting" C-m
tmux send-keys -t meta_overlapping "python main.py --lc meta --tc overlapping" C-m
tmux send-keys -t continual_independent "python main.py --lc continual --tc independent" C-m
tmux send-keys -t continual_noisy "python main.py --lc continual --tc noisy" C-m
tmux send-keys -t continual_drifting "python main.py --lc continual --tc drifting" C-m
tmux send-keys -t continual_overlapping "python main.py --lc continual --tc overlapping" C-m

# start tmux session for tensorboard, launch tensorboard
tmux new -s tensorboard -d
tmux send-keys -t "tensorboard" "source ~/envs/cata/bin/activate" C-m
tmux send-keys -t "tensorboard" "tensorboard --logdir results/" C-m
