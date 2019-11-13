# kill sessions already running
tmux kill-session -t seed_5
tmux kill-session -t seed_10
tmux kill-session -t seed_15
tmux kill-session -t seed_20
tmux kill-session -t seed_25

tmux kill-session -t tensorboard

# start tmux sessions one for each experiment
tmux new -s seed_5 -d
tmux new -s seed_10 -d
tmux new -s seed_15 -d
tmux new -s seed_20 -d
tmux new -s seed_25 -d

# initialise virtual environments
tmux send-keys -t seed_5 "source ~/envs/cata/bin/activate" C-m
tmux send-keys -t seed_10 "source ~/envs/cata/bin/activate" C-m
tmux send-keys -t seed_15 "source ~/envs/cata/bin/activate" C-m
tmux send-keys -t seed_20 "source ~/envs/cata/bin/activate" C-m
tmux send-keys -t seed_25 "source ~/envs/cata/bin/activate" C-m

# send keys to run experiments
tmux send-keys -t seed_5 "python main.py --lc continual --tc overlapping --sc threshold --ts 100000 --v 0 --cp '/content/drive/My Drive/colab_results/threshold' --nl relu --to '[0, 0]' --s 5 --en seed5" C-m
tmux send-keys -t seed_10 "python main.py --lc continual --tc overlapping --sc threshold --ts 100000 --v 0 --cp '/content/drive/My Drive/colab_results/threshold' --nl relu --to '[0, 0]' --s 10 --en seed10" C-m
tmux send-keys -t seed_15 "python main.py --lc continual --tc overlapping --sc threshold --ts 100000 --v 0 --cp '/content/drive/My Drive/colab_results/threshold' --nl relu --to '[0, 0]' --s 15 --en seed15" C-m
tmux send-keys -t seed_20 "python main.py --lc continual --tc overlapping --sc threshold --ts 100000 --v 0 --cp '/content/drive/My Drive/colab_results/threshold' --nl relu --to '[0, 0]' --s 20 --en seed20" C-m
tmux send-keys -t seed_25 "python main.py --lc continual --tc overlapping --sc threshold --ts 100000 --v 0 --cp '/content/drive/My Drive/colab_results/threshold' --nl relu --to '[0, 0]' --s 25 --en seed25" C-m

# start tmux session for tensorboard, launch tensorboard
tmux new -s tensorboard -d
tmux send-keys -t "tensorboard" "source ~/envs/cata/bin/activate" C-m
tmux send-keys -t "tensorboard" "tensorboard --logdir results/" C-m
