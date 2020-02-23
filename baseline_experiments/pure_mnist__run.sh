# kill sessions already running
tmux kill-session -t schedule_1
tmux kill-session -t schedule_2

tmux kill-session -t tensorboard_1
tmux kill-session -t tensorboard_2

# start tmux sessions one for each experiment
tmux new -s schedule_1 -d
tmux new -s schedule_2 -d

# initialise virtual environments
tmux send-keys -t schedule_1 "source ~/envs/cata/bin/activate" C-m
tmux send-keys -t schedule_2 "source ~/envs/cata/bin/activate" C-m

# send keys to run experiments

tmux send-keys -t schedule_1 "python main.py -config configs/pure_mnist/relu/base_config.yaml --ac additional_configs/ --en pure_mnist_classification_relu"
tmum send-keys -t schedule_2 "python main.py -config configs/pure_mnist/sigmoid/base_config.yaml --ac additional_configs/ --en pure_mnist_classification_sigmoid"

# start tmux session for tensorboard, launch tensorboard
# tmux new -s tensorboard_1 -d
# tmux new -s tensorboard_2 -d
# tmux new -s tensorboard_3 -d
# tmux new -s tensorboard_4 -d
# tmux new -s tensorboard_5 -d
# tmux new -s tensorboard_6 -d

# tmux send-keys -t tensorboard_1 "source ~/envs/cata/bin/activate" C-m
# tmux send-keys -t tensorboard_2 "source ~/envs/cata/bin/activate" C-m
# tmux send-keys -t tensorboard_3 "source ~/envs/cata/bin/activate" C-m
# tmux send-keys -t tensorboard_4 "source ~/envs/cata/bin/activate" C-m
# tmux send-keys -t tensorboard_5 "source ~/envs/cata/bin/activate" C-m
# tmux send-keys -t tensorboard_6 "source ~/envs/cata/bin/activate" C-m

# # tmux send-keys -t tensorboard_1 "cd results/2020*/schedule_1; tensorboard --logdir . --port=6006" C-m
# # tmux send-keys -t tensorboard_2 "cd results/2020*/schedule_2; tensorboard --logdir . --port=6007" C-m
# tmux send-keys -t tensorboard_3 "cd results/2020*/big_hidden_relu_1; tensorboard --logdir . --port=6008" C-m
# tmux send-keys -t tensorboard_4 "cd results/2020*/big_hidden_relu_2; tensorboard --logdir . --port=6009" C-m
# tmux send-keys -t tensorboard_5 "cd results/2020*/big_hidden_lin_1; tensorboard --logdir . --port=6010" C-m
# tmux send-keys -t tensorboard_6 "cd results/2020*/big_hidden_lin_2; tensorboard --logdir . --port=6011" C-m
