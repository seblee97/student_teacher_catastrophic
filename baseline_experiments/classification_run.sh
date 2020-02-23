# kill sessions already running
tmux kill-session -t schedule_1
tmux kill-session -t schedule_2
tmux kill-session -t schedule_3
tmux kill-session -t schedule_4
tmux kill-session -t schedule_5
tmux kill-session -t schedule_6
tmux kill-session -t schedule_7
tmux kill-session -t schedule_8

tmux kill-session -t tensorboard_1
tmux kill-session -t tensorboard_2
tmux kill-session -t tensorboard_3
tmux kill-session -t tensorboard_4
tmux kill-session -t tensorboard_5
tmux kill-session -t tensorboard_6
tmux kill-session -t tensorboard_7
tmux kill-session -t tensorboard_8

# start tmux sessions one for each experiment
tmux new -s schedule_1 -d
tmux new -s schedule_2 -d
tmux new -s schedule_3 -d
tmux new -s schedule_4 -d
tmux new -s schedule_5 -d
tmux new -s schedule_6 -d
tmux new -s schedule_7 -d
tmux new -s schedule_8 -d

# initialise virtual environments
tmux send-keys -t schedule_1 "source ~/envs/cata/bin/activate" C-m
tmux send-keys -t schedule_2 "source ~/envs/cata/bin/activate" C-m
tmux send-keys -t schedule_3 "source ~/envs/cata/bin/activate" C-m
tmux send-keys -t schedule_4 "source ~/envs/cata/bin/activate" C-m
tmux send-keys -t schedule_5 "source ~/envs/cata/bin/activate" C-m
tmux send-keys -t schedule_6 "source ~/envs/cata/bin/activate" C-m
tmux send-keys -t schedule_7 "source ~/envs/cata/bin/activate" C-m
tmux send-keys -t schedule_8 "source ~/envs/cata/bin/activate" C-m

# send keys to run experiments

tmux send-keys -t schedule_1 "python main.py -config configs/classification/relu_relu_big/base_config.yaml --ac additional_configs/ --en clssification_relu_relu_big"
tmux send-keys -t schedule_2 "python main.py -config configs/classification/relu_relu_small/base_config.yaml --ac additional_configs/ --en clssification_relu_relu_small"
tmux send-keys -t schedule_3 "python main.py -config configs/classification/relu_sigmoid_big/base_config.yaml --ac additional_configs/ --en clssification_relu_sigmoid_big"
tmux send-keys -t schedule_4 "python main.py -config configs/classification/relu_sigmoid_small/base_config.yaml --ac additional_configs/ --en clssification_relu_sigmoid_small"
tmux send-keys -t schedule_5 "python main.py -config configs/classification/sigmoid_relu_big/base_config.yaml --ac additional_configs/ --en clssification_sigmoid_relu_big"
tmux send-keys -t schedule_6 "python main.py -config configs/classification/sigmoid_relu_small/base_config.yaml --ac additional_configs/ --en clssification_sigmoid_relu_small"
tmux send-keys -t schedule_7 "python main.py -config configs/classification/sigmoid_sigmoid_big/base_config.yaml --ac additional_configs/ --en clssification_sigmoid_sigmoid_big"
tmux send-keys -t schedule_8 "python main.py -config configs/classification/sigmoid_sigmoid_small/base_config.yaml --ac additional_configs/ --en clssification_sigmoid_sigmoid_small"

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
