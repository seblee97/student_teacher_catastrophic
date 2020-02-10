# kill sessions already running
tmux kill-session -t schedule_1
tmux kill-session -t schedule_2
tmux kill-session -t schedule_3
tmux kill-session -t schedule_4
tmux kill-session -t schedule_5
tmux kill-session -t schedule_6

tmux kill-session -t tensorboard_1
tmux kill-session -t tensorboard_2
tmux kill-session -t tensorboard_3
tmux kill-session -t tensorboard_4
tmux kill-session -t tensorboard_5
tmux kill-session -t tensorboard_6

# start tmux sessions one for each experiment
tmux new -s schedule_1 -d
tmux new -s schedule_2 -d
tmux new -s schedule_3 -d
tmux new -s schedule_4 -d
tmux new -s schedule_5 -d
tmux new -s schedule_6 -d

# initialise virtual environments
tmux send-keys -t schedule_1 "source ~/envs/cata/bin/activate" C-m
tmux send-keys -t schedule_2 "source ~/envs/cata/bin/activate" C-m
tmux send-keys -t schedule_3 "source ~/envs/cata/bin/activate" C-m
tmux send-keys -t schedule_4 "source ~/envs/cata/bin/activate" C-m
tmux send-keys -t schedule_5 "source ~/envs/cata/bin/activate" C-m
tmux send-keys -t schedule_6 "source ~/envs/cata/bin/activate" C-m

# send keys to run experiments
tmux send-keys -t schedule_1 "python main.py --lc continual --tc overlapping --sc fixed_period --fp 5000 --ts 100000 --v 1 --snl relu --tnl '[sigmoid, sigmoid]' --to '[0, 0]' --s 5 --en schedule_1" C-m
tmux send-keys -t schedule_2 "python main.py --lc continual --tc overlapping --sc fixed_period --fp 5000 --ts 100000 --v 1 --snl sigmoid --tnl '[relu, relu]' --to '[0, 0]' --s 5 --en schedule_2" C-m

tmux send-keys -t schedule_3 "python main.py --lc continual --tc overlapping --sc fixed_period --fp 5000 --ts 100000 --v 1 --snl relu --tnl '[relu, relu]' --to '[0, 0]' --s 5 --en big_hidden_relu_1 --th '[20]' --sh '[40]'" C-m
tmux send-keys -t schedule_4 "python main.py --lc continual --tc overlapping --sc fixed_period --fp 5000 --ts 100000 --v 1 --snl relu --tnl '[relu, relu]' --to '[0, 0]' --s 10 --en big_hidden_relu_2 --th '[20]' --sh '[40]'" C-m

tmux send-keys -t schedule_5 "python main.py --lc continual --tc overlapping --sc fixed_period --fp 5000 --ts 100000 --v 1 --snl linear --tnl '[linear, linear]' --to '[0, 0]' --s 5 --en big_hidden_lin_1 --th '[20]' --sh '[40]'" C-m
tmux send-keys -t schedule_6 "python main.py --lc continual --tc overlapping --sc fixed_period --fp 5000 --ts 100000 --v 1 --snl linear --tnl '[linear, linear]' --to '[0, 0]' --s 10 --en big_hidden_lin_2 --th '[20]' --sh '[40]'" C-m

# start tmux session for tensorboard, launch tensorboard
tmux new -s tensorboard_1 -d
tmux new -s tensorboard_2 -d
tmux new -s tensorboard_3 -d
tmux new -s tensorboard_4 -d
tmux new -s tensorboard_5 -d
tmux new -s tensorboard_6 -d

tmux send-keys -t tensorboard_1 "source ~/envs/cata/bin/activate" C-m
tmux send-keys -t tensorboard_2 "source ~/envs/cata/bin/activate" C-m
tmux send-keys -t tensorboard_3 "source ~/envs/cata/bin/activate" C-m
tmux send-keys -t tensorboard_4 "source ~/envs/cata/bin/activate" C-m
tmux send-keys -t tensorboard_5 "source ~/envs/cata/bin/activate" C-m
tmux send-keys -t tensorboard_6 "source ~/envs/cata/bin/activate" C-m
