# regression
python main.py -config base_config.yaml --ac additional_configs/ --en regression_relu_relu_small --lty regression --snl relu --tnl relu --lr 0.2 --lf mse --tc overlapping --inp_s iid_gaussian
python main.py -config base_config.yaml --ac additional_configs/ --en regression_relu_sigmoid_small --lty regression --snl relu --tnl sigmoid --lr 0.2 --lf mse --tc overlapping --inp_s iid_gaussian
python main.py -config base_config.yaml --ac additional_configs/ --en regression_sigmoid_relu_small --lty regression --snl sigmoid --tnl relu --lr 0.2 --lf mse --tc overlapping --inp_s iid_gaussian
python main.py -config base_config.yaml --ac additional_configs/ --en regression_sigmoid_sigmoid_small --lty regression --snl sigmoid --tnl sigmoid --lr 0.2 --lf mse --tc overlapping --inp_s iid_gaussian

# classification
python main.py -config base_config.yaml --ac additional_configs/ --en clssification_relu_relu_small --lty classification --snl relu --tnl relu --lr 0.2 --lf nll --tc overlapping --inp_s iid_gaussian
python main.py -config base_config.yaml --ac additional_configs/ --en clssification_relu_sigmoid_small --lty classification --snl relu --tnl sigmoid --lr 0.2 --lf nll --tc overlapping --inp_s iid_gaussian
python main.py -config base_config.yaml --ac additional_configs/ --en clssification_sigmoid_relu_small --lty classification --snl sigmoid --tnl sigmoid --lr 0.2 --lf nll --tc overlapping --inp_s iid_gaussian
python main.py -config base_config.yaml --ac additional_configs/ --en clssification_sigmoid_sigmoid_small --lty classification --snl sigmoid --tnl sigmoid --lr 0.2 --lf nll --tc overlapping --inp_s iid_gaussian

# mnist-regression
python main.py -config base_config.yaml --ac additional_configs/ --en mnist_regression_relu_relu_small --lty regression --snl relu --tnl relu --lr 0.2 --lf mse --tc overlapping --inp_s mnist
python main.py -config base_config.yaml --ac additional_configs/ --en mnist_regression_relu_sigmoid_small --lty regression --snl relu --tnl sigmoid --lr 0.2 --lf mse --tc overlapping --inp_s mnist
python main.py -config base_config.yaml --ac additional_configs/ --en mnist_regression_sigmoid_relu_small --lty regression --snl sigmoid --tnl relu --lr 0.2 --lf mse --tc overlapping --inp_s mnist
python main.py -config base_config.yaml --ac additional_configs/ --en mnist_regression_sigmoid_sigmoid_small --lty regression --snl sigmoid --tnl sigmoid --lr 0.2 --lf mse --tc overlapping --inp_s mnist

# mnist-classification
python main.py -config base_config.yaml --ac additional_configs/ --en mnist_classification_relu_relu_small --lty classification --snl relu --tnl relu --lr 0.2 --lf nll --tc overlapping --inp_s mnist
python main.py -config base_config.yaml --ac additional_configs/ --en mnist_classification_relu_sigmoid_small --lty classification --snl relu --tnl sigmoid --lr 0.2 --lf nll --tc overlapping --inp_s mnist
python main.py -config base_config.yaml --ac additional_configs/ --en mnist_classification_sigmoid_relu_small --lty classification --snl sigmoid --tnl relu --lr 0.2 --lf nll --tc overlapping --inp_s mnist
python main.py -config base_config.yaml --ac additional_configs/ --en mnist_classification_sigmoid_sigmoid_small --lty classification --snl sigmoid --tnl sigmoid --lr 0.2 --lf nll --tc overlapping --inp_s mnist

# pure-mnist
python main.py -config base_config.yaml --ac additional_configs/ --en pure_mnist_classification_relu --lty not_applicable --snl relu --tnl relu --lr 0.2 --lf cross_entropy --tc mnist --inp_s iid_gaussian
python main.py -config base_config.yaml --ac additional_configs/ --en pure_mnist_classification_sigmoid --lty not_applicable --snl sigmoid --tnl sigmoid --lr 0.2 --lf cross_entropy --tc mnist --inp_s iid_gaussian
