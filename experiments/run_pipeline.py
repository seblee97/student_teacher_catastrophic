import argparse

from experiments.config_template import ConfigTemplate


if __name__ == "__main__":

    ConfigTemplate.base
    parser = argparse.ArgumentParser()
    args = Argparser.process_parser(parser)

    if args.ppp is not None:
        post_process(args)
    else:
        run(args)