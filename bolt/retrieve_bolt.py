import turibolt as bolt
import argparse
import os

arg_parser = argparse.ArgumentParser()

arg_parser.add_argument("--parent_id")
arg_parser.add_argument("--save_dir")

if __name__ == "__main__":
    args = arg_parser.parse_args()
    parent_trial = bolt.get_task(args.parent_id)
    for c, child in enumerate(parent_trial.children):
        if c > 21:
            print(child.name)
            child.artifacts.download_file(
                src='results_zip.zip', dest=os.path.join(args.save_dir, f"{c}_results.zip"))
