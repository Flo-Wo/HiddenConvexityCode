import argparse


class Parser(argparse.ArgumentParser):
    def __init__(self):
        super().__init__()
        self.add_argument(
            "--run_algos",
            action="store_true",
            help="Run Algorithms. If false, only the problem will be visualized.",
        )
        self.add_argument(
            "--hide_feasible_set",
            action="store_true",
            help="Show the feasible set. If false, only the level sets are displayed.",
        )

    def parse(self):
        return self.parse_args()
