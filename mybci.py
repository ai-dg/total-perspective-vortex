import sys
import argparse
from logreg import LogReg


class MyBCI:
    """
        Main interface class for the Brain-Computer Interface system.
        Handles command-line arguments and orchestrates training and
        prediction workflows.
    """

    def __init__(self, args: argparse.Namespace):
        """
            Logic:
            - Stores command-line arguments (subject_id, run, mode)
            - Initializes a LogReg instance for model operations
            Return:
            - None
        """
        self.args = args
        self.logreg = LogReg()

    def ft_run_all_subjects_training(self):
        """
            Logic:
            - Runs comprehensive experiments across all 109 subjects
            - Tests 6 different training configurations
              (different run combinations)
            - For each experiment: trains on one random run, tests on
              remaining runs
            - Computes mean accuracy per experiment and overall mean accuracy
            - Prints detailed results for each subject and experiment
            Return:
            - None
        """
        experiments = [
            [3, 7, 11],
            [4, 8, 12],
            [3, 4, 7, 8, 11, 12],
            [5, 9, 13],
            [6, 10, 14],
            [5, 6, 9, 10, 13, 14],
        ]
        SUBJECT_IDS = range(1, 110)
        exp_mean_accuracies = {
            0: 0.0,
            1: 0.0,
            2: 0.0,
            3: 0.0,
            4: 0.0,
            5: 0.0,
        }

        for exp_id, experiment in enumerate(experiments):
            exp_subject_acc = []

            for subject_id in SUBJECT_IDS:

                train_run = experiment[0]
                self.logreg.ft_train_model(subject_id, train_run, False)

                accuracies = []

                for run in experiment:
                    if run == train_run:
                        continue
                    acc = self.logreg.ft_predict_model(
                        subject_id, run, "summary")
                    accuracies.append(acc)

                if len(accuracies) > 0:
                    experiment_accuracy = sum(accuracies) / len(accuracies)
                else:
                    experiment_accuracy = 0.0

                exp_subject_acc.append(experiment_accuracy)

                print(
                    f"experiment {exp_id}: subject {subject_id:03d}: "
                    f"accuracy = {experiment_accuracy:.1f}"
                )

            mean_exp = sum(exp_subject_acc) / len(exp_subject_acc)
            exp_mean_accuracies[exp_id] = mean_exp

        print(
            "Mean accuracy of the six different experiments "
            "for all 109 subjects:")
        for exp_id, mean_exp in exp_mean_accuracies.items():
            print(
                f"experiment {exp_id}: "
                f"accuracy = {mean_exp:.4f}")

        overall_mean = sum(exp_mean_accuracies.values()) / \
            len(exp_mean_accuracies)
        print(
            f"Mean accuracy of {len(experiments)} "
            f"experiments: {overall_mean:.4f}")

    def ft_launch_training(self):
        """
            Logic:
            - Routes to appropriate mode based on command-line arguments
            - 'train': trains model with verbose output
            - 'predict': runs prediction with full detailed output
            - 'stream': runs prediction in stream mode (epoch by epoch)
            - No arguments: runs comprehensive experiments on all subjects
            Return:
            - None
        """
        print("launching training...")
        print("Run types: ")
        print("left_fist_right_fist : runs 3,4,7,8,11,12")
        print("both_fists_both_feet : runs 5,6,9,10,13,14")
        if self.args.mode == 'train':
            self.logreg.ft_train_model(
                self.args.subject_id, self.args.run, True)
        elif self.args.mode == 'predict':
            self.logreg.ft_predict_model(
                self.args.subject_id, self.args.run, 'full')
        elif (self.args.mode is None and self.args.subject_id is None
              and self.args.run is None):
            self.ft_run_all_subjects_training()
        elif self.args.mode == 'stream':
            self.logreg.ft_predict_model(
                self.args.subject_id, self.args.run, 'full', True)

    args: argparse.Namespace
    logreg: LogReg


def main():
    parser = argparse.ArgumentParser(
        description='Total Perspective Vortex - Brain Computer Interface',
    )

    parser.add_argument('subject_id', type=int, nargs='?',
                        help='Subject ID (1-109)')
    parser.add_argument('run', type=int, nargs='?',
                        help='Run number (1-14)')
    parser.add_argument(
        'mode',
        type=str,
        nargs='?',
        choices=[
            'train',
            'predict',
            'stream'],
        help='Mode: train, predict, stream')

    args = parser.parse_args()

    if args.mode is None and args.subject_id and args.run:
        print(
            "If you put subject_id and run, "
            "you must put a mode (train or predict)")
        sys.exit(1)

    mybci = MyBCI(args)

    mybci.ft_launch_training()


if __name__ == "__main__":
    main()
