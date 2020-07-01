import argparse

class options():
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.initialized = False

    def initialize(self):
        # experiment specifics

        # Training Options
        self.parser.add_argument('--epochs', type=int, nargs='?', default=30, help='total number of training episodes')

        self.parser.add_argument('--batch_size', type=int, nargs='?', default=512, help='batch size to be used')

        self.parser.add_argument('--lstmDim', type=int, nargs='?', default=64, help='batch size to be used')

        self.parser.add_argument('--lr', type=int, nargs='?', default=0.0001, help='learning rate')
        self.parser.add_argument('--beta1', type=int, nargs='?', default=0.5, help='beta1 for adam optimizer')
        self.parser.add_argument('--beta2', type=int, nargs='?', default=0.999, help='beta2 for adam optimizer')

        self.parser.add_argument('--lrelu_val', type=int, nargs='?', default=0.01, help='leaky Relu Value')
        self.parser.add_argument('--dropout', type=float, nargs='?', default=0.20, help='dropout rate')

        self.parser.add_argument('--resume', type=bool, nargs='?', default=False, help='Resume Training')
        self.parser.add_argument('--save_progress', type=bool, nargs='?', default=True, help='save training progress')

        self.parser.add_argument('--numClasses', type=int, nargs='?', default=10, help='number of classes')


    def parse(self):
        self.initialize()
        self.opt = self.parser.parse_args()

        return self.opt

