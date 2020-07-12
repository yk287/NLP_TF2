import argparse

class options():
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.initialized = False

    def initialize(self):
        # experiment specifics

        # Training Options
        self.parser.add_argument('--epochs', type=int, nargs='?', default=3, help='total number of training episodes')

        self.parser.add_argument('--batch_size', type=int, nargs='?', default=6, help='batch size to be used')
        self.parser.add_argument('--max_len', type=int, nargs='?', default=512, help='max length of sentences to be used')

        self.parser.add_argument('--file_name', type=str, nargs='?', default='/home/youngwook/Downloads/nlp/imdb.csv', help='filename of dataset being used')
        self.parser.add_argument('--model_loc', type=str, nargs='?', default='/home/youngwook/Documents/Bert',
                                 help='where the model resides, can be local or from tf hub')

        self.parser.add_argument('--lr', type=float, nargs='?', default=2e-5, help='learning rate')
        self.parser.add_argument('--beta1', type=float, nargs='?', default=0.5, help='beta1 for adam optimizer')
        self.parser.add_argument('--beta2', type=float, nargs='?', default=0.999, help='beta2 for adam optimizer')

        self.parser.add_argument('--lrelu_val', type=int, nargs='?', default=0.01, help='leaky Relu Value')
        self.parser.add_argument('--dropout', type=float, nargs='?', default=0.20, help='dropout rate')

        self.parser.add_argument('--resume', type=bool, nargs='?', default=False, help='Resume Training')
        self.parser.add_argument('--save_progress', type=bool, nargs='?', default=True, help='save training progress')

        self.parser.add_argument('--num_classes', type=int, nargs='?', default=1, help='number of classes to predict')

        self.parser.add_argument('--random_seed', type=int, nargs='?', default=666, help='random seed for train test split')
        self.parser.add_argument('--test_split', type=float, nargs='?', default=0.20, help='proportion of test data')
        self.parser.add_argument('--validation_split', type=float, nargs='?', default=0.20, help='proportion of validation data')

        self.parser.add_argument('--print_model', type=bool, nargs='?', default=True, help='print model configs')

    def parse(self):
        self.initialize()
        self.opt = self.parser.parse_args()

        return self.opt

