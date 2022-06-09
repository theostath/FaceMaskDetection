import argparse
import os


class BaseOptions():
    """
    This class defines options used during both training and test time.
    It also implements several helper functions such as parsing, printing, and saving the options.
    """

    def __init__(self, cmd_line=None):
        """
        Reset the class. 
        Indicates the class hasn't been initailized.
        """
        self.initialized = False
        self.cmd_line = None
        if cmd_line is not None:
            self.cmd_line = cmd_line.split()

      
    def initialize(self, parser):
        """
        Define the common options that are used in both training and test.
        """
        # basic parameters
        parser.add_argument('--dataroot', default='./face-mask-dataset', help='path to images')
        parser.add_argument('--name', type=str, default='experiment_name', help='name of the experiment. It decides where to store samples and models')
        parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here')
        # model parameters
        parser.add_argument('--nf', type=int, default=100, help='# of filters in the first conv layer')
        parser.add_argument('--no_dropout', action='store_true', help='If specified, do not use dropout for the CNN')
        # train parameter (used in test for the name of the save_dir)
        parser.add_argument('--n_epochs', type=int, default=20, help='number of epochs with the initial learning rate')
        # dataset parameters
        parser.add_argument('--batch_size', type=int, default=32, help='input batch size')
        parser.add_argument('--no_flip', action='store_true', help='if specified, do not flip the images for data augmentation')
        # additional parameters
        parser.add_argument('--suffix', default='', type=str, help='customized suffix: opt.name = opt.name + suffix: e.g., {model}_{netG}_size{load_size}')
        parser.add_argument('--verbose', action='store_true', help='if specified, print more debugging information and save .txt')
        
        self.initialized = True
        return parser

    def gather_options(self):
        """
        Initialize our parser with basic options(only once).
        """
        # check if it has been initialized
        if not self.initialized:
            parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            parser = self.initialize(parser)

        # get the basic options
        if self.cmd_line is None:
            opt, _ = parser.parse_known_args()
        else:
            opt, _ = parser.parse_known_args(self.cmd_line)

        # modify model-related parser options
        if self.cmd_line is None:
            opt, _ = parser.parse_known_args()  # parse again with new defaults
        else:
            opt, _ = parser.parse_known_args(self.cmd_line)  # parse again with new defaults

        # save and return the parser
        self.parser = parser
        if self.cmd_line is None:
            return parser.parse_args()
        else:
            return parser.parse_args(self.cmd_line)

    def print_options(self, opt):
        """
        Print and save options.
        It will print both current options and default values(if different).
        It will save options into a text file / [checkpoints_dir] / [name] / opt.txt
        """
        message = ''
        message += '----------------- Options ---------------\n'
        for k, v in sorted(vars(opt).items()):
            comment = ''
            default = self.parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'
        print(message)

        # save to the disk
        expr_dir = os.path.join(opt.checkpoints_dir, opt.name)

        if not os.path.exists(expr_dir):
            os.makedirs(expr_dir)
        
        file_name = os.path.join(expr_dir, '{}_opt.txt'.format(opt.phase))
        try:
            with open(file_name, 'wt') as opt_file:
                opt_file.write(message)
                opt_file.write('\n')
        except PermissionError as error:
            print("permission error {}".format(error))
            pass

    def parse(self):
        """
        Parse our options, create checkpoints directory suffix.
        """
        opt = self.gather_options()
        opt.isTrain = self.isTrain   # train or test

        # process opt.suffix
        if opt.suffix:
            suffix = ('_' + opt.suffix.format(**vars(opt))) if opt.suffix != '' else ''
            opt.name = opt.name + suffix

        self.print_options(opt)

        self.opt = opt
        return self.opt