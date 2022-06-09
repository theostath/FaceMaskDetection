from .base_options import BaseOptions

class TrainOptions(BaseOptions):
    """
    This class includes training options.
    It also includes shared options defined in BaseOptions.
    """

    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)
        # network saving and loading parameters
        parser.add_argument('--phase', type=str, default='train', help='train, val, test, etc')

        # training parameters
        parser.add_argument('--lr', type=float, default=0.0001, help='initial learning rate for Adam')
        parser.add_argument('--beta1', type=float, default=0.9, help='initial beta1 -exponential decay rate for 1st moment estimates- for Adam')
        parser.add_argument('--beta2', type=float, default=0.999, help='initial beta2 -exponential decay rate for 2nd moment estimates- for Adam')

        self.isTrain = True
        return parser
