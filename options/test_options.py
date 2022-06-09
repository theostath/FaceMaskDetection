from .base_options import BaseOptions

class TestOptions(BaseOptions):
    """
    This class includes test options.
    It also includes shared options defined in BaseOptions.
    """

    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)  # define shared options
        #parser.add_argument('--results_dir', type=str, default='./results/', help='saves results here.')
        parser.add_argument('--phase', type=str, default='test', help='train, val, test, etc')
        parser.add_argument('--confidence', type=float, default=0.85, help='confidence level for face mask detection')

        self.isTrain = False
        return parser