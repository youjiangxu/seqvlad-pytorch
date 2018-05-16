import argparse
parser = argparse.ArgumentParser(description="PyTorch implementation of Temporal Segment Networks")
parser.add_argument('dataset', type=str, choices=['ucf101', 'hmdb51', 'kinetics'])
parser.add_argument('modality', type=str, choices=['RGB', 'Flow', 'RGBDiff'])
parser.add_argument('train_list', type=str)
parser.add_argument('val_list', type=str)

# ========================= Model Configs ==========================
parser.add_argument('--arch', type=str, default="resnet101")
parser.add_argument('--timesteps', type=int, default=3)
parser.add_argument('--consensus_type', type=str, default='avg',
                    choices=['avg', 'max', 'topk', 'identity', 'rnn', 'cnn'])
parser.add_argument('--k', type=int, default=3)

parser.add_argument('--dropout', '--do', default=0.5, type=float,
                    metavar='DO', help='dropout ratio (default: 0.5)')
parser.add_argument('--loss_type', type=str, default="nll",
                    choices=['nll'])

# ========================= Learning Configs ==========================
parser.add_argument('--epochs', default=45, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--lr_steps', default=[20, 40], type=float, nargs="+",
                    metavar='LRSteps', help='epochs to decay learning rate by 10')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float,
                    metavar='W', help='weight decay (default: 5e-4)')
parser.add_argument('--clip-gradient', '--gd', default=None, type=float,
                    metavar='W', help='gradient norm clipping (default: disabled)')
parser.add_argument('--no_partialbn', '--npb', default=False, action="store_true")

# ========================= Monitor Configs ==========================
parser.add_argument('--print-freq', '-p', default=20, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--eval-freq', '-ef', default=5, type=int,
                    metavar='N', help='evaluation frequency (default: 5)')


# ========================= Runtime Configs ==========================
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--snapshot_pref', type=str, default="")
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--gpus', nargs='+', type=int, default=None)
parser.add_argument('--flow_prefix', default="", type=str)


# ======================== add root_path =============================
parser.add_argument('-s', '--sources', default='', type=str, metavar='PATH',
                    help='the sources of the dataset')
parser.add_argument('--num_centers', default=64, type=int, metavar='N',
                    help='the number of centers for seqvlad')

parser.add_argument('--redu_dim', default=512, type=int, metavar='N',
                    help='redu channels for input feature maps for seqvlad')

parser.add_argument('--resume_type', default='same', type=str, metavar='',
                    help='set the type of the pretrained model, must be one of [same, tsn] ')

parser.add_argument('--with_relu', action='store_true', default=False,
                    help='set relu for reduction convolution')
parser.add_argument('--activation', type=str, default=None,
                    help='define the activation of the assignments, default is None')

parser.add_argument('--optim', type=str, default="SGD", choices=['SGD', 'Adam'],
                    help='define the optimizer, default is SGD ')

parser.add_argument('--two_steps', default=None, type=int, metavar='N',
                    help='in the first step, we only train')

parser.add_argument('--sampling_method', default='tsn', type=str, choices=['tsn', 'random', 'reverse'],
                    help='defint sampling method for training procedure')

parser.add_argument('--reverse', default=False, action='store_true',
                    help='reverse the sampling order, there will be 0.5 probability to reverse the sequence if set True')


parser.add_argument('--seqvlad_type', default='seqvlad', choices=['seqvlad', 'bidirect', 'unshare_bidirect'],
                    help='use seqvlad_type, defaults is seqvlad')

parser.add_argument('--lossweight', default=1.0, type=float, metavar='M',
                    help='lossweight')


parser.add_argument('--init_method', default='orthogonal', choices=['xavier_normal', 'orthogonal', 'uniform'],
                    help='set init method for hidden to hidden paramters, e.g., U_z, U_r, and U_h.')
