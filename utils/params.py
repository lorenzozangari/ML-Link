import argparse


def set_params():
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type=str, default="ckm", help='Name of the input network (subfolder of data/nets).')
    parser.add_argument('--gpu', type=int, default=-1, help='Which GPU to use (-1 for CPU).')
    parser.add_argument('--seed', type=int, default=72, help='Random seed.')
    parser.add_argument('--runs', type=int, default=10, help='Number of runs.')

    parser.add_argument('--edge_dim', type=int, default=8, help='Hidden dimension of the edge MLP.')
    parser.add_argument('--node_dim', type=int, default=128, help='Hidden dimension of node MLP.')
    parser.add_argument('--phi_dim', type=int, default=128, help='Hidden dimension of context MLP.')
    parser.add_argument('--hidden_dim', type=int, default=256, help='Hidden dimension of the GNN layers.')
    parser.add_argument("--num_hidden", type=int, default=2,
                        help="Numbers of hidden layers of the GNN.")
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs.')

    parser.add_argument('--n_heads', type=int, default=1, help='Number of attentions heads of GNN.')
    parser.add_argument('--heads_mode', type=str, default='concat', help='Concatenate (concat) or averaging (avg)'
                                                                         ' the multiple attention heads.')
    parser.add_argument('--predictor', type=str, default="mlp", help='MLP decoder.')
    parser.add_argument('--omn', type=str, default='oan;maan', help='Types of overlapping multilayer neighborhoods.'
                                                                    'Supported types are oan and maan. '
                                                                    'Each overlapping multilayer neighborhood must be followed by a semicolon, e.g., oan;maan')

    parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate.')
    parser.add_argument('--attn_dropout', type=float, default=0.7,
                        help='Attention dropout rate for attention based GNN models.')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-5, help='Weight decay (L2 loss).')
    parser.add_argument('--psi', type=float, default=0.5, help='Impact of overlapping multilayer neighborhoods.')
    parser.add_argument('--no_gnn', action='store_true', help='Whether to use only the NN-NPN component.')
    parser.add_argument('--no_struct', action='store_true', help='Whether to use only the GNN-NE component.')

    parser.add_argument('--root', type=str, default='./data/nets/', help='Root directory of input data.')
    parser.add_argument('--save_dir', type=str, default='./artifacts/', help='Folder where the performance'
                                                                             ' scores are saved.')
    parser.add_argument('--prep_dir', type=str, default='./data/prep_nets/', help='Folder storing the preprocessed data.')
    parser.add_argument('--ck_dir', type=str, default="checkpoint", help='Folder where the checkpoint model is stored')
    args, _ = parser.parse_known_args()
    return args
