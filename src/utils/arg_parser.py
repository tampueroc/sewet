import argparse


def arg_parser():
    parser = argparse.ArgumentParser(description='Sewet')
    parser.add_argument('--data_dir', type=str, default="data/deep_crown_dataset/organized_spreads",
                        help="Path to the dataset directory (default: %(default)s)")
    parser.add_argument('--sequence_length', type=int, default=6, required=True,
                        help="Sequence length for the dataset (default: %(default)s)")
    parser.add_argument('--metrics_threshold', type=float, default=0.5,
                        help="Threshold for binary classification metrics (default: %(default)s)")
    parser.add_argument('--save_weights', action='store_true', default=True,
                        help="Enable saving model weights")
    parser.add_argument('--drop_last', action='store_true',
                        help="Drop the last batch of the DataLoader")
    parser.add_argument('--early_stop_patience', type=int, default=5,
                        help="Early stopper patience (default: %(default)s)")
    parser.add_argument('--early_stop_minimum_delta', type=float, default=0.01,
                        help="Early stopper minimum delta (default: %(default)s)")
    parser.add_argument('--pin_memory', action='store_true',
                        help="Enable pin_memory for DataLoader (default: False)")
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                        help="Learning rate for the optimizer (default: %(default)s)")
    parser.add_argument('--num_epochs', type=int, default=10, required=True,
                        help="Number of epochs for training (default: %(default)s)")
    parser.add_argument('-depth' , type=int, default=6,
                        help='Depth of the model')

    parser.add_argument('-gpu', type=str, default='0',
                      help='GPU to run the model on')
    parser.add_argument('-device', type=str, default='gpu',
                      help='Device to run the model on')
    parser.add_argument('-batch_size', type=int, default=32,
                    help='Batch size for the fire')
    parser.add_argument('-num_workers', type=int, default=4,
                    help='Number of workers for the fire')
    parser.add_argument('-drop_last', type=bool, default=True,
                    help='Drop last for the fire')
    return parser.parse_args()
