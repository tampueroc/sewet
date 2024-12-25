import argparse


def arg_parser():
    args = argparse.ArgumentParser(description='Sewet')

    args.add_argument('-gpu', type=str, default='0',
                      help='GPU to run the model on')
    args.add_argument('-device', type=str, default='gpu',
                      help='Device to run the model on')
    args.add_argument('-sequence_lenght', type=int, default=2,
                    help='Sequence lenght for the fire')
    args.add_argument('-batch_size', type=int, default=1,
                    help='Batch size for the fire')
    args.add_argument('-num_workers', type=int, default=4,
                    help='Number of workers for the fire')
    args.add_argument('-drop_last', type=bool, default=True,
                    help='Drop last for the fire')
    args = args.parse_args()
    return args
