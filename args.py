import argparse


parser = argparse.ArgumentParser(description='Byte level generation using Multiplicative LSTM')
parser.add_argument('--epochs', default=10, help='Number of epochs for training')
parser.add_argument('--save_model', default='model_output', help='Model filename to store')
parser.add_argument('--load_model', default='', help='Model filename for loading')
parser.add_argument('--train_data', default='', help='Dataset filename for training model')
parser.add_argument('--test', action='store_true', help='Testing the model')
parser.add_argument('--primer', default=u' ', help='Primer for generating sample')
parser.add_argument('--paperspace', action='store_true', help='Set to true to run model in paperspace')
parser.add_argument('--n', default=2000, help='Number of timesteps for output')
parser.add_argument('--cuda', action='store_true', help="Enable CUDA")
