import argparse


parser = argparse.ArgumentParser(description='Takes input for main.py')
parser.add_argument('-save_model', default='model_output', help='Model filename to store')
parser.add_argument('-load_model', default='', help='Model filename for loading')
parser.add_argument('-train_data', default='', help='Dataset filename for training model')
parser.add_argument('-cuda', action='store_true', help="Use CUDA")