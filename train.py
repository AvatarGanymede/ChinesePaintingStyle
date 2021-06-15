from utils import create_dataset
from models import create_model
from utils.visualizer import Visualizer
import warnings
import torch

warnings.filterwarnings("ignore")

if __name__ == '__main__':
    dataset = create_dataset('train')
    dataset_size = len(dataset)
    print('The number of training images = %d' % dataset_size)

    model = create_model(True)  # create a model given opt.model and other options
    model.setup()  # regular setup: load and print networks; create schedulers
    visualizer = Visualizer(isTrain=True)  # create a visualizer that display/save images and plots
    total_iters = 0  # the total number of training iterations
