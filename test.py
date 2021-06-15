import os
from utils import create_dataset
from models import create_model
from utils.visualizer import save_images
from utils import html
from utils.params import opt
import warnings
warnings.filterwarnings("ignore")


if __name__ == '__main__':
    dataset = create_dataset('test')  # create a dataset given opt.dataset_mode and other options
    dataset_size = len(dataset)
    print('The number of training images = %d' % dataset_size)
    model = create_model(False)      # create a model given opt.model and other options
    model.setup()               # regular setup: load and print networks; create schedulers
    # create a website
    web_dir = os.path.join(opt['results_dir'], opt['name'], '{}_{}'.format(opt['phase'], opt['epoch']))  # define the website directory
    if opt['load_iter'] > 0:  # load_iter is 0 by default
        web_dir = '{:s}_iter{:d}'.format(web_dir, opt.load_iter)
    print('creating web directory', web_dir)
    webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt['name'], opt['phase'], opt['epoch']))
    # test with eval mode. This only affects layers like batchnorm and dropout.
    # For [pix2pix]: we use batchnorm and dropout in the original pix2pix. You can experiment it with and without eval() mode.
    # For [CycleGAN]: It should not affect CycleGAN as CycleGAN uses instancenorm without dropout.
    # if opt.eval:
    #     model.eval()
    for i, data in enumerate(dataset):
        if i >= opt['num_test']:  # only apply our model to opt.num_test images.
            break
        model.set_input(data)  # unpack data from data loader
        model.test()           # run inference
        visuals = model.get_current_visuals()  # get image results
        img_path = model.get_image_paths()     # get image paths
        if i % 5 == 0:  # save images to an HTML file
            print('processing (%04d)-th image... %s' % (i, img_path))
        save_images(webpage, visuals, img_path, aspect_ratio=opt['aspect_ratio'], width=opt['display_winsize'])
    webpage.save()  # save the HTML
