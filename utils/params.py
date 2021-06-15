opt = dict( load_size=286,          # scale images to this size
            crop_size=256,          # then crop to this size
            batch_size=1,           # input batch size
            num_threads=4,          # treads for loading data
            gpu_ids=[0],            # id of gpu
            lambda_identity=0.5,    # use identity mapping. Setting lambda_identity other than 0 has an effect of scaling the weight of the identity mapping loss. For example, if the weight of the identity loss should be 10 times smaller than the weight of the reconstruction loss, please set lambda_identity = 0.1
            lamda_A=10.0,           # weight for cycle loss (A -> B -> A)
            lamda_B=10.0,           # weight for cycle loss (B -> A -> B)
            input_nc=3,
            output_nc=3,
            ngf=64,                 # of gen filters in the last conv layer
            ndf=64,                 # of discriminator filters in the first conv layer
            no_dropout=False,
            init_gain=0.02,         # scaling factor for normal, xavier and orthogonal.
            pool_size=50,           # the size of image buffer that stores previously generated images
            lr=0.0002,              # initial learning rate for adam
            beta1=0.5,              # momentum term of adam
            display_id=1,           # window id of the web display
            no_html=False,
            display_winsize=256,    # display window size for both visdom and HTML
            name='Chinese Painting Style',
            display_port=8888,      # visdom port of the web display
            display_ncols=4,        # if positive, display all images in a single visdom web panel with certain number of images per row.
            display_server="http://localhost",  # visdom server of the web display
            display_env='main',
            checkpoints_dir='.\\checkpoints',
            n_layers_D=3,           # only used if netD==n_layers

            epoch_count=1,
            n_epochs=100,
            n_epochs_decay=100,

            continue_train=False,
            load_iter=0,
            epoch='latest',
            verbose=False,

           )
