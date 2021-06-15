from . import cycle_gan


def create_model(opt):
    instance = cycle_gan.CycleGANModel(opt)
    print("model [%s] was created" % type(instance).__name__)
    return instance
