from . import cycle_gan


def create_model(isTrain):
    instance = cycle_gan.CycleGANModel(isTrain)
    print("model [%s] was created" % type(instance).__name__)
    return instance
