from utils import create_dataset


if __name__ == '__main__':
    dataset = create_dataset('train')
    dataset_size = len(dataset)
    print(dataset_size)