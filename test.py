from util.magtense.prism_grid import PrismGridDataset, create_dataset, create_prism_grid


if __name__ == '__main__':
    x, m, y = create_prism_grid(2,2,seed=0, res=32)
    print(x)