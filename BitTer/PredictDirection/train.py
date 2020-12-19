
from model import TCN


if __name__ == '__main__':
    n_levels = 4
    n_hidden = 150
    input_size = 60
    n_channels = [n_hidden] * n_levels

    model = TCN(input_size=input_size, output_size=1, num_channels=n_channels, kernel_size=3)
    print(model)
