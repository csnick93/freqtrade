import torch


class MLP(torch.nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.layer_1 = torch.nn.Linear(500, 256)
        self.layer_2 = torch.nn.Linear(256, 128)
        self.layer_3 = torch.nn.Linear(128, 2)

    def forward(self, x):
        x_1 = self.layer_1(x)
        x_1_a = torch.relu(x_1)
        x_2 = self.layer_2(x_1_a)
        x_2_a = torch.relu(x_2)
        out = self.layer_3(x_2_a)
        return out


class CNN(torch.nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.layer_1 = torch.nn.Conv1d(in_channels=1,
                                       out_channels=3,
                                       kernel_size=11)
        self.layer_2 = torch.nn.Conv1d(in_channels=3,
                                       out_channels=5,
                                       kernel_size=11)
        self.layer_3 = torch.nn.Conv1d(in_channels=5,
                                       out_channels=3,
                                       kernel_size=11)
        self.layer_4 = torch.nn.Linear(126, 2)

    def forward(self, x):
        x_1 = self.layer_1(x)
        x_1_a = torch.nn.LeakyReLU()(x_1)
        x_2 = self.layer_2(x_1_a)
        x_2_a = torch.nn.LeakyReLU()(x_2)
        x_3 = self.layer_3(x_2_a)
        x_3_a = torch.nn.LeakyReLU()(x_3)
        x_4 = torch.flatten(x_3_a, start_dim=1)
        x_4_a = self.layer_4(x_4)
        return x_4_a
