import torch
from torch import nn, optim
from torch.autograd.variable import Variable
from torchvision import transforms, datasets

from utils import Logger


"""----------UTILITY FUNCTIONS------------------"""
def mnist_data():
    compose = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((.5, .5, .5), (.5, .5, .5))
        ])
    out_dir = './dataset'
    return datasets.MNIST(root=out_dir, train=True, transform=compose, download=True)

def im2vec(images):
    return images.view(images.size(0), 784)

def vec2im(vectors):
    return vectors.view(vectors.size(0), 1, 28, 28)

def gen_noise(size):
    if torch.cuda.is_available():
        return Variable(torch.randn(size, 100)).cuda()

    return Variable(torch.randn(size, 100))

def ones_target(size):
    if torch.cuda.is_available():
        return Variable(torch.ones(size, 1)).cuda()
    return Variable(torch.ones(size, 1))

def zeros_target(size):
    if torch.cuda.is_available():
        return Variable(torch.ones(size, 1)).cuda()

    return Variable(torch.zeros(size, 1))

"""------------ NETWORKS --------------------- """

class DiscriminatorNet(nn.Module):
    """
    The discrimator network
    """

    def __init__(self):
        super(DiscriminatorNet, self).__init__()
        n_features = 784
        n_out = 1

        self.h0 = nn.Sequential(
            nn.Linear(n_features, 1024),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3)
        )

        self.h1 = nn.Sequential(
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3)
        )

        self.h2 = nn.Sequential(
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3)
        )

        self.out = nn.Sequential(
            nn.Linear(256, n_out),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.h0(x)
        x = self.h1(x)
        x = self.h2(x)
        x = self.out(x)

        return x

class GeneratorNet(nn.Module):
    """
    The generator network
    """
    def __init__(self):
        super(GeneratorNet, self).__init__()
        n_features = 100
        n_out = 784

        self.h0 = nn.Sequential(
            nn.Linear(n_features, 256),
            nn.LeakyReLU(0.2)
        )

        self.h1 = nn.Sequential(
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2)
        )

        self.h2 = nn.Sequential(
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2)
        )

        self.out = nn.Sequential(
            nn.Linear(1024, n_out),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.h0(x)
        x = self.h1(x)
        x = self.h2(x)
        x = self.out(x)

        return x

"""------------------TRAINING FUNCTIONS---------------"""

def train_discriminator(optimizer, real_data, fake_data):
    N = real_data.size(0)
    optimizer.zero_grad()

    pred_real = dis(real_data)
    err_real = loss(pred_real, ones_target(N))
    err_real.backward()

    pred_fake = dis(fake_data)
    err_fake = loss(pred_fake, zeros_target(N))
    err_fake.backward()

    optimizer.step()

    return err_real + err_fake, pred_real, pred_fake

def train_generator(optimizer, fake_data):
    N = fake_data.size(0)
    optimizer.zero_grad()

    pred = dis(fake_data)
    err = loss(pred, ones_target(N))
    err.backward()

    optimizer.step()

    return err

dis = DiscriminatorNet()
gen = GeneratorNet()
if torch.cuda.is_available():
    dis.cuda()
    gen.cuda()

d_opt = optim.Adam(dis.parameters(), lr=0.0002)
g_opt = optim.Adam(gen.parameters(), lr=0.0002)
loss = nn.BCELoss()

logger = Logger(model_name='VGAN', data_name='MNIST')

data = mnist_data()
data_loader = torch.utils.data.DataLoader(data, batch_size=100, shuffle=True)
num_batches = len(data_loader)

epochs = 200

num_test_samples = 16
test_noise = gen_noise(num_test_samples)

for epoch in range(epochs):
    for n_batch, (real_batch, _) in enumerate(data_loader):
        N = real_batch.size(0)

        real_data = Variable(im2vec(real_batch))
        if torch.cuda.is_available():
            real_data = real_data.cuda()

        fake_data = gen(gen_noise(N)).detach()
        d_err, d_pred_real, d_pred_fake = train_discriminator(d_opt, real_data, fake_data)

        fake_data = gen(gen_noise(N))
        g_err = train_generator(g_opt, fake_data)

        logger.log(d_err, g_err, epoch, n_batch, num_batches)

        if n_batch % 100 == 0: 
            test_images = vec2im(gen(test_noise))
            test_images = test_images.data
            logger.log_images(
                test_images, num_test_samples, 
                epoch, n_batch, num_batches
            )

            logger.display_status(
                epoch, epochs, n_batch, num_batches,
                d_err, g_err, d_pred_real, d_pred_fake
            )

        logger.save_models(gen, dis, epoch)

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
