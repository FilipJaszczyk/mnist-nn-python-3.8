from loader import Dataloader
from simple_network import Network

dl = Dataloader("./data/mnist.pkl.gz")
train, validate, test = dl.load()
net = Network([784, 30, 10])
net.SGD(train, 30, 10, 3.0, test_data=test)