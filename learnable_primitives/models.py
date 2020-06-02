import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import models


class NetworkParameters(object):
    def __init__(self, architecture, n_primitives=32,
                 mu=0.0, sigma=0.001, add_gaussian_noise=False,
                 use_sq=False, make_dense=False,
                 use_deformations=False,
                 train_with_bernoulli=False):
        self.architecture = architecture
        self.n_primitives = n_primitives
        self.train_with_bernoulli = train_with_bernoulli
        self.add_gaussian_noise = add_gaussian_noise
        self.gaussian_noise_layer = get_gaussian_noise_layer(
            self.add_gaussian_noise,
            mu,
            sigma
        )
        self.use_sq = use_sq
        self.use_deformations = use_deformations
        self.make_dense = make_dense

    @classmethod
    def from_options(cls, argument_parser):
        # Make Namespace to dictionary to be able to use it
        args = vars(argument_parser)

        architecture = args["architecture"]
        n_primitives = args.get("n_primitives", 32)

        add_gaussian_noise = args.get("add_gaussian_noise", False)
        mu = args.get("mu", 0.0)
        sigma = args.get("sigma", 0.001)

        # By default train without learning Bernoulli priors
        train_with_bernoulli = args.get("train_with_bernoulli", False)
        use_sq = args.get("use_sq", False)
        use_deformations = args.get("use_deformations", False)
        make_dense = args.get("make_dense", False)

        return cls(
            architecture,
            n_primitives=n_primitives,
            mu=mu,
            sigma=sigma,
            add_gaussian_noise=add_gaussian_noise,
            use_sq=use_sq,
            use_deformations=use_deformations,
            train_with_bernoulli=train_with_bernoulli,
            make_dense=make_dense
        )

    @property
    def network(self):
        networks = dict(
            tulsiani=TulsianiNetwork,
            octnet=OctnetNetwork,
            resnet18=ResNet18
        )

        return networks[self.architecture.lower()]

    def primitive_layer(self, n_primitives, input_channels):
        modules = self._build_modules(n_primitives, input_channels)
        module = GeometricPrimitive(n_primitives, modules)
        return module

    def _build_modules(self, n_primitives, input_channels):
        modules = {
            "translations": Translation(n_primitives, input_channels, self.make_dense),
            "rotations": Rotation(n_primitives, input_channels, self.make_dense),
            "sizes": Size(n_primitives, input_channels, self.make_dense)
        }
        if self.train_with_bernoulli:
            modules["probs"] = Probability(n_primitives, input_channels, self.make_dense)
        if self.use_sq and not self.use_deformations:
            modules["shapes"] = Shape(n_primitives, input_channels, self.make_dense)
        if self.use_sq and self.use_deformations:
            modules["shapes"] = Shape(n_primitives, input_channels, self.make_dense)
            modules["deformations"] = Deformation(
                n_primitives, input_channels, self.make_dense)

        return modules


class TulsianiNetwork(nn.Module):
    def __init__(self, network_params):
        super(TulsianiNetwork, self).__init__()
        self._network_params = network_params

        # Initialize some useful variables
        n_filters = 4
        input_channels = 1

        encoder_layers = []
        # Create an encoder using a stack of convolutions
        for i in range(5):
            encoder_layers.append(
                nn.Conv3d(input_channels, n_filters, kernel_size=3, padding=1)
            )
            encoder_layers.append(nn.BatchNorm3d(n_filters))
            encoder_layers.append(nn.LeakyReLU(0.2, True))
            encoder_layers.append(nn.MaxPool3d(kernel_size=2, stride=2))

            input_channels = n_filters
            # Double the number of filters after every layer
            n_filters *= 2

        # Add the two fully connected layers
        input_channels = n_filters / 2
        n_filters = 100
        for i in range(2):
            encoder_layers.append(nn.Conv3d(input_channels, n_filters, 1))
            # encoder_layers.append(nn.BatchNorm3d(n_filters))
            encoder_layers.append(nn.LeakyReLU(0.2, True))

            input_channels = n_filters

        self._features_extractor = nn.Sequential(*encoder_layers)
        self._primitive_layer = self._network_params.primitive_layer(
            self._network_params.n_primitives,
            n_filters
        )

    def forward(self, X):
        x = self._features_extractor(X)
        return self._primitive_layer(x)


class OctnetNetwork(nn.Module):
    def __init__(self, network_params):
        super(OctnetNetwork, self).__init__()

        self.encoder_conv = nn.Sequential(
            nn.Conv3d(1, 8, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv3d(8, 8, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv3d(8, 8, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2),

            nn.Conv3d(8, 16, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv3d(16, 16, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv3d(16, 16, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2),

            nn.Conv3d(16, 32, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv3d(32, 32, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv3d(32, 32, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2),

            nn.Conv3d(32, 64, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv3d(64, 64, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv3d(64, 64, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2),
        )
        self.encoder_fc = nn.Sequential(
            nn.Linear(2*2*2*64, 1024), nn.ReLU(),
            # nn.Dropout(0.5),
            nn.Linear(1024, 1024), nn.ReLU()
        )
        self.primitive_layer = network_params.primitive_layer(
            network_params.n_primitives,
            1024
        )

    def forward(self, X):
        X = self.encoder_conv(X)
        X = self.encoder_fc(X.view(-1, 2*2*2*64))
        return self.primitive_layer(X.view(-1, 1024, 1, 1, 1))


class ResNet18(nn.Module):
    def __init__(self, network_params):
        super(ResNet18, self).__init__()
        self._network_params = network_params

        self._features_extractor = models.resnet18(pretrained=True)
        self._features_extractor.fc = nn.Sequential(
            nn.Linear(512, 512), nn.ReLU(),
            nn.Linear(512, 512), nn.ReLU()
        )
        self._features_extractor.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self._primitive_layer = self._network_params.primitive_layer(
            self._network_params.n_primitives,
            512
        )

    def forward(self, X):
        X = X.float() / 255.0
        x = self._features_extractor(X)
        return self._primitive_layer(x.view(-1, 512, 1, 1, 1))


class Translation(nn.Module):
    """A layer that predicts the translation vector
    """
    def __init__(self, n_primitives, input_channels, make_dense=False):
        super(Translation, self).__init__()
        self._n_primitives = n_primitives

        self._make_dense = make_dense
        if self._make_dense:
            self._fc = nn.Conv3d(input_channels, input_channels, 1)
            self._nonlin = nn.LeakyReLU(0.2, True)

        # Layer used to infer the translation vector of each primitive, namely
        # BxMx3
        self._translation_layer = nn.Conv3d(
            input_channels, self._n_primitives*3, 1
        )

    def forward(self, X):
        if self._make_dense:
            X = self._nonlin(self._fc(X))

        # Compute the BxM*3 translation vectors for every primitive and ensure
        # that they lie inside the unit cube
        translations = torch.tanh(self._translation_layer(X)) * 0.51

        return translations[:, :, 0, 0, 0]


class Rotation(nn.Module):
    """A layer that predicts the rotation vector
    """
    def __init__(self, n_primitives, input_channels, make_dense=False):
        super(Rotation, self).__init__()
        self._n_primitives = n_primitives

        self._make_dense = make_dense
        if self._make_dense:
            self._fc = nn.Conv3d(input_channels, input_channels, 1)
            self._nonlin = nn.LeakyReLU(0.2, True)

        # Layer used to infer the 4 quaternions of each primitive, namely
        # BxMx4
        self._rotation_layer = nn.Conv3d(
            input_channels, self._n_primitives*4, 1
        )

    def forward(self, X):
        if self._make_dense:
            X = self._nonlin(self._fc(X))

        # Compute the 4 parameters of the quaternion for every primitive
        # and add a non-linearity as L2-normalization to enforce the unit
        # norm constrain
        quats = self._rotation_layer(X)[:, :, 0, 0, 0]
        quats = quats.view(-1, self._n_primitives, 4)
        rotations = quats / torch.norm(quats, 2, -1, keepdim=True)
        rotations = rotations.view(-1, self._n_primitives*4)

        return rotations


class Size(nn.Module):
    """A layer that predicts the size vector
    """
    def __init__(self, n_primitives, input_channels, make_dense=False):
        super(Size, self).__init__()
        self._n_primitives = n_primitives

        self._make_dense = make_dense
        if self._make_dense:
            self._fc = nn.Conv3d(input_channels, input_channels, 1)
            self._nonlin = nn.LeakyReLU(0.2, True)

        # Layer used to infer the size of each primitive, along each axis,
        # namely BxMx3.
        self._size_layer = nn.Conv3d(
            input_channels, self._n_primitives*3, 1
        )

    def forward(self, X):
        if self._make_dense:
            X = self._nonlin(self._fc(X))

        # Bound the sizes so that they won't take values larger than 0.51 and
        # smaller than 1e-2 (to avoid numerical instabilities with the
        # inside-outside function)
        sizes = torch.sigmoid(self._size_layer(X)) * 0.5 + 0.03
        sizes = sizes[:, :, 0, 0, 0]

        return sizes


class Shape(nn.Module):
    """A layer that predicts the shape vector
    """
    def __init__(self, n_primitives, input_channels, make_dense=False):
        super(Shape, self).__init__()
        self._n_primitives = n_primitives

        self._make_dense = make_dense
        if self._make_dense:
            self._fc = nn.Conv3d(input_channels, input_channels, 1)
            self._nonlin = nn.LeakyReLU(0.2, True)

        # Layer used to infer the shape of each primitive, along each axis,
        # namely BxMx3.
        self._shape_layer = nn.Conv3d(
            input_channels, self._n_primitives*2, 1
        )

    def forward(self, X):
        if self._make_dense:
            X = self._nonlin(self._fc(X))

        # Bound the predicted shapes to avoid numerical instabilities with
        # the inside-outside function
        shapes = torch.sigmoid(self._shape_layer(X))*1.1 + 0.4
        shapes = shapes[:, :, 0, 0, 0]

        return shapes


class Deformation(nn.Module):
    """A layer that predicts the deformations
    """
    def __init__(self, n_primitives, input_channels, make_dense=False):
        super(Deformation, self).__init__()
        self._n_primitives = n_primitives

        self._make_dense = make_dense
        if self._make_dense:
            self._fc = nn.Conv3d(input_channels, input_channels, 1)
            self._nonlin = nn.LeakyReLU(0.2, True)

        # Layer used to infer the tapering parameters of each primitive.
        self._tapering_layer =\
            nn.Conv3d(input_channels, self._n_primitives*2, 1)

    def forward(self, X):
        if self._make_dense:
            X = self._nonlin(self._fc(X))

        # The tapering parameters are from -1 to 1
        taperings = torch.tanh(self._tapering_layer(X))*0.9
        taperings = taperings[:, :, 0, 0, 0]

        return taperings


class Probability(nn.Module):
    """A layer that predicts the probabilities
    """
    def __init__(self, n_primitives, input_channels, make_dense=False):
        super(Probability, self).__init__()
        self._n_primitives = n_primitives

        self._make_dense = make_dense
        if self._make_dense:
            self._fc = nn.Conv3d(input_channels, input_channels, 1)
            self._nonlin = nn.LeakyReLU(0.2, True)

        # Layer used to infer the probability of existence for the M
        # primitives, namely BxM numbers, where B is the batch size
        self._probability_layer = nn.Conv3d(
            input_channels, self._n_primitives, 1
        )

    def forward(self, X):
        if self._make_dense:
            X = self._nonlin(self._fc(X))

        # Compute the BxM probabilities of existence for the M primitives and
        # remove unwanted axis with size 1
        probs = torch.sigmoid(
           self._probability_layer(X)
        ).view(-1, self._n_primitives)

        return probs


class PrimitiveParameters(object):
    """Represents the \lambda_m."""
    def __init__(self, probs, translations, rotations, sizes, shapes,
                 deformations):
        self.probs = probs
        self.translations = translations
        self.rotations = rotations
        self.sizes = sizes
        self.shapes = shapes
        self.deformations = deformations

        # Check that everything has a len(shape) > 1
        for x in self.members[:-2]:
            assert len(x.shape) > 1

    def __getattr__(self, name):
        if not name.endswith("_r"):
            raise AttributeError()

        prop = getattr(self, name[:-2])
        if not torch.is_tensor(prop):
            raise AttributeError()

        return prop.view(self.batch_size, self.n_primitives, -1)

    @property
    def members(self):
        return (
            self.probs,
            self.translations,
            self.rotations,
            self.sizes,
            self.shapes,
            self.deformations
        )

    @property
    def batch_size(self):
        return self.probs.shape[0]

    @property
    def n_primitives(self):
        return self.probs.shape[1]

    def __len__(self):
        return len(self.members)

    def __getitem__(self, i):
        return self.members[i]


class GeometricPrimitive(nn.Module):
    def __init__(self, n_primitives, primitive_params):
        super(GeometricPrimitive, self).__init__()
        self._n_primitives = n_primitives
        self._primitive_params = primitive_params

        self._update_params()

    def _update_params(self):
        for i, m in enumerate(self._primitive_params.values()):
            self.add_module("layer%d" % (i,), m)

    def forward(self, X):
        if "probs" not in self._primitive_params.keys():
            probs = X.new_ones((X.shape[0], self._n_primitives))
        else:
            probs = self._primitive_params["probs"].forward(X)

        translations = self._primitive_params["translations"].forward(X)
        rotations = self._primitive_params["rotations"].forward(X)
        sizes = self._primitive_params["sizes"].forward(X)

        # By default the geometric primitive is a cuboid
        if "shapes" not in self._primitive_params.keys():
            shapes = X.new_ones((X.shape[0], self._n_primitives*2)) * 0.25
        else:
            shapes = self._primitive_params["shapes"].forward(X)

        if "deformations" not in self._primitive_params.keys():
            deformations = X.new_zeros((X.shape[0], self._n_primitives*2))
        else:
            deformations = self._primitive_params["deformations"].forward(X)

        return PrimitiveParameters(
            probs, translations, rotations, sizes,
            shapes, deformations
        )


class GaussianNoise(nn.Module):
    def __init__(self, mu=0.0, sigma=0.01):
        super(GaussianNoise, self).__init__()
        # Mean of the distribution
        self.mu = mu
        # Standard deviation of the distribution
        self.sigma = sigma

    def forward(self, X):
        if self.training and self.sigma != 0:
            n = X.new_zeros(*X.size()).normal_(self.mu, self.sigma)
            X = X + n
        return X


def train_on_batch(
    model,
    optimizer,
    loss_fn,
    X,
    y_target,
    regularizer_terms,
    sq_sampler,
    loss_options
):
    # Zero the gradient's buffer
    optimizer.zero_grad()
    y_hat = model(X)
    loss, debug_stats = loss_fn(
        y_hat,
        y_target,
        regularizer_terms,
        sq_sampler,
        loss_options
    )
    # Do the backpropagation
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), 1)
    # Do the update
    optimizer.step()

    return (
        loss.item(),
        [x.data if hasattr(x, "data") else x for x in y_hat],
        debug_stats
    )


def get_gaussian_noise_layer(add_gaussian_noise, mu=0.0, sigma=0.01):
    if add_gaussian_noise:
        return GaussianNoise(mu=mu, sigma=sigma)
    else:
        return GaussianNoise(mu=0.0, sigma=0.0)


def optimizer_factory(args, model):
    """Based on the input arguments create a suitable optimizer object
    """
    if args.probs_only:
        params = model._primitive_layer._primitive_params["probs"].parameters()
    else:
        params = model.parameters()

    if args.optimizer == "SGD":
        return optim.SGD(
            params,
            lr=args.lr,
            momentum=args.momentum
        )
    elif args.optimizer == "Adam":
        return optim.Adam(
            params,
            lr=args.lr
        )
