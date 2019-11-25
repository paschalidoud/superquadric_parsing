
def add_voxelizer_parameters(parser):
    parser.add_argument(
        "--voxelizer_factory",
        choices=[
            "occupancy_grid",
            "tsdf_grid",
            "image"
        ],
        default="occupancy_grid",
        help="The voxelizer factory to be used (default=occupancy_grid)"
    )

    parser.add_argument(
        "--grid_shape",
        type=lambda x: tuple(map(int, x.split(","))),
        default="32,32,32",
        help="The dimensionality of the voxel grid (default=(32, 32, 32)"
    )
    parser.add_argument(
        "--save_voxels_to",
        default=None,
        help="Path to save the voxelised input to the network"
    )
    parser.add_argument(
        "--image_shape",
        type=lambda x: tuple(map(int, x.split(","))),
        default="3,137,137",
        help="The dimensionality of the voxel grid (default=(3,137,137)"
    )


def add_training_parameters(parser):
    """Add arguments to a parser that are related with the training of the
    network.
    """
    parser.add_argument(
        "--epochs",
        type=int,
        default=150,
        help="Number of times to iterate over the dataset (default=150)"
    )
    parser.add_argument(
        "--steps_per_epoch",
        type=int,
        default=500,
        help=("Total number of steps (batches of samples) before declaring one"
              " epoch finished and starting the next epoch (default=500)")
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Number of samples in a batch (default=32)"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate (default 1e-3)"
    )
    parser.add_argument(
        "--lr_epochs",
        type=lambda x: list(map(int, x.split(","))),
        default="500,1000,1500",
        help="Training epochs with diminishing learning rate"
    )
    parser.add_argument(
        "--lr_factor",
        type=float,
        default=1.0,
        help=("Factor according to which the learning rate will be diminished"
              " (default=None)")
    )
    parser.add_argument(
        "--optimizer",
        choices=["Adam", "SGD"],
        default="Adam",
        help="The optimizer to be used (default=Adam)"
    )
    parser.add_argument(
        "--momentum",
        type=float,
        default=0.9,
        help=("Parameter used to update momentum in case of SGD optimizer"
              " (default=0.9)")
    )


def add_dataset_parameters(parser):
    parser.add_argument(
        "--dataset_type",
        default="shapenet_quad",
        choices=[
            "shapenet_quad",
            "shapenet_v1",
            "shapenet_v2",
            "surreal_bodies",
            "dynamic_faust"
        ],
        help="The type of the dataset type to be used"
    )
    parser.add_argument(
        "--n_points_from_mesh",
        type=int,
        default=1000,
        help="The maximum number of points sampled from mesh (default=1000)"
    )
    parser.add_argument(
        "--model_tags",
        type=lambda x: x.split(":"),
        default=[],
        help="The tags to the model to be used for testing",
    )


def add_nn_parameters(parser):
    """Add arguments to control the design of the neural network architecture.
    """
    parser.add_argument(
        "--architecture",
        choices=["tulsiani", "octnet", "resnet18"],
        default="tulsiani",
        help="Choose the architecture to train"
    )
    parser.add_argument(
        "--train_with_bernoulli",
        action="store_true",
        help="Learn the Bernoulli priors during training"
    )
    parser.add_argument(
        "--make_dense",
        action="store_true",
        help="When true use an additional FC before its regressor"
    )


def add_regularizer_parameters(parser):
    parser.add_argument(
        "--regularizer_type",
        choices=[
            "bernoulli_regularizer",
            "entropy_bernoulli_regularizer",
            "parsimony_regularizer",
            "overlapping_regularizer",
            "sparsity_regularizer"
        ],
        nargs="+",
        default=[],
        help=("The type of the regularizer on the shapes to be used"
              " (default=None)")

    )
    parser.add_argument(
        "--bernoulli_regularizer_weight",
        type=float,
        default=0.0,
        help=("The importance of the regularization term on Bernoulli priors"
              " (default=0.0)")
    )
    parser.add_argument(
        "--maximum_number_of_primitives",
        type=int,
        default=5000,
        help=("The maximum number of primitives in the predicted shape "
              " (default=5000)")
    )
    parser.add_argument(
        "--minimum_number_of_primitives",
        type=int,
        default=5,
        help=("The minimum number of primitives in the predicted shape "
              " (default=5)")
    )
    parser.add_argument(
        "--entropy_bernoulli_regularizer_weight",
        type=float,
        default=0.0,
        help=("The importance of the regularizer term on the entropy of"
              " the bernoullis (default=0.0)")
    )
    parser.add_argument(
        "--sparsity_regularizer_weight",
        type=float,
        default=0.0,
        help="The weight on the sparsity regularizer (default=0.0)"
    )
    parser.add_argument(
        "--parsimony_regularizer_weight",
        type=float,
        default=0.0,
        help="The weight on the parsimony regularizer (default=0.0)"
    )
    parser.add_argument(
        "--overlapping_regularizer_weight",
        type=float,
        default=0.0,
        help="The weight on the overlapping regularizer (default=0.0)"
    )
    parser.add_argument(
        "--enable_regularizer_after_epoch",
        type=int,
        default=0,
        help="Epoch after which regularizer is enabled (default=10)"
    )
    parser.add_argument(
        "--w1",
        type=float,
        default=0.005,
        help="The weight on the first term of the sparsity regularizer (default=0.005)"
    )
    parser.add_argument(
        "--w2",
        type=float,
        default=0.005,
        help="The weight on the second term of the sparsity regularizer (default=0.005)"
    )


def add_sq_mesh_sampler_parameters(parser):
    parser.add_argument(
        "--D_eta",
        type=float,
        default=0.05,
        help="Step along the eta (default=0.05)"
    )
    parser.add_argument(
        "--D_omega",
        type=float,
        default=0.05,
        help="Step along the omega (default=0.05)"
    )
    parser.add_argument(
        "--n_points_from_sq_mesh",
        type=int,
        default=180,
        help="Number of points to sample from the mesh of the SQ (default=180)"
    )


def add_gaussian_noise_layer_parameters(parser):
    parser.add_argument(
        "--add_gaussian_noise",
        action="store_true",
        help="Add Gaussian noise in the layers"
    )
    parser.add_argument(
        "--mu",
        type=float,
        default=0.0,
        help="Mean value of the Gaussian distribution"
    )
    parser.add_argument(
        "--sigma",
        type=float,
        default=0.001,
        help="Standard deviation of the Gaussian distribution"
    )


def add_loss_parameters(parser):
    parser.add_argument(
        "--prim_to_pcl_loss_weight",
        default=1.0,
        type=float,
        help=("The importance of the primitive-to-pointcloud loss in the "
              "final loss (default = 1.0)")
    )
    parser.add_argument(
        "--pcl_to_prim_loss_weight",
        default=1.0,
        type=float,
        help=("The importance of the pointcloud-to-primitive loss in the "
              "final loss (default = 1.0)")
    )


def add_loss_options_parameters(parser):
    parser.add_argument(
        "--use_sq",
        action="store_true",
        help="Use Superquadrics as geometric primitives"
    )
    parser.add_argument(
        "--use_cuboids",
        action="store_true",
        help="Use cuboids as geometric primitives"
    )
    parser.add_argument(
        "--use_chamfer",
        action="store_true",
        help="Use the chamfer distance"
    )


def voxelizer_shape(args):
    if args.voxelizer_factory == "occupancy_grid":
        return args.grid_shape
    elif args.voxelizer_factory == "image":
        return args.image_shape
    elif args.voxelizer_factory == "tsdf_grid":
        return (args.resolution,)*3


def get_loss_weights(args):
    args = vars(args)
    loss_weights = {
        "pcl_to_prim_weight": args.get("pcl_to_prim_loss_weight", 1.0),
        "prim_to_pcl_weight": args.get("prim_to_pcl_loss_weight", 1.0),
    }

    return loss_weights


def get_loss_options(args):
    loss_weights = get_loss_weights(args)

    args = vars(args)
    # Create a dicitionary with the loss options based on the input arguments
    loss_options = {
        "use_sq": args.get("use_sq", False),
        "use_cuboids": args.get("use_cuboids", False),
        "use_chamfer": args.get("use_chamfer", False),
        "loss_weights": loss_weights
    }

    return loss_options
