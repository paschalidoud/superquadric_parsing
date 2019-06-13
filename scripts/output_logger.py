from progress.bar import Bar


def chamfer_loss_logger(current_epoch, epochs, steps_per_epoch):
    bar = Bar(
        "Epoch %d/%d" % (current_epoch, epochs),
        suffix=("%(index)d/%(max)d - loss: %(loss).7f - "
                "pcl_to_prim: %(pcl_to_prim_loss).7f - "
                "prim_to_pcl: %(prim_to_pcl_loss).7f - "
                # "bern_reg: %(bernoulli_regularizer).8f "
                # "entr_bern_reg: %(entropy_bernoulli_regularizer).8f "
                # "sp_reg: %(sparsity_regularizer).8f "
                # "overl_reg: %(overlapping_regularizer).8f "
                # "parsimony_reg: %(parsimony_regularizer).8f "
                "exp_n_prims: %(exp_n_prims).4f"
                ""),
        max=steps_per_epoch,
        loss=0.0,
        pcl_to_prim_loss=0.0, prim_to_pcl_loss=0.0,
        sparsity_regularizer=0.0,
        entropy_bernoulli_regularizer=0.0,
        bernoulli_regularizer=0.0,
        parsimony_regularizer=0.0,
        overlapping_regularizer=0.0,
        exp_n_prims=0.0
    )
    bar.hide_cursor = False
    return bar


def get_logger(loss_type, current_epoch, epochs, steps_per_epoch):
    return {
        "euclidean_dual_loss": chamfer_loss_logger
    }[loss_type](current_epoch, epochs, steps_per_epoch)
