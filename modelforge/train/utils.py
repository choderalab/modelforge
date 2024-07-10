def shared_config_prior():
    from modelforge.utils.io import check_import

    check_import(
        "ray"
    )  # check that ray is installed before trying to import submodules
    from ray import tune

    return {
        "lr": tune.loguniform(1e-5, 1e-1),
        "weight_decay": tune.loguniform(1e-5, 1e-1),
        "batch_size": tune.choice([32, 64, 128, 256, 512]),
    }
