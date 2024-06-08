# This is an example script that trains an implemented model on the QM9 dataset.
# tensorboard --logdir tb_logs


if __name__ == "__main__":

    import fire
    from modelforge.train.training import read_config_and_train

    fire.Fire(read_config_and_train)
