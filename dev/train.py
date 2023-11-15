import pytorch_lightning as pl

import src


def main(config: src.config.Config) -> None:
    trainer = pl.Trainer(
        max_epochs=config.max_epochs,
        log_every_n_steps=config.log_every_n_steps,
        limit_train_batches=config.limit_train_batches,
    )
    model = src.model.LitModel()

    train_loader = src.model.DataLoader(
        data_dir=config.data_dir,
        mode=src.model.Mode.TRAIN,
        batch_size=config.batch_size,
    )
    trainer.fit(model=model, train_dataloaders=train_loader)

    test_loader = src.model.DataLoader(
        data_dir=config.data_dir,
        mode=src.model.Mode.TEST,
        batch_size=config.batch_size,
    )
    trainer.test(dataloaders=test_loader)


if __name__ == "__main__":
    main(src.config.Config())
