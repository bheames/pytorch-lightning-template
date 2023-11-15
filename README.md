# PyTorch Lightning template

A template for PyTorch Lightning projects.

### Usage

1) Install requirements:
```bash
pip install -r dev/requirements.txt
```

2) Train a model:
```bash
python dev/train.py
```

3) View results (including default `tensorboard` logs):
```bash
tree -a lightning_logs/
lightning_logs/
└── version_0
    ├── checkpoints
    │   └── epoch=2-step=7500.ckpt
    ├── events.out.tfevents.0
    ├── events.out.tfevents.1
    └── hparams.yaml
```

### Development

1) Install pre-commit hooks:
```bash
pip install pre-commit
pre-commit install --install-hooks
pre-commit install -t commit-msg
```

2) Run tests:
```bash
pip install tox
tox
```
