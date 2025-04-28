import wandb


class WandbLogger:
    def __init__(self, project: str, run_name: str, args) -> None:
        self.run = wandb.init(project=project, name=run_name, config=vars(args))
        self._step = 0

    def log(self, **kwargs) -> "WandbLogger":
        self.run.log(kwargs, step=self._step)
        return self

    def step(self) -> "WandbLogger":
        self._step += 1
        return self
