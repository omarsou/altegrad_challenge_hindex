import time
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from src.dataloader import make_data_iter, Batch


def validate_on_data(model, data, batch_size, use_cuda, author_dim, paper_dim):
    valid_iter = make_data_iter(
        dataset=data,
        batch_size=batch_size,
        shuffle=False,
        train=False)

    model.eval()
    with torch.no_grad():
        total_rmse = 0.0
        total_mae = 0.0
        for valid_batch in iter(valid_iter):
            batch = Batch(
                is_train=True,
                torch_batch=valid_batch,
                use_cuda=use_cuda,
                author_dim=author_dim,
                paper_dim=paper_dim)
            mse, rmse, mae = model.get_metrics_for_batch(batch=batch)
            total_rmse += rmse
            total_mae += mae

    return total_rmse / len(valid_iter), total_mae / len(valid_iter)


class TrainManager:
    def __init__(self, model, config):
        self.model_dir = config.model_dir
        self.epochs = config.epochs
        self.author_dim = config.author_dim
        self.paper_dim = config.paper_dim
        self.use_cuda = config.use_cuda
        self.batch_size = config.batch_size
        self.eval_batch_size = config.batch_size
        self.validation_freq = config.validation_freq
        self.training_freq = config.training_freq
        self.use_cuda = config.use_cuda
        self.name_model = config.name_model
        self.model = model
        self.batch_multiplier = 1
        self.steps = 1
        self.optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', patience=config.patience, factor=config.factor
                                           , verbose=True)
        self.new_best = float('inf')
        self.is_best = (lambda score: score < self.new_best)

    def _save_checkpoint(self) -> None:
        model_path = "{}/{}.ckpt".format(self.model_dir, self.name_model)
        state = {
            "steps": self.steps,
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": self.scheduler.state_dict()
            if self.scheduler is not None
            else None,
        }
        torch.save(state, model_path)

    def train_and_validate(self, train_data, valid_data):
        train_iter = make_data_iter(
            train_data,
            batch_size=self.batch_size,
            train=True,
            shuffle=True)

        for epoch_no in range(self.epochs):

            self.model.train()
            update = True
            total_steps = self.epochs * len(train_iter)
            start = time.time()
            tot_rmse, tot_mae, i = 0, 0, 0

            for batch in iter(train_iter):
                i += 1
                batch = Batch(
                    is_train=True,
                    torch_batch=batch,
                    use_cuda=self.use_cuda,
                    author_dim=self.author_dim,
                    paper_dim=self.paper_dim
                )

                hindex_loss, rmse, mae = self._train_batch(
                    batch)
                tot_rmse += rmse
                tot_mae += mae

                if self.steps % self.training_freq == 0:
                    print(
                        f"Time elapsed : {round(time.time() - start, 3)}, steps : {self.steps}/{total_steps} "
                        f"Train_RMSE : {tot_rmse / i} // Train_MAE : {tot_mae / i}")

                # validate on the entire dev set
                if self.steps % self.validation_freq == 0 and update:
                    val_rmse, val_mae = validate_on_data(
                        model=self.model,
                        data=valid_data,
                        batch_size=self.eval_batch_size,
                        use_cuda=self.use_cuda,
                        author_dim=self.author_dim,
                        paper_dim=self.paper_dim,
                    )
                    self.scheduler.step(val_mae)
                    self.model.train()

                    if self.is_best(val_mae):
                        self.new_best = val_mae
                        print("Yes! New best validation result!")
                        print(f"Val_RMSE : {val_rmse} // Val_MAE : {val_mae}")
                        self._save_checkpoint()
            print(f"Epoch {epoch_no}/{self.epochs} , Time : {round(time.time() - start, 3)}")

    def _train_batch(self, batch):
        hindex_loss, rmse, mae = self.model.get_metrics_for_batch(
            batch=batch)
        hindex_loss.backward()

        # make gradient step
        self.optimizer.step()
        self.optimizer.zero_grad()

        # increment step counter
        self.steps += 1

        return hindex_loss, rmse, mae