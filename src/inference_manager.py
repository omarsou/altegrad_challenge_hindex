from src.dataloader import Batch, make_data_iter


class TestManager:
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
        self.model = model

    def predict(self, test_data):
        test_iter = make_data_iter(
            test_data,
            batch_size=self.batch_size,
            train=False,
            shuffle=False)
        self.model.eval()
        y_pred = list()
        authors = list()

        for batch in iter(test_iter):
            batch = Batch(
                is_train=False,
                torch_batch=batch,
                use_cuda=self.use_cuda,
                author_dim=self.author_dim,
                paper_dim=self.paper_dim
            )

            y_pred_batch = self.predict_batch(batch)

            y_pred_batch = y_pred_batch.detach().cpu().numpy()
            authors_match = batch.author
            y_pred.append(y_pred_batch)
            authors.append(authors_match)
        return y_pred, authors

    def predict_batch(self, batch):
        y_pred = self.model.forward(batch)
        return y_pred