from pytorch_lightning.callbacks import Callback

class SaveCallback(Callback):

    def __init__(self, dirpath, filename):
        self.dirpath = dirpath
        self.filename = filename

    def on_train_end(self, trainer, pl_module):
        model = trainer.model
        model.save_model(self.dirpath,self.filename)

class AutoregressiveSaveCallback(Callback):
    def __init__(self, dirpath):
        self.dirpath = dirpath

    def on_train_end(self, trainer, pl_module) -> None:
        model = trainer.model.model
        embedding_model = trainer.model.embedding_model
        model.save_model(self.dirpath)
        embedding_model.save_model(self.dirpath, 'embedding')
