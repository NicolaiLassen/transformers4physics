from pytorch_lightning.callbacks import Callback


class SaveCallback(Callback):

    def __init__(self, dirpath, filename):
        self.dirpath = dirpath
        self.filename = filename

    def on_train_end(self, trainer, pl_module):
        model = trainer.model
        model.save_model(self.dirpath,self.filename)