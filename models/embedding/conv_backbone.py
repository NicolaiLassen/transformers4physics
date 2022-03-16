from models.embedding.embedding_backbone import EmbeddingBackbone


class ConvBackbone(EmbeddingBackbone):
    def __init__(
        self,
        peg_kernel_size = 3,
        dropout = 0.
    ):
        super().__init__()

    def observableNet(self, x):
        return self.layers(x)

    def recoveryNet(self, x):
         return self.layers(x)

    def forward(self, x):
        raise NotImplementedError("Parameterized AE only has up and down")
