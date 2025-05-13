from .builder import DATASETS
# from .custom import CustomDataset
from .fcis_custom import FCISCustomDataset


@DATASETS.register_module()
class FCISDSBDataset(FCISCustomDataset):
    """LiverCell dataset Nuclei segmentation dataset."""

    CLASSES = ('background', 'nuclei')

    PALETTE = [[0, 0, 0], [255, 2, 255]]

    def __init__(self, **kwargs):
        super().__init__(img_suffix='.png', sem_suffix='.png', inst_suffix='.npy', adj_suffix='.yaml', **kwargs)
