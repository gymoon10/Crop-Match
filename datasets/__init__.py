
from datasets.CornDataset import CornDataset
from datasets.CornDatasetCrop import CornDatasetCrop
from datasets.CityscapesDataset import CityscapesDataset


def get_dataset(name, dataset_opts):
    if name == "CornDataset":
        return CornDataset(**dataset_opts)

    elif name == "CornDatasetCrop":
        return CornDatasetCrop(**dataset_opts)

    elif name == "CityscapesDataset":
        return CityscapesDataset(**dataset_opts)

    else:
        raise RuntimeError("Dataset {} not available".format(name))