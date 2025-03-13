from mmseg.registry import DATASETS
from .basesegdataset import BaseSegDataset


@DATASETS.register_module()
class MS2D(BaseSegDataset):
    """MS2D Segmentation Dataset.

    This dataset is designed for semantic segmentation tasks where the 
    segmentation map annotation defines two classes: 'background' and 
    'organoid'. In the segmentation map:
    - `0` represents the background.
    - `1` represents the organoid.

    The dataset uses:
    - `img_suffix`: Fixed to '.jpg', indicating the format of input images.
    - `seg_map_suffix`: Fixed to '.tif', indicating the format of segmentation maps.
    - `reduce_zero_label`: A boolean that determines whether to reduce the zero label in the segmentation map. 
      By default, it is set to False, meaning the background class is included.

    The parameter `reduce_zero_label` can be set to True if you want to ignore the background class during evaluation.
    """
    METAINFO = dict(
        classes=('background', 'organoid'),
        palette=[[0], [1]])  # Color palette for visualization

    def __init__(self,
                 img_suffix='.jpg',  # Image file suffix
                 seg_map_suffix='.tif',  # Segmentation map file suffix
                 reduce_zero_label=False,  # Whether to ignore the background class
                 **kwargs) -> None:
        super().__init__(
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            reduce_zero_label=reduce_zero_label,
            **kwargs)