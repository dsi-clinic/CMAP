# Model Evaluation Results: baseline 2024 Fall 

## Overall Accuracy
- **Training Jaccard Index (IoU):** 0.5844 (std:0.042)
- **Test Jaccard Index (IoU):** 0.5130 (std:0.053)

## Class-Specific Accuracy (IoU):
| Class                | IoU Value  | Accuracy (%) |
|----------------------|------------|--------------|
| BACKGROUND           | 0.000000   | 0%           |
| POND                 | 0.682153   | 68.2%        |
| WETLAND              | 0.447889   | 44.8%        |
| DRY BOTTOM - TURF    | 0.511011   | 51.1%        |
| DRY BOTTOM - MESIC   | 0.165727   | 16.6%        |

## Loss Values:
- **Training Loss:** 0.209 
- **Test Loss:** 0.983615

## Examples:
- **False positive:** (output/example_images/CMAP_false_positive_example.png)
- **False negative:** (output/example_images/CMAP_false_negative_example.png)


