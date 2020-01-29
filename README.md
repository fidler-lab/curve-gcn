# Curve-GCN

This is the official PyTorch implementation of Curve-GCN (CVPR 2019). This repository provides the dataloader ([Cityscapes-Hard](https://github.com/fidler-lab/curve-gcn/tree/dataloader#code-structure)) we used in our paper. For comparisons, we also provide the Cityscapes-Stretch, which is compatiable with [DEXTR](https://github.com/scaelles/DEXTR-PyTorch) and [DELSE](https://github.com/fidler-lab/delse). For technical details, please refer to:  
----------------------- ------------------------------------
**Fast Interactive Object Annotation with Curve-GCN**  
[Huan Ling](http:///www.cs.toronto.edu/~linghuan/)\* <sup>1,2</sup>, [Jun Gao](http://www.cs.toronto.edu/~jungao/)\* <sup>1,2</sup>, [Amlan Kar](http://www.cs.toronto.edu/~amlan/)<sup>1,2</sup>, [Wenzheng Chen](http://www.cs.toronto.edu/~wenzheng/)<sup>1,2</sup>, [Sanja Fidler](http://www.cs.toronto.edu/~fidler/)<sup>1,2,3</sup>   
<sup>1</sup> University of Toronto  <sup>2</sup> Vector Institute <sup>3</sup> NVIDIA  
**[[Paper](https://arxiv.org/pdf/1903.06874.pdf)] [[Video](https://youtu.be/ycD2BtO-QzU)] [[Demo Coming Soon]()] [[Supplementary](http://www.cs.toronto.edu/~linghuan/notes/supplementary_curvegcn.pdf)]**

**CVPR 2019**

<img src = "docs/model.png" width="56%"/>

*Manually labeling objects by tracing their boundaries is
a laborious process. In Polyrnn, the authors proposed Polygon-
RNN that produces polygonal annotations in a recurrent
manner using a CNN-RNN architecture, allowing interactive
correction via humans-in-the-loop. We propose a new framework
that alleviates the sequential nature of Polygon-RNN,
by predicting all vertices simultaneously using a Graph Convolutional
Network (GCN). Our model is trained end-to-end,
and runs in real time. It supports object annotation by either
polygons or splines, facilitating labeling efficiency for both
line-based and curved objects. We show that Curve-GCN outperforms
all existing approaches in automatic mode, including
the powerful PSP-DeepLab and is significantly
more efficient in interactive mode than Polygon-RNN++.
Our model runs at 29.3ms in automatic, and 2.6ms in interactive
mode, making it 10x and 100x faster than Polygon-
RNN++.*  
(\* denotes equal contribution)    
----------------------- ------------------------------------



# Where is the code?
To get the full code, please [signup](http://www.cs.toronto.edu/annotation/curvegcn/code_signup/) here. If you use this code, please cite:

    @inproceedings{CurveGCN2019,
    title={Fast Interactive Object Annotation with Curve-GCN},
    author={Huan Ling and Jun Gao and Amlan Kar and Wenzheng Chen and Sanja Fidler},
    booktitle={CVPR},
    year={2019}
    }
    @inproceedings{AcunaCVPR18,
	title={Efficient Interactive Annotation of Segmentation Datasets with Polygon-RNN++},
	author={David Acuna and Huan Ling and Amlan Kar and Sanja Fidler},
	booktitle={CVPR},
	year={2018}
	}
	@inproceedings{CastrejonCVPR17,
	title = {Annotating Object Instances with a Polygon-RNN},
	author = {Lluis Castrejon and Kaustav Kundu and Raquel Urtasun and Sanja Fidler},
	booktitle = {CVPR},
	year = {2017}
	}


# License

This work is licensed under a *GNU GENERAL PUBLIC LICENSE Version 3* License.

# Code Structure
```
.
├── Dataloaders
│   ├── cityscapes_processed_hard.py # Cityscapes Hard Dataloader
│   ├── cityscapes_processed_stretch.py # Cityscapes Stretch Dataloader
│   ├── custom_transforms.py # Customs transform functions for dataloader
│   └── helper.py   # Helper functions for dataloader
├── Evaluation
│   └── metrics.py # Metrics for evaluation
├── Experiments
│   └── gnn-active-spline.json  # Experiment json files for Cityscapes Hard Dataloader
├── Scripts
│   ├── data
│   │   └── change_paths.py   # Change the data path to the desired format.
│   ├── get_score_boundary_F.py # Scripts to get boundary F scores.
│   ├── get_scores.py  # Scripts to get mIOU scores.
│   ├── hard_loader.py # Instantiate Cityscapes Hard Dataloader
│   └── stretch_loader.py # Instantiate Cityscapes Stretch Dataloader
```


## Data 

### Cityscapes
- Our dataloaders work with our processed annotation files which can be downloaded from [here](http://www.cs.toronto.edu/~amlan/data/polygon/cityscapes.tar.gz).
- We also refer to the original Cityscapes dataset (leftImg8bit\_trainvaltest.zip) from the official [website](https://www.cityscapes-dataset.com/downloads/) [11 GB].
- From the root directory, run the following command with appropriate paths to get the annotation files ready for your machine:
```
python Scripts/data/change_paths.py --city_dir <path_to_downloaded_leftImg8bit_folder> --json_dir <path_to_downloaded_annotation_file> --out_dir <output_dir>
```

## Evaluation
Calculate mIOU:  
```
python Scripts/get_scores.py --pred <path to output dir> --output  <path to output txt file>
```

Calculate F scores:  
```
python Scripts/get_score_boundary_F.py --pred <path to output dir> --output  <path to output txt file>
```
