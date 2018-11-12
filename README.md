# Learning Pose Specific Representations by Predicting Different Views
This repository contains the code for the semi-supervised method we proposed in:  

[**Learning Pose Specific Representations by Predicting Different Views**](https://arxiv.org/abs/1804.03390)  
Georg Poier, David Schinagl and Horst Bischof.  
In *Proc. CVPR*, 2018. ([Project Page](https://poier.github.io/PreView/)).

---

![Sketch for learning a pose specific representation from unlabeled data](./doc/idea_sketch.png)  
We learn to predict a low-dimensional latent representation and, subsequently,
a different view of the input, 
*solely* from the latent representation.
The error of the view prediction is used as feedback,
enforcing the latent representation to capture pose specific information
without requiring labeled data.


## Usage
1. Download dataset
2. Adapt paths in configuration to point to the dataset
3. Run code

### Download dataset
We provide data-loaders for two datasets: 
(i) the NYU dataset [[1]](https://cims.nyu.edu/~tompson/NYU_Hand_Pose_Dataset.htm), and 
(ii) the MV-hands dataset [[2]](https://files.icg.tugraz.at/f/a190309bd4474ec2b13f/) 
published together with the paper.

### Adapt configuration 
You need to change the respective paths in `config/config_data_nyu.py` for the NYU dataset, 
or `config/config_data_icg.py` for the MV-hands/ICG dataset, resp.
For the MV-hands data you also need to change to the corresponding configuration 
by uncommenting the following line in `main_run.py`:

```python
from config.config_data_icg import args_data
```

### Run code

    python main_run.py

It will log the training and validation error using crayon 
(see [https://github.com/torrvision/crayon](https://github.com/torrvision/crayon)),
and output intermediate images and final results in the `results` folder.
When using the MV-hands dataset you need to change the camera view, which 
is to be predicted, by adding `--output-cam-ids-train 2` to the call.
To change further settings you can adapt the respective configuration files 
in the `config` folder or via the command-line 
(see `python main_run.py --help` for details). 
The default settings should be fine 
to reproduce the results from the paper, however.

#### Training/Testing speed
In our case, loading of the data is the bottleneck. 
Hence, it's very beneficial if the data is stored on a disk with fast access times (e.g., SSD).
Several workers are concurrently loading (and pre-processing) data samples.
The number of workers can be changed by adjusting `args.num_loader_workers` 
in `config/config.py`.

#### Faster training/testing on NYU dataset
We use binary files to speed up training/testing for the NYU dataset. 
The binary files can be loaded faster, which will usually yield a significant 
speed up for training and testing. 

To make use of the binary files, you need to set `args_data.use_pickled_cache = True` 
in `config/config_data_nyu.py`. Then, the binary files are used instead of the original images. 
If a binary file for an image does not exist already it is automatically 
written the first time the image should be loaded. 
Hence, the process will be slower the first time training/testing is done with 
`args_data.use_pickled_cache = True`.

To ensure that all binary files will be properly written, 
it's probably the best/easiest to remove the 
`WeightedRandomSampler` for a single epoch the first time you use the binary cache files.
To do so, e.g., just comment out the `sampler` keyword argument at the 
creation of the `DataLoader` in `data/LoaderFactory.py`, 
train for one epoch (e.g., using command-line parameter `--epochs 1`), 
and uncomment the `sampler` again. 
Currently, the `sampler` creation can be found in the lines 97-99 of `data/LoaderFactory.py`.
(And/Or use only a single worker to load the data 
using `args.num_loader_workers` in `config/config.py`.)
Note, this process is not always necessary but prevents possible issues during 
creation of the binary files.

#### Train with adversarial loss
For training with the additional adversarial loss just change the training type
using the corresponding command-line parameter. 
That is, call `python main_run.py --training-type 2` instead.
However, note that with this additional loss we merely obtained similar results 
for the cost of additional training time (see the paper for details).

#### Use pre-trained model
In `./source/results` you find a model pre-trained on the NYU dataset.
You can generate results using this one by calling:

    python main_run.py --model-filepath </path/to/model.mdl> --no-train


## Requirements
We used Python 2.7.
To run the code you can, e.g., install the following requirements:

 * [PyTorch](http://pytorch.org/) (0.3.1; torch, torchvision)
 * enum34
 * matplotlib
 * scipy
 * [pycrayon](https://github.com/torrvision/crayon)

### pycrayon
The code sends the data to port 8889 of "localhost". 
That is, you could start the server exactly as in the usage example in the 
[crayon README](https://github.com/torrvision/crayon/blob/master/README.md) 
(i.e., by calling `docker run -d -p 8888:8888 -p 8889:8889 --name crayon alband/crayon`).
See [https://github.com/torrvision/crayon](https://github.com/torrvision/crayon) 
for details.


## Citation
If you can make use of this work, please cite:

Learning Pose Specific Representations by Predicting Different Views.  
Georg Poier, David Schinagl and Horst Bischof.  
In *Proc. CVPR*, 2018.

Bibtex:
```
@inproceedings{Poier2018cvpr_preview,  
  author = {Georg Poier and David Schinagl and Horst Bischof},  
  title = {Learning Pose Specific Representations by Predicting Different Views},  
  booktitle = {{Proc. IEEE/CVF Conf. on Computer Vision and Pattern Recognition (CVPR)}},  
  year = {2018}
}
```

## References
\[1] [https://cims.nyu.edu/~tompson/NYU\_Hand\_Pose\_Dataset.htm](https://cims.nyu.edu/~tompson/NYU_Hand_Pose_Dataset.htm)  
\[2] [https://files.icg.tugraz.at/f/a190309bd4474ec2b13f/](https://files.icg.tugraz.at/f/a190309bd4474ec2b13f/)  

