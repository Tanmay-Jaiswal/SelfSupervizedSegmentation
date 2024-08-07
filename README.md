# Self Supervised Segmentation
This project explores how self-supervised learning can be 
used to augment the performance of the segmentation model
This is accomplished by first pre-training the model on 
the image reconstruction task on a large unannotated 
dataset similar to the smaller annotated dataset.

We use UNet, an encoder-decoder model to perform 
segmentation.

The pretraining set is a large set of medical cell slides that are not annotated in any way. This data is generally easier to come by.
The fine-tuning set is a smaller set of medical cell slides that are annotated by experts to segment cancerous cells differently from non-cancerous cells.

The baseline model uses the the small dataset by itself.
The finetuned model is first pretrained on image reconstruction on a large number of slides, then finetuned on the annotated datset.

## Requirements -
PyTorch - GPU
Numpy
Matplotlib
argparse

## Getting started

The following shows the basic folder structure.
```
├── data
│   ├── test_data
│   │   ├── patches
│   │   └── masks
│   └── train_data 
│       └── patches
├── train_segmentation.py # testing code
├── pretrain.py # training code
├── model.py
├── torch_dataset.py
├── train_utils.py
├── snapshots
```
You can choose to place the snapshots and 
data else were and pass the appropriate parameters
for `snapshots_folder` and `datadir` while running the 
script.

The `test_data` folder  must contain mask annotations while
the `train_data` can have unannotated files.

### Run Pretraining
```bash
python pretrain.py
```

It should be as simple as that assuming that your data 
is organized as recommended.  
If not, you can pass optional parameters -
```
--datadir
--num_epochs
--train_batch_size
--val_batch_size
--num_workers
--print_freq
--snapshot_iter
--snapshots_folder
--load_pretrain
--pretrain_weight_path
```

### Fine-tuning

```bash
python train_segmentation.py --pretrain_weight_path "snapshots_folder/pretrain-19.pt"
```
Additionally, you can pass the following arguments
```
--datadir
--num_epochs
--train_batch_size
--val_batch_size
--num_workers
--print_freq
--snapshot_iter
--snapshots_folder
--load_pretrain
--pretrain_weight_path
```
