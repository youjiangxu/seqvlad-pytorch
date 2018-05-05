# SeqVLAD-Pytorch

*Now in experimental release, suggestions welcome*.

This is an implementation of Sequential VLAD (SeqVLAD) in PyTorch.

**Note**: always use `git clone --recursive https://github.com/youjiangxu/seqvlad-pytorch` to clone this project.
Otherwise you will not be able to use the inception series CNN archs.


## Training

To train a new model, use the `main.py` script.

The command to reproduce the original SeqVLAD experiments of RGB modality on UCF101 can be

```bash
split=1
timesteps=25
num_centers=64
lr=0.02
dropout=0.8

first_step=80
second_step=150
total_epoch=210
two_steps=120
optim=SGD
prefix=ucf101_rgb_split${split}
python ./main.py ucf101 RGB ./data/ucf101_splits/rgb/train_split${split}.txt ./data/ucf101_splits/rgb/test_split${split}.txt \
      --arch BNInception \
      --timesteps ${timesteps} --num_centers ${num_centers} --redu_dim 512 \
      --gd 20 --lr ${lr} --lr_steps ${first_step} ${second_step} --epochs ${total_epoch} \
      -b 64 -j 8 --dropout ${dropout} \
      --snapshot_pref ./models/rgb/${prefix} \
      --sources <path to source rgb frames of ucf101> \
      --two_steps ${two_steps} \
      --activation softmax \
      --optim ${optim}
```

For flow models:

```bash
split=1
timesteps=25
num_centers=64
lr=0.01
dropout=0.7

first_step=90
second_step=180
third_step=210
total_epoch=240
two_steps=120

optim=SGD

prefix=ucf101_flow_split${split}

python /mnt/lustre/xuyoujiang/action/seqvlad-pytorch/main.py ucf101 Flow ./data/ucf101_splits/flow/train_split${split}.txt ./data/ucf101_splits/flow/test_split${split}.txt \
   --arch BNInception \
   --timesteps ${timesteps} --num_centers 64 --redu_dim 512 \
   --gd 20 --lr ${lr} --lr_steps ${first_step} ${second_step} --epochs ${total_epoch} \
   -b 64 -j 8 --dropout ${dropout} \
   --snapshot_pref ./models/flow/${prefix} \
   --sources <path to source optical frame of ucf101> \
   --resume <path to tsn flow pretrained model> \
   --resume_type tsn --two_steps ${two_steps} \
   --activation softmax \
   --flow_pref flow_ \
   --optim ${optim}
```

## Testing

After training, there will checkpoints saved by pytorch, for example `ucf101_rgb_split1_checkpoint.pth`.

Use the following command to test its performance in the standard TSN testing protocol:

```bash
python test_models.py ucf101 RGB ./data/ucf101_splits/rgb/test_split${split}.txt \
       ucf101_rgb_split1_checkpoint.pth \
       --arch BNInception \
       --save_scores seqvlad_split1_rgb_scores \
       --num_centers 64 \
       --timesteps 25 \
       --redu_dim 512 \
       --sources <path to source rgb frames of ucf101> \
       --activation softmax \
       --test_segments 1
```

Or for flow models:

```bash   
python test_models.py ucf101 Flow ./data/ucf101_splits/flow/test_split${split}.txt \
       ucf101_flow_split1_checkpoint.pth \
       --arch BNInception \
       --save_scores seqvlad_split1_flow_scores \
       --num_centers 64 \
       --timesteps 25 \
       --redu_dim 512 \
       --sources <path to source optical frames of ucf101> \
       --activation softmax \
       --test_segments 1 \
       --flow_pref flow_
```

**Note**: We first build our SeqVLAD on the repository of [old-seqvlad-pytorch](https://github.com/youjiangxu/tsn-pytorch/tree/seqvlad), which is folked from [tsn-pytorch](https://github.com/yjxiong/tsn-pytorch). And in order to reproduct our method easily, we release the source code in this repository [SeqVLAD-Pytorch](https://github.com/youjiangxu/seqvlad-pytorch).

### Useful Links
- [tsn-pytorch](https://github.com/yjxiong/tsn-pytorch)
- [tsn-caffe](https://github.com/yjxiong/temporal-segment-networks)
