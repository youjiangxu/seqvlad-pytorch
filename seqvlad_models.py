from torch import nn

from ops.basic_ops import ConsensusModule, Identity, SeqVLADModule, BiSeqVLADModule, UnshareBiSeqVLADModule
from ops.basic_ops1 import SeqVLADUniformModule
from transforms import *
from torch.nn.init import normal, constant


import collections
class SeqVLAD(nn.Module):
    def __init__(self, num_class, num_centers, modality,
                
                 timesteps=1, redu_dim=512,with_relu=False,
                 base_model='resnet101', new_length=None,
                 consensus_type='avg', before_softmax=True,
                 dropout=0.8,
                 activation=None,
                 seqvlad_type='seqvlad',
                 init_method='xavier_normal',
                 crop_num=1, partial_bn=True):

        super(SeqVLAD, self).__init__()
        self.modality = modality
        self.num_centers = num_centers
        self.reshape = True
        self.before_softmax = before_softmax
        self.dropout = dropout
        self.crop_num = crop_num
        self.with_relu = with_relu
        self.seqvlad_type = seqvlad_type
        self.consensus_type = consensus_type
        self.init_method = init_method
        if not before_softmax and consensus_type != 'avg':
            raise ValueError("Only avg consensus can be used after Softmax")

        if new_length is None:
            self.new_length = 1 if modality == "RGB" else 5
        else:
            self.new_length = new_length


        self.timesteps = timesteps
        self.redu_dim = redu_dim
        self.activation = activation
        print('self.activation,',self.activation)
        print(("""
Initializing SeqVLAD with base model: {}.
SeqVLAD Configurations:
    input_modality:     {}
    num_centers:       {}
    new_length:         {}
    consensus_module:   {}
    dropout_ratio:      {}
    init_method:        {}
        """.format(base_model, self.modality, self.num_centers, self.new_length, consensus_type, self.dropout, self.init_method)))

        self._prepare_base_model(base_model)

        self._add_seqvlad_layer(base_model)

        self._add_classifier_layer(base_model, num_class)
        # print(self.base_model)
        # feature_dim = self._prepare_tsn(num_class)

        if self.modality == 'Flow':
            print("Converting the ImageNet model to a flow init model")
            self.base_model = self._construct_flow_model(self.base_model)
            print("Done. Flow model ready...")
        elif self.modality == 'RGBDiff':
            print("Converting the ImageNet model to RGB+Diff init model")
            self.base_model = self._construct_diff_model(self.base_model)
            print("Done. RGBDiff model ready.")

        # self.consensus = ConsensusModule(consensus_type)

        if not self.before_softmax:
            self.softmax = nn.Softmax()

        self._enable_pbn = partial_bn
        if partial_bn:
            self.partialBN(True)

    def _add_seqvlad_layer(self, base_model):
        if base_model == 'BNInception':
            # print( list(self.base_model.named_children())[:-2])
            #model = nn.Sequential(
            #            *list(self.base_model.children())[:-2]
                        #collections.OrderedDict(list(self.base_model.named_children()))[:-2]
                        #collections.OrderedDict(list(self.base_model.named_modules())[:-2])
                        #*list(self.base_model.named_modules())[:-2]
            #        )
            #model.add_module('SeqVLAD_Module', SeqVLADModule(self.timesteps, self.num_centers, self.redu_dim))
            #self.global_pool = SeqVLADModule(self.timesteps, self.num_centers, self.redu_dim)
            if self.seqvlad_type == 'seqvlad':
                setattr(self.base_model, 'global_pool', 
                        SeqVLADModule(self.timesteps, self.num_centers, self.redu_dim, self.with_relu, self.activation, init_method=self.init_method))
            elif self.seqvlad_type == 'bidirect':
                setattr(self.base_model, 'global_pool', 
                        BiSeqVLADModule(self.timesteps, self.num_centers, self.redu_dim, self.with_relu, self.activation))
            elif self.seqvlad_type == 'unshare_bidirect':
                 setattr(self.base_model, 'global_pool', 
                        UnshareBiSeqVLADModule(self.timesteps, self.num_centers, self.redu_dim, self.with_relu, self.activation))
            #self.base_model = model
        

    def _add_classifier_layer(self, base_model, num_class):
        if base_model == 'BNInception':
            #if self.dropout == 0:
            #    self.base_model.add_module('fc', nn.Linear(self.num_centers*self.redu_dim, num_class))

            #else:
            #    self.base_model.add_module('dropout', nn.Dropout(p=self.dropout))
            #    self.base_model.add_module('fc', nn.Linear(self.num_centers*self.redu_dim, num_class))

            #std = 0.001
            #normal(self.base_model.fc.weight, 0, std)
            #constant(self.base_model.fc.bias, 0)
            feature_dim = self.num_centers*self.redu_dim
            if self.dropout == 0:
                setattr(self.base_model, self.base_model.last_layer_name, nn.Linear(feature_dim, num_class))
                self.new_fc = None
            else:
                setattr(self.base_model, self.base_model.last_layer_name, nn.Dropout(p=self.dropout))
                self.new_fc = nn.Linear(feature_dim, num_class)

            std = 0.001
            if self.new_fc is None:
                normal(getattr(self.base_model, self.base_model.last_layer_name).weight, 0, std)
                constant(getattr(self.base_model, self.base_model.last_layer_name).bias, 0)
            else:
                normal(self.new_fc.weight, 0, std)
                constant(self.new_fc.bias, 0)
            return feature_dim

       
        
    # def _prepare_tsn(self, num_class):
    #     feature_dim = getattr(self.base_model, self.base_model.last_layer_name).in_features
    #     if self.dropout == 0:
    #         setattr(self.base_model, self.base_model.last_layer_name, nn.Linear(feature_dim, num_class))
    #         self.new_fc = None
    #     else:
    #         setattr(self.base_model, self.base_model.last_layer_name, nn.Dropout(p=self.dropout))
    #         self.new_fc = nn.Linear(feature_dim, num_class)

    #     std = 0.001
    #     if self.new_fc is None:
    #         normal(getattr(self.base_model, self.base_model.last_layer_name).weight, 0, std)
    #         constant(getattr(self.base_model, self.base_model.last_layer_name).bias, 0)
    #     else:
    #         normal(self.new_fc.weight, 0, std)
    #         constant(self.new_fc.bias, 0)
    #     return feature_dim

    def _prepare_base_model(self, base_model):

        if 'resnet' in base_model or 'vgg' in base_model:
            self.base_model = getattr(torchvision.models, base_model)(True)
            self.base_model.last_layer_name = 'fc'
            self.input_size = 224
            self.input_mean = [0.485, 0.456, 0.406]
            self.input_std = [0.229, 0.224, 0.225]

            if self.modality == 'Flow':
                self.input_mean = [0.5]
                self.input_std = [np.mean(self.input_std)]
            elif self.modality == 'RGBDiff':
                self.input_mean = [0.485, 0.456, 0.406] + [0] * 3 * self.new_length
                self.input_std = self.input_std + [np.mean(self.input_std) * 2] * 3 * self.new_length
        elif base_model == 'BNInception':
            import tf_model_zoo
            self.base_model = getattr(tf_model_zoo, base_model)()
            self.base_model.last_layer_name = 'fc'
            self.input_size = 224
            self.input_mean = [104, 117, 128]
            self.input_std = [1]

            if self.modality == 'Flow':
                self.input_mean = [128]
            elif self.modality == 'RGBDiff':
                self.input_mean = self.input_mean * (1 + self.new_length)

        elif 'inception' in base_model:
            import tf_model_zoo
            self.base_model = getattr(tf_model_zoo, base_model)()
            self.base_model.last_layer_name = 'classif'
            self.input_size = 299
            self.input_mean = [0.5]
            self.input_std = [0.5]
        else:
            raise ValueError('Unknown base model: {}'.format(base_model))

    def train(self, mode=True):
        """
        Override the default train() to freeze the BN parameters
        :return:
        """
        super(SeqVLAD, self).train(mode)
        count = 0
        if self._enable_pbn:
            print("Freezing BatchNorm2D except the first one.")
            for m in self.base_model.modules():
                if isinstance(m, nn.BatchNorm2d):
                    count += 1
                    if count >= (2 if self._enable_pbn else 1):
                        m.eval()

                        # shutdown update in frozen mode
                        m.weight.requires_grad = False
                        m.bias.requires_grad = False

    def partialBN(self, enable):
        self._enable_pbn = enable
    def get_sub_optim_policies(self):
        first_conv_weight = []
        first_conv_weight = []
        first_conv_bias = []
        normal_weight = []
        normal_bias = []
        bn = []

        conv_cnt = 0
        bn_cnt = 0
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Conv1d):
                #ps = list(m.parameters())
                #conv_cnt += 1
                #if conv_cnt == 1:
                #    first_conv_weight.append(ps[0])
                #    if len(ps) == 2:
                #        first_conv_bias.append(ps[1])
                #else:
                #    normal_weight.append(ps[0])
                
                #    if len(ps) == 2:
                #        normal_bias.append(ps[1])
                pass
            elif isinstance(m, torch.nn.Linear):
                ps = list(m.parameters())
                normal_weight.append(ps[0])
                if len(ps) == 2:
                    normal_bias.append(ps[1])
            elif isinstance(m, torch.nn.BatchNorm1d):
                #bn.extend(list(m.parameters()))
                pass
            elif isinstance(m, torch.nn.BatchNorm2d):
                ##bn_cnt += 1
                ## later BN's are frozen
                #if not self._enable_pbn or bn_cnt == 1:
                #    bn.extend(list(m.parameters()))
                pass
            elif isinstance(m, SeqVLADModule) or isinstance(m, BiSeqVLADModule):
                print('this is SeqVlad module, and adding the trainable parameters to train')
                assert len(list(m.parameters())) == 8, "the number parameters of seqvlad should be equal to 8"
                ps = list(m.parameters())
                conv_cnt += 5
                normal_weight.extend(ps[0:5])
                normal_bias.extend(ps[5::])
                print('len of weight %d' %(len(ps[0:5])))
                print('len of bias %d' %(len(ps[5::])))
            elif isinstance(m, UnshareBiSeqVLADModule):
                print('this is unshare SeqVlad module, and adding the trainable parameters to train')
                assert len(list(m.parameters())) == 13, "the number parameters of seqvlad should be equal to 13"
                ps = list(m.parameters())
                conv_cnt += 9
                normal_weight.extend(ps[0:9])
                normal_bias.extend(ps[9::])
                print('len of weight %d' %(len(ps[0:9])))
                print('len of bias %d' %(len(ps[9::])))

            elif len(m._modules) == 0:
                if len(list(m.parameters())) > 0:
                    raise ValueError("New atomic module type: {}. Need to give it a learning policy".format(type(m)))


        return [
            {'params': first_conv_weight, 'lr_mult': 5 if self.modality == 'Flow' else 1, 'decay_mult': 1,
             'name': "first_conv_weight"},
            {'params': first_conv_bias, 'lr_mult': 10 if self.modality == 'Flow' else 2, 'decay_mult': 0,
             'name': "first_conv_bias"},
            {'params': normal_weight, 'lr_mult': 1, 'decay_mult': 1,
             'name': "normal_weight"},
            {'params': normal_bias, 'lr_mult': 2, 'decay_mult': 0,
             'name': "normal_bias"},
            {'params': bn, 'lr_mult': 1, 'decay_mult': 0,
             'name': "BN scale/shift"},
        ]
    def get_optim_policies(self):
        first_conv_weight = []
        first_conv_bias = []
        normal_weight = []
        normal_bias = []
        bn = []

        conv_cnt = 0
        bn_cnt = 0
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Conv1d):
                ps = list(m.parameters())
                conv_cnt += 1
                if conv_cnt == 1:
                    first_conv_weight.append(ps[0])
                    if len(ps) == 2:
                        first_conv_bias.append(ps[1])
                else:
                    normal_weight.append(ps[0])
                    if len(ps) == 2:
                        normal_bias.append(ps[1])
            elif isinstance(m, torch.nn.Linear):
                ps = list(m.parameters())
                normal_weight.append(ps[0])
                if len(ps) == 2:
                    normal_bias.append(ps[1])
                  
            elif isinstance(m, torch.nn.BatchNorm1d):
                bn.extend(list(m.parameters()))
            elif isinstance(m, torch.nn.BatchNorm2d):
                bn_cnt += 1
                # later BN's are frozen
                if not self._enable_pbn or bn_cnt == 1:
                    bn.extend(list(m.parameters()))
            elif isinstance(m, SeqVLADModule) or isinstance(m, BiSeqVLADModule):
                print('this is SeqVlad module, and adding the trainable parameters to train')
                assert len(list(m.parameters())) == 8, "the number parameters of seqvlad should be equal to 8"
                ps = list(m.parameters())
                conv_cnt += 5
                normal_weight.extend(ps[0:5])
                normal_bias.extend(ps[5::])
                print('len of weight %d' %(len(ps[0:5])))
                print('len of bias %d' %(len(ps[5::])))

            elif isinstance(m, UnshareBiSeqVLADModule):
                print('this is unshare SeqVlad module, and adding the trainable parameters to train')
                assert len(list(m.parameters())) == 13, "the number parameters of seqvlad should be equal to 13"
                ps = list(m.parameters())
                conv_cnt += 9
                normal_weight.extend(ps[0:9])
                normal_bias.extend(ps[9::])
                print('len of weight %d' %(len(ps[0:9])))
                print('len of bias %d' %(len(ps[9::])))

            elif len(m._modules) == 0:
                if len(list(m.parameters())) > 0:
                    raise ValueError("New atomic module type: {}. Need to give it a learning policy".format(type(m)))
           

        return [
            {'params': first_conv_weight, 'lr_mult': 5 if self.modality == 'Flow' else 1, 'decay_mult': 1,
             'name': "first_conv_weight"},
            {'params': first_conv_bias, 'lr_mult': 10 if self.modality == 'Flow' else 2, 'decay_mult': 0,
             'name': "first_conv_bias"},
            {'params': normal_weight, 'lr_mult': 1, 'decay_mult': 1,
             'name': "normal_weight"},
            {'params': normal_bias, 'lr_mult': 2, 'decay_mult': 0,
             'name': "normal_bias"},
            {'params': bn, 'lr_mult': 1, 'decay_mult': 0,
             'name': "BN scale/shift"},
        ]

    def forward(self, input):
        sample_len = (3 if self.modality == "RGB" else 2) * self.new_length
        if self.modality == 'RGBDiff':
            sample_len = 3 * self.new_length
            input = self._get_diff(input)

        # print('shape', input.view((-1, sample_len) + input.size()[-2:]).size())
        # print('type', type(input.view((-1, sample_len) + input.size()[-2:])))

        base_out = self.base_model(input.view((-1, sample_len) + input.size()[-2:]))

        if self.dropout > 0:
            base_out = self.new_fc(base_out)

        if not self.before_softmax:
            base_out = self.softmax(base_out)
        # if self.reshape:
        #     self.in_shape = input.size()
        #     self.batch_size = self.in_shape[0]//self.timesteps
        #     base_out = base_out.view((self.batch_size) + base_out.size()[1:])

        output = base_out
        # print('seqvlad output:', output.size)
        return output.squeeze(1)

    def _get_diff(self, input, keep_rgb=False):
        input_c = 3 if self.modality in ["RGB", "RGBDiff"] else 2
        input_view = input.view((-1, self.num_segments, self.new_length + 1, input_c,) + input.size()[2:])
        if keep_rgb:
            new_data = input_view.clone()
        else:
            new_data = input_view[:, :, 1:, :, :, :].clone()

        for x in reversed(list(range(1, self.new_length + 1))):
            if keep_rgb:
                new_data[:, :, x, :, :, :] = input_view[:, :, x, :, :, :] - input_view[:, :, x - 1, :, :, :]
            else:
                new_data[:, :, x - 1, :, :, :] = input_view[:, :, x, :, :, :] - input_view[:, :, x - 1, :, :, :]

        return new_data


    def _construct_flow_model(self, base_model):
        # modify the convolution layers
        # Torch models are usually defined in a hierarchical way.
        # nn.modules.children() return all sub modules in a DFS manner
        modules = list(self.base_model.modules())
        first_conv_idx = list(filter(lambda x: isinstance(modules[x], nn.Conv2d), list(range(len(modules)))))[0]
        conv_layer = modules[first_conv_idx]
        container = modules[first_conv_idx - 1]

        # modify parameters, assume the first blob contains the convolution kernels
        params = [x.clone() for x in conv_layer.parameters()]
        kernel_size = params[0].size()
        new_kernel_size = kernel_size[:1] + (2 * self.new_length, ) + kernel_size[2:]
        new_kernels = params[0].data.mean(dim=1, keepdim=True).expand(new_kernel_size).contiguous()

        new_conv = nn.Conv2d(2 * self.new_length, conv_layer.out_channels,
                             conv_layer.kernel_size, conv_layer.stride, conv_layer.padding,
                             bias=True if len(params) == 2 else False)
        new_conv.weight.data = new_kernels
        if len(params) == 2:
            new_conv.bias.data = params[1].data # add bias if neccessary
        layer_name = list(container.state_dict().keys())[0][:-7] # remove .weight suffix to get the layer name

        # replace the first convlution layer
        setattr(container, layer_name, new_conv)
        return base_model

    def _construct_diff_model(self, base_model, keep_rgb=False):
        # modify the convolution layers
        # Torch models are usually defined in a hierarchical way.
        # nn.modules.children() return all sub modules in a DFS manner
        modules = list(self.base_model.modules())
        first_conv_idx = filter(lambda x: isinstance(modules[x], nn.Conv2d), list(range(len(modules))))[0]
        conv_layer = modules[first_conv_idx]
        container = modules[first_conv_idx - 1]

        # modify parameters, assume the first blob contains the convolution kernels
        params = [x.clone() for x in conv_layer.parameters()]
        kernel_size = params[0].size()
        if not keep_rgb:
            new_kernel_size = kernel_size[:1] + (3 * self.new_length,) + kernel_size[2:]
            new_kernels = params[0].data.mean(dim=1, keepdim=True).expand(new_kernel_size).contiguous()
        else:
            new_kernel_size = kernel_size[:1] + (3 * self.new_length,) + kernel_size[2:]
            new_kernels = torch.cat((params[0].data, params[0].data.mean(dim=1, keepdim=True).expand(new_kernel_size).contiguous()),
                                    1)
            new_kernel_size = kernel_size[:1] + (3 + 3 * self.new_length,) + kernel_size[2:]

        new_conv = nn.Conv2d(new_kernel_size[1], conv_layer.out_channels,
                             conv_layer.kernel_size, conv_layer.stride, conv_layer.padding,
                             bias=True if len(params) == 2 else False)
        new_conv.weight.data = new_kernels
        if len(params) == 2:
            new_conv.bias.data = params[1].data  # add bias if neccessary
        layer_name = list(container.state_dict().keys())[0][:-7]  # remove .weight suffix to get the layer name

        # replace the first convolution layer
        setattr(container, layer_name, new_conv)
        return base_model

    @property
    def crop_size(self):
        return self.input_size

    @property
    def scale_size(self):
        return self.input_size * 256 // 224

    def get_augmentation(self):
        if self.modality == 'RGB':
            return torchvision.transforms.Compose([GroupMultiScaleCrop(self.input_size, [1, .875, .75, .66]),
                                                   GroupRandomHorizontalFlip(is_flow=False)])
        elif self.modality == 'Flow':
            return torchvision.transforms.Compose([GroupMultiScaleCrop(self.input_size, [1, .875, .75]),
                                                   GroupRandomHorizontalFlip(is_flow=True)])
        elif self.modality == 'RGBDiff':
            return torchvision.transforms.Compose([GroupMultiScaleCrop(self.input_size, [1, .875, .75]),
                                                   GroupRandomHorizontalFlip(is_flow=False)])




class SeqVLAD_with_conv_centers(nn.Module):
    def __init__(self, num_class, num_centers, modality,
                
                 timesteps=1, redu_dim=512,with_relu=False,
                 base_model='resnet101', new_length=None,
                 consensus_type='avg', before_softmax=True,
                 dropout=0.8,
                 activation=None,
                 crop_num=1, partial_bn=True):
        super(SeqVLAD_with_conv_centers, self).__init__()
        self.modality = modality
        self.num_centers = num_centers
        self.reshape = True
        self.before_softmax = before_softmax
        self.dropout = dropout
        self.crop_num = crop_num
        self.with_relu = with_relu
        self.consensus_type = consensus_type
        if not before_softmax and consensus_type != 'avg':
            raise ValueError("Only avg consensus can be used after Softmax")

        if new_length is None:
            self.new_length = 1 if modality == "RGB" else 5
        else:
            self.new_length = new_length


        self.timesteps = timesteps
        self.redu_dim = redu_dim
        self.activation = activation
        print('self.activation,',self.activation)
        print(("""
Initializing SeqVLAD with base model: {}.
SeqVLAD Configurations:
    input_modality:     {}
    num_centers:       {}
    new_length:         {}
    consensus_module:   {}
    dropout_ratio:      {}
        """.format(base_model, self.modality, self.num_centers, self.new_length, consensus_type, self.dropout)))

        self._prepare_base_model(base_model)
        self._add_seqvlad_layer(base_model)
        self._add_classifier_layer(base_model, num_class)
        # print(self.base_model)
        # feature_dim = self._prepare_tsn(num_class)

        if self.modality == 'Flow':
            print("Converting the ImageNet model to a flow init model")
            self.base_model = self._construct_flow_model(self.base_model)
            print("Done. Flow model ready...")
        elif self.modality == 'RGBDiff':
            print("Converting the ImageNet model to RGB+Diff init model")
            self.base_model = self._construct_diff_model(self.base_model)
            print("Done. RGBDiff model ready.")

        # self.consensus = ConsensusModule(consensus_type)

        if not self.before_softmax:
            self.softmax = nn.Softmax()

        self._enable_pbn = partial_bn
        if partial_bn:
            self.partialBN(True)

    def _add_seqvlad_layer(self, base_model):
        if base_model == 'BNInception':
            # print( list(self.base_model.named_children())[:-2])
            #model = nn.Sequential(
            #            *list(self.base_model.children())[:-2]
                        #collections.OrderedDict(list(self.base_model.named_children()))[:-2]
                        #collections.OrderedDict(list(self.base_model.named_modules())[:-2])
                        #*list(self.base_model.named_modules())[:-2]
            #        )
            #model.add_module('SeqVLAD_Module', SeqVLADModule(self.timesteps, self.num_centers, self.redu_dim))
            #self.global_pool = SeqVLADModule(self.timesteps, self.num_centers, self.redu_dim)
            setattr(self.base_model, 'global_pool', 
                    SeqVLADModule(self.timesteps, self.num_centers, self.redu_dim, self.with_relu, self.activation))
            #self.base_model = model
        

    def _add_classifier_layer(self, base_model, num_class):
        if base_model == 'BNInception':
            #if self.dropout == 0:
            #    self.base_model.add_module('fc', nn.Linear(self.num_centers*self.redu_dim, num_class))

            #else:
            #    self.base_model.add_module('dropout', nn.Dropout(p=self.dropout))
            #    self.base_model.add_module('fc', nn.Linear(self.num_centers*self.redu_dim, num_class))

            #std = 0.001
            #normal(self.base_model.fc.weight, 0, std)
            #constant(self.base_model.fc.bias, 0)
            feature_dim = self.num_centers*self.redu_dim
            if self.dropout == 0:
     	        setattr(self.base_model, self.base_model.last_layer_name, nn.Linear(feature_dim, num_class))
		self.new_fc = None
            else:
                setattr(self.base_model, self.base_model.last_layer_name, nn.Dropout(p=self.dropout))
                self.new_fc = nn.Linear(feature_dim, num_class)

            std = 0.001
            if self.new_fc is None:
                normal(getattr(self.base_model, self.base_model.last_layer_name).weight, 0, std)
                constant(getattr(self.base_model, self.base_model.last_layer_name).bias, 0)
            else:
                normal(self.new_fc.weight, 0, std)
                constant(self.new_fc.bias, 0)
            return feature_dim

       
        
    # def _prepare_tsn(self, num_class):
    #     feature_dim = getattr(self.base_model, self.base_model.last_layer_name).in_features
    #     if self.dropout == 0:
    #         setattr(self.base_model, self.base_model.last_layer_name, nn.Linear(feature_dim, num_class))
    #         self.new_fc = None
    #     else:
    #         setattr(self.base_model, self.base_model.last_layer_name, nn.Dropout(p=self.dropout))
    #         self.new_fc = nn.Linear(feature_dim, num_class)

    #     std = 0.001
    #     if self.new_fc is None:
    #         normal(getattr(self.base_model, self.base_model.last_layer_name).weight, 0, std)
    #         constant(getattr(self.base_model, self.base_model.last_layer_name).bias, 0)
    #     else:
    #         normal(self.new_fc.weight, 0, std)
    #         constant(self.new_fc.bias, 0)
    #     return feature_dim

    def _prepare_base_model(self, base_model):

        if 'resnet' in base_model or 'vgg' in base_model:
            self.base_model = getattr(torchvision.models, base_model)(True)
            self.base_model.last_layer_name = 'fc'
            self.input_size = 224
            self.input_mean = [0.485, 0.456, 0.406]
            self.input_std = [0.229, 0.224, 0.225]

            if self.modality == 'Flow':
                self.input_mean = [0.5]
                self.input_std = [np.mean(self.input_std)]
            elif self.modality == 'RGBDiff':
                self.input_mean = [0.485, 0.456, 0.406] + [0] * 3 * self.new_length
                self.input_std = self.input_std + [np.mean(self.input_std) * 2] * 3 * self.new_length
        elif base_model == 'BNInception':
            import tf_model_zoo
            self.base_model = getattr(tf_model_zoo, base_model)()
            self.base_model.last_layer_name = 'fc'
            self.input_size = 224
            self.input_mean = [104, 117, 128]
            self.input_std = [1]

            if self.modality == 'Flow':
                self.input_mean = [128]
            elif self.modality == 'RGBDiff':
                self.input_mean = self.input_mean * (1 + self.new_length)

        elif 'inception' in base_model:
            import tf_model_zoo
            self.base_model = getattr(tf_model_zoo, base_model)()
            self.base_model.last_layer_name = 'classif'
            self.input_size = 299
            self.input_mean = [0.5]
            self.input_std = [0.5]
        else:
            raise ValueError('Unknown base model: {}'.format(base_model))

    def train(self, mode=True):
        """
        Override the default train() to freeze the BN parameters
        :return:
        """
        super(SeqVLAD_with_conv_centers, self).train(mode)
        count = 0
        if self._enable_pbn:
            print("Freezing BatchNorm2D except the first one.")
            for m in self.base_model.modules():
                if isinstance(m, nn.BatchNorm2d):
                    count += 1
                    if count >= (2 if self._enable_pbn else 1):
                        m.eval()

                        # shutdown update in frozen mode
                        m.weight.requires_grad = False
                        m.bias.requires_grad = False

    def partialBN(self, enable):
        self._enable_pbn = enable
    def get_sub_optim_policies(self):
        first_conv_weight = []
        first_conv_weight = []
        first_conv_bias = []
        normal_weight = []
        normal_bias = []
        bn = []

        conv_cnt = 0
        bn_cnt = 0
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Conv1d):
                #ps = list(m.parameters())
                #conv_cnt += 1
                #if conv_cnt == 1:
                #    first_conv_weight.append(ps[0])
                #    if len(ps) == 2:
                #        first_conv_bias.append(ps[1])
                #else:
                #    normal_weight.append(ps[0])
                
                #    if len(ps) == 2:
                #        normal_bias.append(ps[1])
                pass
            elif isinstance(m, torch.nn.Linear):
                ps = list(m.parameters())
                normal_weight.append(ps[0])
                if len(ps) == 2:
                    normal_bias.append(ps[1])
            elif isinstance(m, torch.nn.BatchNorm1d):
                #bn.extend(list(m.parameters()))
                pass
            elif isinstance(m, torch.nn.BatchNorm2d):
                #bn_cnt += 1
                # later BN's are frozen
                #if not self._enable_pbn or bn_cnt == 1:
                #    bn.extend(list(m.parameters()))
                pass
            elif isinstance(m, SeqVLADModule):
                print('this is SeqVlad module, and adding the trainable parameters to train')
                assert len(list(m.parameters())) == 8, "the number parameters of seqvlad should be equal to 8"
                ps = list(m.parameters())
                conv_cnt += 6
                normal_weight.extend(ps[0:6])
                normal_bias.extend(ps[6::])
                print('len of weight %d' %(len(ps[0:6])))
                print('len of bias %d' %(len(ps[6::])))

            elif len(m._modules) == 0:
                if len(list(m.parameters())) > 0:
                    raise ValueError("New atomic module type: {}. Need to give it a learning policy".format(type(m)))


        return [
            {'params': first_conv_weight, 'lr_mult': 5 if self.modality == 'Flow' else 1, 'decay_mult': 1,
             'name': "first_conv_weight"},
            {'params': first_conv_bias, 'lr_mult': 10 if self.modality == 'Flow' else 2, 'decay_mult': 0,
             'name': "first_conv_bias"},
            {'params': normal_weight, 'lr_mult': 1, 'decay_mult': 1,
             'name': "normal_weight"},
            {'params': normal_bias, 'lr_mult': 2, 'decay_mult': 0,
             'name': "normal_bias"},
            {'params': bn, 'lr_mult': 1, 'decay_mult': 0,
             'name': "BN scale/shift"},
        ]
    def get_optim_policies(self):
        first_conv_weight = []
        first_conv_bias = []
        normal_weight = []
        normal_bias = []
        bn = []

        conv_cnt = 0
        bn_cnt = 0
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Conv1d):
                ps = list(m.parameters())
                conv_cnt += 1
                if conv_cnt == 1:
                    first_conv_weight.append(ps[0])
                    if len(ps) == 2:
                        first_conv_bias.append(ps[1])
                else:
                    normal_weight.append(ps[0])
                    if len(ps) == 2:
                        normal_bias.append(ps[1])
            elif isinstance(m, torch.nn.Linear):
                ps = list(m.parameters())
                normal_weight.append(ps[0])
                if len(ps) == 2:
                    normal_bias.append(ps[1])
                  
            elif isinstance(m, torch.nn.BatchNorm1d):
                bn.extend(list(m.parameters()))
            elif isinstance(m, torch.nn.BatchNorm2d):
                bn_cnt += 1
                # later BN's are frozen
                if not self._enable_pbn or bn_cnt == 1:
                    bn.extend(list(m.parameters()))
            elif isinstance(m, SeqVLADModule):
                print('this is SeqVlad module, and adding the trainable parameters to train')
                assert len(list(m.parameters())) == 8, "the number parameters of seqvlad should be equal to 8"
                ps = list(m.parameters())
                conv_cnt += 6
                normal_weight.extend(ps[0:6])
                normal_bias.extend(ps[6::])
                print('len of weight %d' %(len(ps[0:6])))
                print('len of bias %d' %(len(ps[6::])))

            elif len(m._modules) == 0:
                if len(list(m.parameters())) > 0:
                    raise ValueError("New atomic module type: {}. Need to give it a learning policy".format(type(m)))
           

        return [
            {'params': first_conv_weight, 'lr_mult': 5 if self.modality == 'Flow' else 1, 'decay_mult': 1,
             'name': "first_conv_weight"},
            {'params': first_conv_bias, 'lr_mult': 10 if self.modality == 'Flow' else 2, 'decay_mult': 0,
             'name': "first_conv_bias"},
            {'params': normal_weight, 'lr_mult': 1, 'decay_mult': 1,
             'name': "normal_weight"},
            {'params': normal_bias, 'lr_mult': 2, 'decay_mult': 0,
             'name': "normal_bias"},
            {'params': bn, 'lr_mult': 1, 'decay_mult': 0,
             'name': "BN scale/shift"},
        ]

    def forward(self, input):
        sample_len = (3 if self.modality == "RGB" else 2) * self.new_length
        if self.modality == 'RGBDiff':
            sample_len = 3 * self.new_length
            input = self._get_diff(input)

        # print('shape', input.view((-1, sample_len) + input.size()[-2:]).size())
        # print('type', type(input.view((-1, sample_len) + input.size()[-2:])))

        base_out = self.base_model(input.view((-1, sample_len) + input.size()[-2:]))

        if self.dropout > 0:
            base_out = self.new_fc(base_out)

        if not self.before_softmax:
            base_out = self.softmax(base_out)
        # if self.reshape:
        #     self.in_shape = input.size()
        #     self.batch_size = self.in_shape[0]//self.timesteps
        #     base_out = base_out.view((self.batch_size) + base_out.size()[1:])

        output = base_out
        # print('seqvlad output:', output.size)
        return output.squeeze(1)

    def _get_diff(self, input, keep_rgb=False):
        input_c = 3 if self.modality in ["RGB", "RGBDiff"] else 2
        input_view = input.view((-1, self.num_segments, self.new_length + 1, input_c,) + input.size()[2:])
        if keep_rgb:
            new_data = input_view.clone()
        else:
            new_data = input_view[:, :, 1:, :, :, :].clone()

        for x in reversed(list(range(1, self.new_length + 1))):
            if keep_rgb:
                new_data[:, :, x, :, :, :] = input_view[:, :, x, :, :, :] - input_view[:, :, x - 1, :, :, :]
            else:
                new_data[:, :, x - 1, :, :, :] = input_view[:, :, x, :, :, :] - input_view[:, :, x - 1, :, :, :]

        return new_data


    def _construct_flow_model(self, base_model):
        # modify the convolution layers
        # Torch models are usually defined in a hierarchical way.
        # nn.modules.children() return all sub modules in a DFS manner
        modules = list(self.base_model.modules())
        first_conv_idx = list(filter(lambda x: isinstance(modules[x], nn.Conv2d), list(range(len(modules)))))[0]
        conv_layer = modules[first_conv_idx]
        container = modules[first_conv_idx - 1]

        # modify parameters, assume the first blob contains the convolution kernels
        params = [x.clone() for x in conv_layer.parameters()]
        kernel_size = params[0].size()
        new_kernel_size = kernel_size[:1] + (2 * self.new_length, ) + kernel_size[2:]
        new_kernels = params[0].data.mean(dim=1, keepdim=True).expand(new_kernel_size).contiguous()

        new_conv = nn.Conv2d(2 * self.new_length, conv_layer.out_channels,
                             conv_layer.kernel_size, conv_layer.stride, conv_layer.padding,
                             bias=True if len(params) == 2 else False)
        new_conv.weight.data = new_kernels
        if len(params) == 2:
            new_conv.bias.data = params[1].data # add bias if neccessary
        layer_name = list(container.state_dict().keys())[0][:-7] # remove .weight suffix to get the layer name

        # replace the first convlution layer
        setattr(container, layer_name, new_conv)
        return base_model

    def _construct_diff_model(self, base_model, keep_rgb=False):
        # modify the convolution layers
        # Torch models are usually defined in a hierarchical way.
        # nn.modules.children() return all sub modules in a DFS manner
        modules = list(self.base_model.modules())
        first_conv_idx = filter(lambda x: isinstance(modules[x], nn.Conv2d), list(range(len(modules))))[0]
        conv_layer = modules[first_conv_idx]
        container = modules[first_conv_idx - 1]

        # modify parameters, assume the first blob contains the convolution kernels
        params = [x.clone() for x in conv_layer.parameters()]
        kernel_size = params[0].size()
        if not keep_rgb:
            new_kernel_size = kernel_size[:1] + (3 * self.new_length,) + kernel_size[2:]
            new_kernels = params[0].data.mean(dim=1, keepdim=True).expand(new_kernel_size).contiguous()
        else:
            new_kernel_size = kernel_size[:1] + (3 * self.new_length,) + kernel_size[2:]
            new_kernels = torch.cat((params[0].data, params[0].data.mean(dim=1, keepdim=True).expand(new_kernel_size).contiguous()),
                                    1)
            new_kernel_size = kernel_size[:1] + (3 + 3 * self.new_length,) + kernel_size[2:]

        new_conv = nn.Conv2d(new_kernel_size[1], conv_layer.out_channels,
                             conv_layer.kernel_size, conv_layer.stride, conv_layer.padding,
                             bias=True if len(params) == 2 else False)
        new_conv.weight.data = new_kernels
        if len(params) == 2:
            new_conv.bias.data = params[1].data  # add bias if neccessary
        layer_name = list(container.state_dict().keys())[0][:-7]  # remove .weight suffix to get the layer name

        # replace the first convolution layer
        setattr(container, layer_name, new_conv)
        return base_model

    @property
    def crop_size(self):
        return self.input_size

    @property
    def scale_size(self):
        return self.input_size * 256 // 224

    def get_augmentation(self):
        if self.modality == 'RGB':
            return torchvision.transforms.Compose([GroupMultiScaleCrop(self.input_size, [1, .875, .75, .66]),
                                                   GroupRandomHorizontalFlip(is_flow=False)])
        elif self.modality == 'Flow':
            return torchvision.transforms.Compose([GroupMultiScaleCrop(self.input_size, [1, .875, .75]),
                                                   GroupRandomHorizontalFlip(is_flow=True)])
        elif self.modality == 'RGBDiff':
            return torchvision.transforms.Compose([GroupMultiScaleCrop(self.input_size, [1, .875, .75]),
                                                   GroupRandomHorizontalFlip(is_flow=False)])



class SeqVLAD_with_uniform_centers(nn.Module):
    def __init__(self, num_class, num_centers, modality,
                
                 timesteps=1, redu_dim=512,with_relu=False,
                 base_model='resnet101', new_length=None,
                 consensus_type='avg', before_softmax=True,
                 dropout=0.8,
                 activation=None,
                 crop_num=1, partial_bn=True):
        super(SeqVLAD_with_uniform_centers, self).__init__()
        self.modality = modality
        self.num_centers = num_centers
        self.reshape = True
        self.before_softmax = before_softmax
        self.dropout = dropout
        self.crop_num = crop_num
        self.with_relu = with_relu
        self.consensus_type = consensus_type
        if not before_softmax and consensus_type != 'avg':
            raise ValueError("Only avg consensus can be used after Softmax")

        if new_length is None:
            self.new_length = 1 if modality == "RGB" else 5
        else:
            self.new_length = new_length


        self.timesteps = timesteps
        self.redu_dim = redu_dim
        self.activation = activation
        print('self.activation,',self.activation)
        print(("""
Initializing SeqVLAD with base model: {}.
SeqVLAD Configurations:
    input_modality:     {}
    num_centers:       {}
    new_length:         {}
    consensus_module:   {}
    dropout_ratio:      {}
        """.format(base_model, self.modality, self.num_centers, self.new_length, consensus_type, self.dropout)))

        self._prepare_base_model(base_model)
        self._add_seqvlad_layer(base_model)
        self._add_classifier_layer(base_model, num_class)
        # print(self.base_model)
        # feature_dim = self._prepare_tsn(num_class)

        if self.modality == 'Flow':
            print("Converting the ImageNet model to a flow init model")
            self.base_model = self._construct_flow_model(self.base_model)
            print("Done. Flow model ready...")
        elif self.modality == 'RGBDiff':
            print("Converting the ImageNet model to RGB+Diff init model")
            self.base_model = self._construct_diff_model(self.base_model)
            print("Done. RGBDiff model ready.")

        # self.consensus = ConsensusModule(consensus_type)

        if not self.before_softmax:
            self.softmax = nn.Softmax()

        self._enable_pbn = partial_bn
        if partial_bn:
            self.partialBN(True)

    def _add_seqvlad_layer(self, base_model):
        if base_model == 'BNInception':
            # print( list(self.base_model.named_children())[:-2])
            #model = nn.Sequential(
            #            *list(self.base_model.children())[:-2]
                        #collections.OrderedDict(list(self.base_model.named_children()))[:-2]
                        #collections.OrderedDict(list(self.base_model.named_modules())[:-2])
                        #*list(self.base_model.named_modules())[:-2]
            #        )
            #model.add_module('SeqVLAD_Module', SeqVLADModule(self.timesteps, self.num_centers, self.redu_dim))
            #self.global_pool = SeqVLADModule(self.timesteps, self.num_centers, self.redu_dim)
            setattr(self.base_model, 'global_pool', 
                    SeqVLADUniformModule(self.timesteps, self.num_centers, self.redu_dim, self.with_relu, self.activation))
            #self.base_model = model
        

    def _add_classifier_layer(self, base_model, num_class):
        if base_model == 'BNInception':
            #if self.dropout == 0:
            #    self.base_model.add_module('fc', nn.Linear(self.num_centers*self.redu_dim, num_class))

            #else:
            #    self.base_model.add_module('dropout', nn.Dropout(p=self.dropout))
            #    self.base_model.add_module('fc', nn.Linear(self.num_centers*self.redu_dim, num_class))

            #std = 0.001
            #normal(self.base_model.fc.weight, 0, std)
            #constant(self.base_model.fc.bias, 0)
            feature_dim = self.num_centers*self.redu_dim
            if self.dropout == 0:
     	        setattr(self.base_model, self.base_model.last_layer_name, nn.Linear(feature_dim, num_class))
		self.new_fc = None
            else:
                setattr(self.base_model, self.base_model.last_layer_name, nn.Dropout(p=self.dropout))
                self.new_fc = nn.Linear(feature_dim, num_class)

            std = 0.001
            if self.new_fc is None:
                normal(getattr(self.base_model, self.base_model.last_layer_name).weight, 0, std)
                constant(getattr(self.base_model, self.base_model.last_layer_name).bias, 0)
            else:
                normal(self.new_fc.weight, 0, std)
                constant(self.new_fc.bias, 0)
            return feature_dim

       
        
    # def _prepare_tsn(self, num_class):
    #     feature_dim = getattr(self.base_model, self.base_model.last_layer_name).in_features
    #     if self.dropout == 0:
    #         setattr(self.base_model, self.base_model.last_layer_name, nn.Linear(feature_dim, num_class))
    #         self.new_fc = None
    #     else:
    #         setattr(self.base_model, self.base_model.last_layer_name, nn.Dropout(p=self.dropout))
    #         self.new_fc = nn.Linear(feature_dim, num_class)

    #     std = 0.001
    #     if self.new_fc is None:
    #         normal(getattr(self.base_model, self.base_model.last_layer_name).weight, 0, std)
    #         constant(getattr(self.base_model, self.base_model.last_layer_name).bias, 0)
    #     else:
    #         normal(self.new_fc.weight, 0, std)
    #         constant(self.new_fc.bias, 0)
    #     return feature_dim

    def _prepare_base_model(self, base_model):

        if 'resnet' in base_model or 'vgg' in base_model:
            self.base_model = getattr(torchvision.models, base_model)(True)
            self.base_model.last_layer_name = 'fc'
            self.input_size = 224
            self.input_mean = [0.485, 0.456, 0.406]
            self.input_std = [0.229, 0.224, 0.225]

            if self.modality == 'Flow':
                self.input_mean = [0.5]
                self.input_std = [np.mean(self.input_std)]
            elif self.modality == 'RGBDiff':
                self.input_mean = [0.485, 0.456, 0.406] + [0] * 3 * self.new_length
                self.input_std = self.input_std + [np.mean(self.input_std) * 2] * 3 * self.new_length
        elif base_model == 'BNInception':
            import tf_model_zoo
            self.base_model = getattr(tf_model_zoo, base_model)()
            self.base_model.last_layer_name = 'fc'
            self.input_size = 224
            self.input_mean = [104, 117, 128]
            self.input_std = [1]

            if self.modality == 'Flow':
                self.input_mean = [128]
            elif self.modality == 'RGBDiff':
                self.input_mean = self.input_mean * (1 + self.new_length)

        elif 'inception' in base_model:
            import tf_model_zoo
            self.base_model = getattr(tf_model_zoo, base_model)()
            self.base_model.last_layer_name = 'classif'
            self.input_size = 299
            self.input_mean = [0.5]
            self.input_std = [0.5]
        else:
            raise ValueError('Unknown base model: {}'.format(base_model))

    def train(self, mode=True):
        """
        Override the default train() to freeze the BN parameters
        :return:
        """
        super(SeqVLAD_with_uniform_centers, self).train(mode)
        count = 0
        if self._enable_pbn:
            print("Freezing BatchNorm2D except the first one.")
            for m in self.base_model.modules():
                if isinstance(m, nn.BatchNorm2d):
                    count += 1
                    if count >= (2 if self._enable_pbn else 1):
                        m.eval()

                        # shutdown update in frozen mode
                        m.weight.requires_grad = False
                        m.bias.requires_grad = False

    def partialBN(self, enable):
        self._enable_pbn = enable
    def get_sub_optim_policies(self):
        first_conv_weight = []
        first_conv_weight = []
        first_conv_bias = []
        normal_weight = []
        normal_bias = []
        bn = []

        conv_cnt = 0
        bn_cnt = 0
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Conv1d):
                #ps = list(m.parameters())
                #conv_cnt += 1
                #if conv_cnt == 1:
                #    first_conv_weight.append(ps[0])
                #    if len(ps) == 2:
                #        first_conv_bias.append(ps[1])
                #else:
                #    normal_weight.append(ps[0])
                
                #    if len(ps) == 2:
                #        normal_bias.append(ps[1])
                pass
            elif isinstance(m, torch.nn.Linear):
                ps = list(m.parameters())
                normal_weight.append(ps[0])
                if len(ps) == 2:
                    normal_bias.append(ps[1])
            elif isinstance(m, torch.nn.BatchNorm1d):
                #bn.extend(list(m.parameters()))
                pass
            elif isinstance(m, torch.nn.BatchNorm2d):
                bn_cnt += 1
                # later BN's are frozen
                #if not self._enable_pbn or bn_cnt == 1:
                #    bn.extend(list(m.parameters()))
                pass
            elif isinstance(m, SeqVLADUniformModule):
                print('this is SeqVlad module, and adding the trainable parameters to train')
                assert len(list(m.parameters())) == 8, "the number parameters of seqvlad should be equal to 8"
                ps = list(m.parameters())
                conv_cnt += 5
                normal_weight.extend(ps[0:5])
                normal_bias.extend(ps[5::])
                print('len of weight %d' %(len(ps[0:5])))
                print('len of bias %d' %(len(ps[5::])))

            elif len(m._modules) == 0:
                if len(list(m.parameters())) > 0:
                    raise ValueError("New atomic module type: {}. Need to give it a learning policy".format(type(m)))


        return [
            {'params': first_conv_weight, 'lr_mult': 5 if self.modality == 'Flow' else 1, 'decay_mult': 1,
             'name': "first_conv_weight"},
            {'params': first_conv_bias, 'lr_mult': 10 if self.modality == 'Flow' else 2, 'decay_mult': 0,
             'name': "first_conv_bias"},
            {'params': normal_weight, 'lr_mult': 1, 'decay_mult': 1,
             'name': "normal_weight"},
            {'params': normal_bias, 'lr_mult': 2, 'decay_mult': 0,
             'name': "normal_bias"},
            {'params': bn, 'lr_mult': 1, 'decay_mult': 0,
             'name': "BN scale/shift"},
        ]
    def get_optim_policies(self):
        first_conv_weight = []
        first_conv_bias = []
        normal_weight = []
        normal_bias = []
        bn = []

        conv_cnt = 0
        bn_cnt = 0
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Conv1d):
                ps = list(m.parameters())
                conv_cnt += 1
                if conv_cnt == 1:
                    first_conv_weight.append(ps[0])
                    if len(ps) == 2:
                        first_conv_bias.append(ps[1])
                else:
                    normal_weight.append(ps[0])
                    if len(ps) == 2:
                        normal_bias.append(ps[1])
            elif isinstance(m, torch.nn.Linear):
                ps = list(m.parameters())
                normal_weight.append(ps[0])
                if len(ps) == 2:
                    normal_bias.append(ps[1])
                  
            elif isinstance(m, torch.nn.BatchNorm1d):
                bn.extend(list(m.parameters()))
            elif isinstance(m, torch.nn.BatchNorm2d):
                bn_cnt += 1
                # later BN's are frozen
                if not self._enable_pbn or bn_cnt == 1:
                    bn.extend(list(m.parameters()))
            elif isinstance(m, SeqVLADUniformModule):
                print('this is SeqVlad module, and adding the trainable parameters to train')
                assert len(list(m.parameters())) == 8, "the number parameters of seqvlad should be equal to 8"
                ps = list(m.parameters())
                conv_cnt += 5
                normal_weight.extend(ps[0:5])
                normal_bias.extend(ps[5::])
                print('len of weight %d' %(len(ps[0:5])))
                print('len of bias %d' %(len(ps[5::])))

            elif len(m._modules) == 0:
                if len(list(m.parameters())) > 0:
                    raise ValueError("New atomic module type: {}. Need to give it a learning policy".format(type(m)))
           

        return [
            {'params': first_conv_weight, 'lr_mult': 5 if self.modality == 'Flow' else 1, 'decay_mult': 1,
             'name': "first_conv_weight"},
            {'params': first_conv_bias, 'lr_mult': 10 if self.modality == 'Flow' else 2, 'decay_mult': 0,
             'name': "first_conv_bias"},
            {'params': normal_weight, 'lr_mult': 1, 'decay_mult': 1,
             'name': "normal_weight"},
            {'params': normal_bias, 'lr_mult': 2, 'decay_mult': 0,
             'name': "normal_bias"},
            {'params': bn, 'lr_mult': 1, 'decay_mult': 0,
             'name': "BN scale/shift"},
        ]

    def forward(self, input):
        sample_len = (3 if self.modality == "RGB" else 2) * self.new_length
        if self.modality == 'RGBDiff':
            sample_len = 3 * self.new_length
            input = self._get_diff(input)

        # print('shape', input.view((-1, sample_len) + input.size()[-2:]).size())
        # print('type', type(input.view((-1, sample_len) + input.size()[-2:])))

        base_out = self.base_model(input.view((-1, sample_len) + input.size()[-2:]))

        if self.dropout > 0:
            base_out = self.new_fc(base_out)

        if not self.before_softmax:
            base_out = self.softmax(base_out)
        # if self.reshape:
        #     self.in_shape = input.size()
        #     self.batch_size = self.in_shape[0]//self.timesteps
        #     base_out = base_out.view((self.batch_size) + base_out.size()[1:])

        output = base_out
        # print('seqvlad output:', output.size)
        return output.squeeze(1)

    def _get_diff(self, input, keep_rgb=False):
        input_c = 3 if self.modality in ["RGB", "RGBDiff"] else 2
        input_view = input.view((-1, self.num_segments, self.new_length + 1, input_c,) + input.size()[2:])
        if keep_rgb:
            new_data = input_view.clone()
        else:
            new_data = input_view[:, :, 1:, :, :, :].clone()

        for x in reversed(list(range(1, self.new_length + 1))):
            if keep_rgb:
                new_data[:, :, x, :, :, :] = input_view[:, :, x, :, :, :] - input_view[:, :, x - 1, :, :, :]
            else:
                new_data[:, :, x - 1, :, :, :] = input_view[:, :, x, :, :, :] - input_view[:, :, x - 1, :, :, :]

        return new_data


    def _construct_flow_model(self, base_model):
        # modify the convolution layers
        # Torch models are usually defined in a hierarchical way.
        # nn.modules.children() return all sub modules in a DFS manner
        modules = list(self.base_model.modules())
        first_conv_idx = list(filter(lambda x: isinstance(modules[x], nn.Conv2d), list(range(len(modules)))))[0]
        conv_layer = modules[first_conv_idx]
        container = modules[first_conv_idx - 1]

        # modify parameters, assume the first blob contains the convolution kernels
        params = [x.clone() for x in conv_layer.parameters()]
        kernel_size = params[0].size()
        new_kernel_size = kernel_size[:1] + (2 * self.new_length, ) + kernel_size[2:]
        new_kernels = params[0].data.mean(dim=1, keepdim=True).expand(new_kernel_size).contiguous()

        new_conv = nn.Conv2d(2 * self.new_length, conv_layer.out_channels,
                             conv_layer.kernel_size, conv_layer.stride, conv_layer.padding,
                             bias=True if len(params) == 2 else False)
        new_conv.weight.data = new_kernels
        if len(params) == 2:
            new_conv.bias.data = params[1].data # add bias if neccessary
        layer_name = list(container.state_dict().keys())[0][:-7] # remove .weight suffix to get the layer name

        # replace the first convlution layer
        setattr(container, layer_name, new_conv)
        return base_model

    def _construct_diff_model(self, base_model, keep_rgb=False):
        # modify the convolution layers
        # Torch models are usually defined in a hierarchical way.
        # nn.modules.children() return all sub modules in a DFS manner
        modules = list(self.base_model.modules())
        first_conv_idx = filter(lambda x: isinstance(modules[x], nn.Conv2d), list(range(len(modules))))[0]
        conv_layer = modules[first_conv_idx]
        container = modules[first_conv_idx - 1]

        # modify parameters, assume the first blob contains the convolution kernels
        params = [x.clone() for x in conv_layer.parameters()]
        kernel_size = params[0].size()
        if not keep_rgb:
            new_kernel_size = kernel_size[:1] + (3 * self.new_length,) + kernel_size[2:]
            new_kernels = params[0].data.mean(dim=1, keepdim=True).expand(new_kernel_size).contiguous()
        else:
            new_kernel_size = kernel_size[:1] + (3 * self.new_length,) + kernel_size[2:]
            new_kernels = torch.cat((params[0].data, params[0].data.mean(dim=1, keepdim=True).expand(new_kernel_size).contiguous()),
                                    1)
            new_kernel_size = kernel_size[:1] + (3 + 3 * self.new_length,) + kernel_size[2:]

        new_conv = nn.Conv2d(new_kernel_size[1], conv_layer.out_channels,
                             conv_layer.kernel_size, conv_layer.stride, conv_layer.padding,
                             bias=True if len(params) == 2 else False)
        new_conv.weight.data = new_kernels
        if len(params) == 2:
            new_conv.bias.data = params[1].data  # add bias if neccessary
        layer_name = list(container.state_dict().keys())[0][:-7]  # remove .weight suffix to get the layer name

        # replace the first convolution layer
        setattr(container, layer_name, new_conv)
        return base_model

    @property
    def crop_size(self):
        return self.input_size

    @property
    def scale_size(self):
        return self.input_size * 256 // 224

    def get_augmentation(self):
        if self.modality == 'RGB':
            return torchvision.transforms.Compose([GroupMultiScaleCrop(self.input_size, [1, .875, .75, .66]),
                                                   GroupRandomHorizontalFlip(is_flow=False)])
        elif self.modality == 'Flow':
            return torchvision.transforms.Compose([GroupMultiScaleCrop(self.input_size, [1, .875, .75]),
                                                   GroupRandomHorizontalFlip(is_flow=True)])
        elif self.modality == 'RGBDiff':
            return torchvision.transforms.Compose([GroupMultiScaleCrop(self.input_size, [1, .875, .75]),
                                                   GroupRandomHorizontalFlip(is_flow=False)])




class SeqVLAD_with_centerloss(nn.Module):
    def __init__(self, num_class, num_centers, modality,
                
                 timesteps=1, redu_dim=512,with_relu=False,
                 base_model='resnet101', new_length=None,
                 before_softmax=True,
                 dropout=0.8,
                 activation=None,
                 seqvlad_type='seqvlad',
                 
                 crop_num=1, partial_bn=True):

        super(SeqVLAD_with_centerloss, self).__init__()
        self.modality = modality
        self.num_centers = num_centers
        self.reshape = True
        self.before_softmax = before_softmax
        self.dropout = dropout
        self.crop_num = crop_num
        self.with_relu = with_relu
        self.seqvlad_type = seqvlad_type
        

        if new_length is None:
            self.new_length = 1 if modality == "RGB" else 5
        else:
            self.new_length = new_length


        self.timesteps = timesteps
        self.redu_dim = redu_dim
        self.activation = activation
        print('self.activation,',self.activation)
        print(("""
Initializing SeqVLAD with base model: {}.
SeqVLAD Configurations:
input_modality:     {}
num_centers:       {}
new_length:         {}
dropout_ratio:      {}
        """.format(base_model, self.modality, self.num_centers, self.new_length, self.dropout)))

        self._prepare_base_model(base_model)

        self._add_seqvlad_layer(base_model)

        self._add_classifier_layer(base_model, num_class)
        # print(self.base_model)
        # feature_dim = self._prepare_tsn(num_class)

        if self.modality == 'Flow':
            print("Converting the ImageNet model to a flow init model")
            self.base_model = self._construct_flow_model(self.base_model)
            print("Done. Flow model ready...")
        elif self.modality == 'RGBDiff':
            print("Converting the ImageNet model to RGB+Diff init model")
            self.base_model = self._construct_diff_model(self.base_model)
            print("Done. RGBDiff model ready.")


        if not self.before_softmax:
            self.softmax = nn.Softmax() 

        self._enable_pbn = partial_bn
        if partial_bn:
            self.partialBN(True)

    def _add_seqvlad_layer(self, base_model):
        if base_model == 'BNInception':
            # print(type(list(self.base_model.named_children())))
            # print(list(self.base_model.named_children()))
            # model = nn.Sequential(
            #             collections.OrderedDict(list(self.base_model.named_children())[:-2])
            #        )
            self.base_model._op_list.remove(('global_pool', 'Pooling', 'global_pool', 'inception_5b_output'))
            self.base_model._op_list.remove(('fc', 'InnerProduct', 'fc_action', 'global_pool'))
            delattr(self.base_model,'global_pool')
            delattr(self.base_model, 'fc')
            # self.base_model = model

            self.seqvlad_layer = SeqVLADModule(self.timesteps, self.num_centers, self.redu_dim, self.with_relu, self.activation, with_center_loss=True)
            # if self.seqvlad_type == 'seqvlad':
            #     setattr(self.base_model, 'global_pool', 
            #             SeqVLADModule(self.timesteps, self.num_centers, self.redu_dim, self.with_relu, self.activation))
            # elif self.seqvlad_type == 'bidirect':
            #     setattr(self.base_model, 'global_pool', 
            #             BiSeqVLADModule(self.timesteps, self.num_centers, self.redu_dim, self.with_relu, self.activation))
            # elif self.seqvlad_type == 'unshare_bidirect':
            #      setattr(self.base_model, 'global_pool', 
            #             UnshareBiSeqVLADModule(self.timesteps, self.num_centers, self.redu_dim, self.with_relu, self.activation))
        

    def _add_classifier_layer(self, base_model, num_class):
        if base_model == 'BNInception':
           
            feature_dim = self.num_centers*self.redu_dim


            if self.dropout == 0:
                pass
            else:
                self.dropout_layer = nn.Dropout(p=self.dropout)
            self.new_fc = nn.Linear(feature_dim, num_class)
            std = 0.001
            normal(self.new_fc.weight, 0, std)
            constant(self.new_fc.bias, 0)
            return feature_dim

       
        
 

    def _prepare_base_model(self, base_model):

        if 'resnet' in base_model or 'vgg' in base_model:
            self.base_model = getattr(torchvision.models, base_model)(True)
            self.base_model.last_layer_name = 'fc'
            self.input_size = 224
            self.input_mean = [0.485, 0.456, 0.406]
            self.input_std = [0.229, 0.224, 0.225]

            if self.modality == 'Flow':
                self.input_mean = [0.5]
                self.input_std = [np.mean(self.input_std)]
            elif self.modality == 'RGBDiff':
                self.input_mean = [0.485, 0.456, 0.406] + [0] * 3 * self.new_length
                self.input_std = self.input_std + [np.mean(self.input_std) * 2] * 3 * self.new_length
        elif base_model == 'BNInception':
            import tf_model_zoo
            self.base_model = getattr(tf_model_zoo, base_model)()
            self.base_model.last_layer_name = 'fc'
            self.input_size = 224
            self.input_mean = [104, 117, 128]
            self.input_std = [1]

            if self.modality == 'Flow':
                self.input_mean = [128]
            elif self.modality == 'RGBDiff':
                self.input_mean = self.input_mean * (1 + self.new_length)

        elif 'inception' in base_model:
            import tf_model_zoo
            self.base_model = getattr(tf_model_zoo, base_model)()
            self.base_model.last_layer_name = 'classif'
            self.input_size = 299
            self.input_mean = [0.5]
            self.input_std = [0.5]
        else:
            raise ValueError('Unknown base model: {}'.format(base_model))

    def train(self, mode=True):
        """
        Override the default train() to freeze the BN parameters
        :return:
        """
        super(SeqVLAD_with_centerloss, self).train(mode)
        count = 0
        if self._enable_pbn:
            print("Freezing BatchNorm2D except the first one.")
            for m in self.base_model.modules():
                if isinstance(m, nn.BatchNorm2d):
                    count += 1
                    if count >= (2 if self._enable_pbn else 1):
                        m.eval()

                        # shutdown update in frozen mode
                        m.weight.requires_grad = False
                        m.bias.requires_grad = False

    def partialBN(self, enable):
        self._enable_pbn = enable
    def get_sub_optim_policies(self):
        first_conv_weight = []
        first_conv_weight = []
        first_conv_bias = []
        normal_weight = []
        normal_bias = []
        bn = []

        conv_cnt = 0
        bn_cnt = 0
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Conv1d):
                #ps = list(m.parameters())
                #conv_cnt += 1
                #if conv_cnt == 1:
                #    first_conv_weight.append(ps[0])
                #    if len(ps) == 2:
                #        first_conv_bias.append(ps[1])
                #else:
                #    normal_weight.append(ps[0])
                
                #    if len(ps) == 2:
                #        normal_bias.append(ps[1])
                pass
            elif isinstance(m, torch.nn.Linear):
                ps = list(m.parameters())
                normal_weight.append(ps[0])
                if len(ps) == 2:
                    normal_bias.append(ps[1])
            elif isinstance(m, torch.nn.BatchNorm1d):
                #bn.extend(list(m.parameters()))
                pass
            elif isinstance(m, torch.nn.BatchNorm2d):
                ##bn_cnt += 1
                ## later BN's are frozen
                #if not self._enable_pbn or bn_cnt == 1:
                #    bn.extend(list(m.parameters()))
                pass
            elif isinstance(m, SeqVLADModule) or isinstance(m, BiSeqVLADModule):
                print('this is SeqVlad module, and adding the trainable parameters to train')
                assert len(list(m.parameters())) == 8, "the number parameters of seqvlad should be equal to 8"
                ps = list(m.parameters())
                conv_cnt += 5
                normal_weight.extend(ps[0:5])
                normal_bias.extend(ps[5::])
                print('len of weight %d' %(len(ps[0:5])))
                print('len of bias %d' %(len(ps[5::])))
            elif isinstance(m, UnshareBiSeqVLADModule):
                print('this is unshare SeqVlad module, and adding the trainable parameters to train')
                assert len(list(m.parameters())) == 13, "the number parameters of seqvlad should be equal to 13"
                ps = list(m.parameters())
                conv_cnt += 9
                normal_weight.extend(ps[0:9])
                normal_bias.extend(ps[9::])
                print('len of weight %d' %(len(ps[0:9])))
                print('len of bias %d' %(len(ps[9::])))

            elif len(m._modules) == 0:
                if len(list(m.parameters())) > 0:
                    raise ValueError("New atomic module type: {}. Need to give it a learning policy".format(type(m)))


        return [
            {'params': first_conv_weight, 'lr_mult': 5 if self.modality == 'Flow' else 1, 'decay_mult': 1,
             'name': "first_conv_weight"},
            {'params': first_conv_bias, 'lr_mult': 10 if self.modality == 'Flow' else 2, 'decay_mult': 0,
             'name': "first_conv_bias"},
            {'params': normal_weight, 'lr_mult': 1, 'decay_mult': 1,
             'name': "normal_weight"},
            {'params': normal_bias, 'lr_mult': 2, 'decay_mult': 0,
             'name': "normal_bias"},
            {'params': bn, 'lr_mult': 1, 'decay_mult': 0,
             'name': "BN scale/shift"},
        ]
    def get_optim_policies(self):
        first_conv_weight = []
        first_conv_bias = []
        normal_weight = []
        normal_bias = []
        bn = []

        conv_cnt = 0
        bn_cnt = 0
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Conv1d):
                ps = list(m.parameters())
                conv_cnt += 1
                if conv_cnt == 1:
                    first_conv_weight.append(ps[0])
                    if len(ps) == 2:
                        first_conv_bias.append(ps[1])
                else:
                    normal_weight.append(ps[0])
                    if len(ps) == 2:
                        normal_bias.append(ps[1])
            elif isinstance(m, torch.nn.Linear):
                ps = list(m.parameters())
                normal_weight.append(ps[0])
                if len(ps) == 2:
                    normal_bias.append(ps[1])
                  
            elif isinstance(m, torch.nn.BatchNorm1d):
                bn.extend(list(m.parameters()))
            elif isinstance(m, torch.nn.BatchNorm2d):
                bn_cnt += 1
                # later BN's are frozen
                if not self._enable_pbn or bn_cnt == 1:
                    bn.extend(list(m.parameters()))
            elif isinstance(m, SeqVLADModule) or isinstance(m, BiSeqVLADModule):
                print('this is SeqVlad module, and adding the trainable parameters to train')
                assert len(list(m.parameters())) == 8, "the number parameters of seqvlad should be equal to 8"
                ps = list(m.parameters())
                conv_cnt += 5
                normal_weight.extend(ps[0:5])
                normal_bias.extend(ps[5::])
                print('len of weight %d' %(len(ps[0:5])))
                print('len of bias %d' %(len(ps[5::])))

            elif isinstance(m, UnshareBiSeqVLADModule):
                print('this is unshare SeqVlad module, and adding the trainable parameters to train')
                assert len(list(m.parameters())) == 13, "the number parameters of seqvlad should be equal to 13"
                ps = list(m.parameters())
                conv_cnt += 9
                normal_weight.extend(ps[0:9])
                normal_bias.extend(ps[9::])
                print('len of weight %d' %(len(ps[0:9])))
                print('len of bias %d' %(len(ps[9::])))

            elif len(m._modules) == 0:
                if len(list(m.parameters())) > 0:
                    raise ValueError("New atomic module type: {}. Need to give it a learning policy".format(type(m)))
           

        return [
            {'params': first_conv_weight, 'lr_mult': 5 if self.modality == 'Flow' else 1, 'decay_mult': 1,
             'name': "first_conv_weight"},
            {'params': first_conv_bias, 'lr_mult': 10 if self.modality == 'Flow' else 2, 'decay_mult': 0,
             'name': "first_conv_bias"},
            {'params': normal_weight, 'lr_mult': 1, 'decay_mult': 1,
             'name': "normal_weight"},
            {'params': normal_bias, 'lr_mult': 2, 'decay_mult': 0,
             'name': "normal_bias"},
            {'params': bn, 'lr_mult': 1, 'decay_mult': 0,
             'name': "BN scale/shift"},
        ]

    def forward(self, input):
        sample_len = (3 if self.modality == "RGB" else 2) * self.new_length
        if self.modality == 'RGBDiff':
            sample_len = 3 * self.new_length
            input = self._get_diff(input)

        # print('shape', input.view((-1, sample_len) + input.size()[-2:]).size())
        # print('type', type(input.view((-1, sample_len) + input.size()[-2:])))
        base_out = self.base_model(input.view((-1, sample_len) + input.size()[-2:]))

        assign_predict, vlad_out = self.seqvlad_layer(base_out)

        if self.dropout > 0:
            vlad_out = self.dropout_layer(vlad_out)
        
        
        output = self.new_fc(vlad_out)

        if not self.before_softmax:
            output = self.softmax(output)

        
        return vlad_out, output.squeeze(1)

    def _get_diff(self, input, keep_rgb=False):
        input_c = 3 if self.modality in ["RGB", "RGBDiff"] else 2
        input_view = input.view((-1, self.num_segments, self.new_length + 1, input_c,) + input.size()[2:])
        if keep_rgb:
            new_data = input_view.clone()
        else:
            new_data = input_view[:, :, 1:, :, :, :].clone()

        for x in reversed(list(range(1, self.new_length + 1))):
            if keep_rgb:
                new_data[:, :, x, :, :, :] = input_view[:, :, x, :, :, :] - input_view[:, :, x - 1, :, :, :]
            else:
                new_data[:, :, x - 1, :, :, :] = input_view[:, :, x, :, :, :] - input_view[:, :, x - 1, :, :, :]

        return new_data


    def _construct_flow_model(self, base_model):
        # modify the convolution layers
        # Torch models are usually defined in a hierarchical way.
        # nn.modules.children() return all sub modules in a DFS manner
        modules = list(self.base_model.modules())
        first_conv_idx = list(filter(lambda x: isinstance(modules[x], nn.Conv2d), list(range(len(modules)))))[0]
        conv_layer = modules[first_conv_idx]
        container = modules[first_conv_idx - 1]

        # modify parameters, assume the first blob contains the convolution kernels
        params = [x.clone() for x in conv_layer.parameters()]
        kernel_size = params[0].size()
        new_kernel_size = kernel_size[:1] + (2 * self.new_length, ) + kernel_size[2:]
        new_kernels = params[0].data.mean(dim=1, keepdim=True).expand(new_kernel_size).contiguous()

        new_conv = nn.Conv2d(2 * self.new_length, conv_layer.out_channels,
                             conv_layer.kernel_size, conv_layer.stride, conv_layer.padding,
                             bias=True if len(params) == 2 else False)
        new_conv.weight.data = new_kernels
        if len(params) == 2:
            new_conv.bias.data = params[1].data # add bias if neccessary
        layer_name = list(container.state_dict().keys())[0][:-7] # remove .weight suffix to get the layer name

        # replace the first convlution layer
        setattr(container, layer_name, new_conv)
        return base_model

    def _construct_diff_model(self, base_model, keep_rgb=False):
        # modify the convolution layers
        # Torch models are usually defined in a hierarchical way.
        # nn.modules.children() return all sub modules in a DFS manner
        modules = list(self.base_model.modules())
        first_conv_idx = filter(lambda x: isinstance(modules[x], nn.Conv2d), list(range(len(modules))))[0]
        conv_layer = modules[first_conv_idx]
        container = modules[first_conv_idx - 1]

        # modify parameters, assume the first blob contains the convolution kernels
        params = [x.clone() for x in conv_layer.parameters()]
        kernel_size = params[0].size()
        if not keep_rgb:
            new_kernel_size = kernel_size[:1] + (3 * self.new_length,) + kernel_size[2:]
            new_kernels = params[0].data.mean(dim=1, keepdim=True).expand(new_kernel_size).contiguous()
        else:
            new_kernel_size = kernel_size[:1] + (3 * self.new_length,) + kernel_size[2:]
            new_kernels = torch.cat((params[0].data, params[0].data.mean(dim=1, keepdim=True).expand(new_kernel_size).contiguous()),
                                    1)
            new_kernel_size = kernel_size[:1] + (3 + 3 * self.new_length,) + kernel_size[2:]

        new_conv = nn.Conv2d(new_kernel_size[1], conv_layer.out_channels,
                             conv_layer.kernel_size, conv_layer.stride, conv_layer.padding,
                             bias=True if len(params) == 2 else False)
        new_conv.weight.data = new_kernels
        if len(params) == 2:
            new_conv.bias.data = params[1].data  # add bias if neccessary
        layer_name = list(container.state_dict().keys())[0][:-7]  # remove .weight suffix to get the layer name

        # replace the first convolution layer
        setattr(container, layer_name, new_conv)
        return base_model

    @property
    def crop_size(self):
        return self.input_size

    @property
    def scale_size(self):
        return self.input_size * 256 // 224

    def get_augmentation(self):
        if self.modality == 'RGB':
            return torchvision.transforms.Compose([GroupMultiScaleCrop(self.input_size, [1, .875, .75, .66]),
                                                   GroupRandomHorizontalFlip(is_flow=False)])
        elif self.modality == 'Flow':
            return torchvision.transforms.Compose([GroupMultiScaleCrop(self.input_size, [1, .875, .75]),
                                                   GroupRandomHorizontalFlip(is_flow=True)])
        elif self.modality == 'RGBDiff':
            return torchvision.transforms.Compose([GroupMultiScaleCrop(self.input_size, [1, .875, .75]),
                                                   GroupRandomHorizontalFlip(is_flow=False)])


