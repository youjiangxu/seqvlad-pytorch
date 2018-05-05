import torch
import math
from torch.autograd import Variable



class SeqVLADUniformModule(torch.nn.Module):

    def __init__(self, timesteps, num_centers, redu_dim, with_relu=False, activation=None):
        '''
            num_centers: set the number of centers for sevlad
            redu_dim: reduce channels for input tensor
        '''
        super(SeqVLADUniformModule, self).__init__()
        self.num_centers = num_centers
        self.redu_dim = redu_dim
        self.timesteps = timesteps
        self.with_relu = with_relu
        # print('## in SeqVLADModule ##',self.num_centers, self.redu_dim)

        self.in_shape = None
        self.out_shape = self.num_centers*self.redu_dim
        self.batch_size = None
        self.activation = activation
        # print('## in SeqVLADModule ##',self.num_centers, self.redu_dim)
        


        

        if self.with_relu:
            print('redu with relu ...')       

        self.U_r = torch.Tensor(self.num_centers, self.num_centers, 3, 3) # weight : out, in , h, w
        #self.U_r = torch.nn.init.kaiming_normal(self.U_r, gain=1) 
        self.U_r = torch.nn.init.kaiming_normal(self.U_r) 
        self.U_r = torch.nn.Parameter(self.U_r, requires_grad=True)

        self.U_z = torch.Tensor(self.num_centers, self.num_centers, 3, 3) # weight : out, in , h, w
        self.U_z = torch.nn.init.kaiming_normal(self.U_z) 
        self.U_z = torch.nn.Parameter(self.U_z, requires_grad=True)

        self.U_h = torch.Tensor(self.num_centers, self.num_centers, 3, 3) # weight : out, in , h, w
        self.U_h = torch.nn.init.kaiming_normal(self.U_h) 
        self.U_h = torch.nn.Parameter(self.U_h, requires_grad=True)

        self.redu_w = torch.Tensor(self.redu_dim, 1024, 1, 1) # weight : out, in , h, w
        self.redu_w = torch.nn.init.kaiming_normal(self.redu_w) 
        self.redu_w = torch.nn.Parameter(self.redu_w, requires_grad=True)

        self.share_w = torch.Tensor(self.num_centers, self.redu_dim, 1, 1) # weight : out, in , h, w
        self.share_w = torch.nn.init.kaiming_normal(self.share_w) 
        self.share_w = torch.nn.Parameter(self.share_w, requires_grad=True)
        
        self.centers = torch.Tensor(self.num_centers, self.redu_dim) # weight : out, in , h, w
        self.centers = torch.nn.init.uniform(self.centers) 
        self.centers = torch.nn.Parameter(self.centers, requires_grad=True)

        self.share_b = torch.Tensor(self.num_centers,) # weight : out, in , h, w
        self.share_b = torch.nn.init.uniform(self.share_b) 
        self.share_b = torch.nn.Parameter(self.share_b, requires_grad=True)

        
        self.redu_b = torch.Tensor(self.redu_dim,) # weight : out, in , h, w
        self.redu_b = torch.nn.init.uniform(self.redu_b) 
        self.redu_b = torch.nn.Parameter(self.redu_b, requires_grad=True)

        # self.i2h_wx =  torch.nn.Conv2d(self.redu_dim, self.num_centers, 1, stride=1, padding=0, dilation=1, groups=1, bias=True)
        # self.h2h_Ur =  torch.nn.Conv2d(self.num_centers, self.num_centers, 1, stride=1, padding=1, dilation=1, groups=1, bias=False)
        # self.h2h_Uz =  torch.nn.Conv2d(self.num_centers, self.num_centers, 1, stride=1, padding=1, dilation=1, groups=1, bias=False)
        # self.h2h_Uh =  torch.nn.Conv2d(self.num_centers, self.num_centers, 1, stride=1, padding=1, dilation=1, groups=1, bias=False)

        # self.redu_conv = torch.nn.Conv2d(1024, self.redu_dim, 1, stride=1, padding=0, dilation=1, groups=1, bias=True)
        # self.redu_relu = torch.nn.ReLU(inplace=True)

    def forward(self, input):
        # print('input type',type(input))
        # #input = torch.autograd.Variable(input, requires_grad=True)
        # # input = torch.Tensor([input]).cuda() # NEW line

        # return VideoSeqvlad(self.timesteps, self.num_centers, self.redu_dim)(input)
        '''
        input_tensor: N*timesteps, C, H, W
        '''
        self.in_shape = input.size()
        # print('self.in_shape', self.in_shape)
        # print('seqvlad, in type', type(input))
        self.batch_size = self.in_shape[0]//self.timesteps
        if self.batch_size == 0:
            self.batch_size = 1

        
        # input_tensor = torch.autograd.Variable(input, requires_grad=True).cuda()
        input_tensor = input

        if self.redu_dim == None:
            self.redu_dim = self.in_shape[1]
        elif self.redu_dim < self.in_shape[1]:
            #input = input.contiguous()
            # input_tensor = self.redu_relu(self.redu_conv(input_tensor))
            input_tensor = torch.nn.functional.conv2d(input_tensor, self.redu_w, bias=self.redu_b, stride=1, padding=0, dilation=1, groups=1)
            if self.with_relu:
                input_tensor = torch.nn.functional.relu(input_tensor)

        self.out_shape = self.num_centers*self.redu_dim
        ## wx_plus_b : N*timesteps, redu_dim, H, W
        wx_plus_b = torch.nn.functional.conv2d(input_tensor, self.share_w, bias=self.share_b, stride=1, padding=0, dilation=1, groups=1)
        # print('self.batch_size', wx_plus_b.size(), self.batch_size) 
        wx_plus_b = wx_plus_b.view(self.batch_size, self.timesteps, self.num_centers, self.in_shape[2], self.in_shape[3])
        ## reshape 


        ## init hidden states
        ## h_tm1 = N, num_centers, H, W
        h_tm1 = torch.autograd.Variable(torch.Tensor(self.batch_size, self.num_centers, self.in_shape[2], self.in_shape[3]), requires_grad=True)
        h_tm1 = torch.nn.init.constant(h_tm1, 0).cuda() 


        ## prepare the input tensor shape
        ## output
        # assignments = torch.autograd.Variable(torch.Tensor(self.timesteps,).zeros(), requires_grad=True)
        assignments = []

        for i in range(self.timesteps):
            wx_plus_b_at_t = wx_plus_b[:,i,:,:,:]

            Uz_h = torch.nn.functional.conv2d(h_tm1, self.U_z, bias=None, stride=1, padding=1) 
            z = torch.nn.functional.sigmoid(wx_plus_b_at_t+Uz_h)

            Ur_h = torch.nn.functional.conv2d(h_tm1, self.U_r, bias=None, stride=1, padding=1) 
            r = torch.nn.functional.sigmoid(wx_plus_b_at_t+Ur_h)

            Uh_h = torch.nn.functional.conv2d(r*h_tm1, self.U_h, bias=None, stride=1, padding=1)
            hh = torch.nn.functional.tanh(wx_plus_b_at_t+Uh_h)

            h = (1 - z) * hh + z*h_tm1
            assignments.append(h)
            h_tm1 = h

        ## timesteps, batch_size , num_centers, h, w

        assignments = torch.stack(assignments, dim=0)
        # print('assignments shape', assignments.size())

        ## timesteps, batch_size, num_centers, h, w ==> batch_size, timesteps, num_centers, h, w
        assignments = torch.transpose(assignments, 0, 1).contiguous()
        # print('transposed assignments shape', assignments.size())

        ## assignments: batch_size, timesteps, num_centers, h*w
        assignments = assignments.view(self.batch_size*self.timesteps, self.num_centers, self.in_shape[2]*self.in_shape[3])
        if self.activation is not None:
            if self.activation == 'softmax':
                assignments = torch.transpose(assignments, 1, 2).contiguous()
                assignments = assignments.view(self.batch_size*self.timesteps*self.in_shape[2]*self.in_shape[3], self.num_centers)
                assignments = torch.nn.functional.softmax(assignments) #my_softmax(assignments, dim=1)
                assignments = assignments.view(self.batch_size*self.timesteps, self.in_shape[2]*self.in_shape[3], self.num_centers)
                assignments = torch.transpose(assignments, 1, 2).contiguous()
            else:
                print('TODO implementation ...')
                exit()

        ## alpha *c 
        ## a_sum: batch_size, timesteps, num_centers, 1
        a_sum = torch.sum(assignments, -1, keepdim=True)

        ## a: batch_size*timesteps, num_centers, redu_dim
        a = a_sum * self.centers.view(1, self.num_centers, self.redu_dim)

        ## alpha* input_tensor
        ## fea_assign: batch_size, timesteps, num_centers, h, w ==> batch_size*timesteps, num_centers, h*w 
        # fea_assign = assignments.view(self.batch_size*self.timesteps, self.num_centers, self.in_shape[2]*self.in_shape[3])

        ## input_tensor: batch_size, timesteps, redu_dim, h, w  ==> batch_size*timesteps, redu_dim, h*w  ==>  batch_size*timesteps, h*w, redu_dim 
        input_tensor = input_tensor.view(self.batch_size*self.timesteps, self.redu_dim, self.in_shape[2]*self.in_shape[3])
        input_tensor = torch.transpose(input_tensor, 1, 2)

        ## x: batch_size*timesteps, num_centers, redu_dim
        x  = torch.matmul(assignments, input_tensor)


        ## batch_size*timesteps, num_centers, redu_dim
        vlad = x - a 

        ## batch_size*timesteps, num_centers, redu_dim ==> batch_size, timesteps, num_centers, redu_dim
        vlad = vlad.view(self.batch_size, self.timesteps, self.num_centers, self.redu_dim)

        ## batch_size, num_centers, redu_dim 
        vlad = torch.sum(vlad, 1, keepdim=False)

        ## intor normalize
        vlad = torch.nn.functional.normalize(vlad, p=2, dim=2)

        ## l2-normalize
        vlad = vlad.view(self.batch_size, self.num_centers*self.redu_dim)
        vlad = torch.nn.functional.normalize(vlad, p=2, dim=1)
        # print('vlad type', type(vlad))
        # print(vlad.size())
        # vlad = torch.Tensor([vlad]).cuda() # NEW line
        return vlad


