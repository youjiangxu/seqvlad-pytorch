import torch
import math
from torch.autograd import Variable


class Identity(torch.nn.Module):
    def forward(self, input):
        return input


class SegmentConsensus(torch.autograd.Function):

    def __init__(self, consensus_type, dim=1):
        self.consensus_type = consensus_type
        self.dim = dim
        self.shape = None

    def forward(self, input_tensor):
        self.shape = input_tensor.size()
        if self.consensus_type == 'avg':
            output = input_tensor.mean(dim=self.dim, keepdim=True)
        elif self.consensus_type == 'identity':
            output = input_tensor
        else:
            output = None

        return output

    def backward(self, grad_output):
        if self.consensus_type == 'avg':
            grad_in = grad_output.expand(self.shape) / float(self.shape[self.dim])
        elif self.consensus_type == 'identity':
            grad_in = grad_output
        else:
            grad_in = None

        return grad_in


class ConsensusModule(torch.nn.Module):

    def __init__(self, consensus_type, dim=1):
        super(ConsensusModule, self).__init__()
        self.consensus_type = consensus_type if consensus_type != 'rnn' else 'identity'
        self.dim = dim

    def forward(self, input):
        return SegmentConsensus(self.consensus_type, self.dim)(input)





class SeqVLADModule(torch.nn.Module):

    def __init__(self, timesteps, num_centers, redu_dim, with_relu=False, activation=None, with_center_loss=False, init_method='xavier_normal'):
        '''
            num_centers: set the number of centers for sevlad
            redu_dim: reduce channels for input tensor
        '''
        super(SeqVLADModule, self).__init__()
        self.num_centers = num_centers
        self.redu_dim = redu_dim
        self.timesteps = timesteps
        self.with_relu = with_relu
        # print('## in SeqVLADModule ##',self.num_centers, self.redu_dim)

        self.in_shape = None
        self.out_shape = self.num_centers*self.redu_dim
        self.batch_size = None
        self.activation = activation

        self.with_center_loss = with_center_loss
        self.init_method = init_method
        # print('## in SeqVLADModule ##',self.num_centers, self.redu_dim)
        


        

        if self.with_relu:
            print('redu with relu ...')
        def init_func(t):
            if self.init_method == 'xavier_normal':
                return torch.nn.init.xavier_normal(t)
            elif self.init_method == 'orthogonal':
                return torch.nn.init.orthogonal(t)
            elif self.init_method == 'uniform':
                return torch.nn.init.uniform(t, a=0, b=0.01)

        self.U_r = torch.Tensor(self.num_centers, self.num_centers, 3, 3) # weight : out, in , h, w
        self.U_r = init_func(self.U_r) 
        self.U_r = torch.nn.Parameter(self.U_r, requires_grad=True)

        self.U_z = torch.Tensor(self.num_centers, self.num_centers, 3, 3) # weight : out, in , h, w
        self.U_z = init_func(self.U_z) 
        self.U_z = torch.nn.Parameter(self.U_z, requires_grad=True)

        self.U_h = torch.Tensor(self.num_centers, self.num_centers, 3, 3) # weight : out, in , h, w
        self.U_h = init_func(self.U_h) 
        self.U_h = torch.nn.Parameter(self.U_h, requires_grad=True)

        self.redu_w = torch.Tensor(self.redu_dim, 1024, 1, 1) # weight : out, in , h, w
        self.redu_w = torch.nn.init.xavier_normal(self.redu_w, gain=1) 
        self.redu_w = torch.nn.Parameter(self.redu_w, requires_grad=True)

        self.share_w = torch.Tensor(self.num_centers, self.redu_dim, 1, 1) # weight : out, in , h, w
        self.share_w = torch.nn.init.xavier_normal(self.share_w, gain=1) 
        self.share_w = torch.nn.Parameter(self.share_w, requires_grad=True)
        
        self.centers = torch.Tensor(self.num_centers, self.redu_dim) # weight : out, in , h, w
        self.centers = init_func(self.centers) 
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
        if not self.with_center_loss:
            return vlad
        else:
            assignments
            assignments = assignments.view(self.batch_size, self.timesteps, self.num_centers, self.in_shape[2]*self.in_shape[3])
            assign_predict = torch.sum(torch.sum(assignments, 3),1)
            return assign_predict, vlad

class BiSeqVLADModule(torch.nn.Module):

    def __init__(self, timesteps, num_centers, redu_dim, with_relu=False, activation=None):
        '''
            num_centers: set the number of centers for sevlad
            redu_dim: reduce channels for input tensor
        '''
        super(BiSeqVLADModule, self).__init__()

        print('add bidirectional seqvlad layer .... ')
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
        self.U_r = torch.nn.init.xavier_normal(self.U_r, gain=1) 
        self.U_r = torch.nn.Parameter(self.U_r, requires_grad=True)

        self.U_z = torch.Tensor(self.num_centers, self.num_centers, 3, 3) # weight : out, in , h, w
        self.U_z = torch.nn.init.xavier_normal(self.U_z, gain=1) 
        self.U_z = torch.nn.Parameter(self.U_z, requires_grad=True)

        self.U_h = torch.Tensor(self.num_centers, self.num_centers, 3, 3) # weight : out, in , h, w
        self.U_h = torch.nn.init.xavier_normal(self.U_h, gain=1) 
        self.U_h = torch.nn.Parameter(self.U_h, requires_grad=True)

        self.redu_w = torch.Tensor(self.redu_dim, 1024, 1, 1) # weight : out, in , h, w
        self.redu_w = torch.nn.init.xavier_normal(self.redu_w, gain=1) 
        self.redu_w = torch.nn.Parameter(self.redu_w, requires_grad=True)

        self.share_w = torch.Tensor(self.num_centers, self.redu_dim, 1, 1) # weight : out, in , h, w
        self.share_w = torch.nn.init.xavier_normal(self.share_w, gain=1) 
        self.share_w = torch.nn.Parameter(self.share_w, requires_grad=True)
        
        self.centers = torch.Tensor(self.num_centers, self.redu_dim) # weight : out, in , h, w
        self.centers = torch.nn.init.xavier_uniform(self.centers, gain=1) 
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

        ## define the func for compute assignments
        def compute_assignments(h_tm1, backward=False):
            assignments_steps = []
            for i in range(self.timesteps):
                if not backward:
                    wx_plus_b_at_t = wx_plus_b[:,i,:,:,:]
                else:
                    wx_plus_b_at_t = wx_plus_b[:,self.timesteps-1-i,:,:,:]

                Uz_h = torch.nn.functional.conv2d(h_tm1, self.U_z, bias=None, stride=1, padding=1) 
                z = torch.nn.functional.sigmoid(wx_plus_b_at_t+Uz_h)

                Ur_h = torch.nn.functional.conv2d(h_tm1, self.U_r, bias=None, stride=1, padding=1) 
                r = torch.nn.functional.sigmoid(wx_plus_b_at_t+Ur_h)

                Uh_h = torch.nn.functional.conv2d(r*h_tm1, self.U_h, bias=None, stride=1, padding=1)
                hh = torch.nn.functional.tanh(wx_plus_b_at_t+Uh_h)

                h = (1 - z) * hh + z*h_tm1
                
                assignments_steps.append(h)


                h_tm1 = h
            return assignments_steps
       

        ## init hidden states
        ## h_tm1 = N, num_centers, H, W
        forward_h_tm1 = torch.autograd.Variable(torch.Tensor(self.batch_size, self.num_centers, self.in_shape[2], self.in_shape[3]), requires_grad=True)
        forward_h_tm1 = torch.nn.init.constant(forward_h_tm1, 0).cuda() 

        backward_h_tm1 = torch.autograd.Variable(torch.Tensor(self.batch_size, self.num_centers, self.in_shape[2], self.in_shape[3]), requires_grad=True)
        backward_h_tm1 = torch.nn.init.constant(backward_h_tm1, 0).cuda() 

        ## prepare the input tensor shape
        ## output
        # assignments = torch.autograd.Variable(torch.Tensor(self.timesteps,).zeros(), requires_grad=True)
        forward_assignments = compute_assignments(forward_h_tm1)
        backward_assignments = compute_assignments(backward_h_tm1, backward=True)
        backward_assignments = backward_assignments[::-1]

        ##add forward and backward assginments
        ## timesteps, batch_size , num_centers, h, w
        assignments = torch.stack(forward_assignments, dim=0) + torch.stack(backward_assignments, dim=0) 

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
class UnshareBiSeqVLADModule(torch.nn.Module):

    def __init__(self, timesteps, num_centers, redu_dim, with_relu=False, activation=None, init_method='xavier_normal'):
        '''
            num_centers: set the number of centers for sevlad
            redu_dim: reduce channels for input tensor
        '''
        super(UnshareBiSeqVLADModule, self).__init__()

        print('add bidirectional seqvlad layer .... ')
        self.num_centers = num_centers
        self.redu_dim = redu_dim
        self.timesteps = timesteps
        self.with_relu = with_relu
        # print('## in SeqVLADModule ##',self.num_centers, self.redu_dim)

        self.in_shape = None
        self.out_shape = self.num_centers*self.redu_dim
        self.batch_size = None
        self.activation = activation
        self.init_method = init_method
        # print('## in SeqVLADModule ##',self.num_centers, self.redu_dim)
        if self.with_relu:
            print('redu with relu ...')
        def init_func(t):
            if self.init_method == 'xavier_normal':
                return torch.nn.init.xavier_normal(t)
            elif self.init_method == 'orthogonal':
                return torch.nn.init.orthogonal(t)
            elif self.init_method == 'uniform':
                return torch.nn.init.uniform(t, a=0, b=0.01)

        self.f_U_r = torch.Tensor(self.num_centers, self.num_centers, 3, 3) # weight : out, in , h, w
        self.f_U_r = init_func(self.f_U_r)
        self.f_U_r = torch.nn.Parameter(self.f_U_r, requires_grad=True)

        self.b_U_r = torch.Tensor(self.num_centers, self.num_centers, 3, 3) # weight : out, in , h, w
        self.b_U_r = init_func(self.b_U_r)
        self.b_U_r = torch.nn.Parameter(self.b_U_r, requires_grad=True)

        self.f_U_z = torch.Tensor(self.num_centers, self.num_centers, 3, 3) # weight : out, in , h, w
        self.f_U_z = init_func(self.f_U_z)
        self.f_U_z = torch.nn.Parameter(self.f_U_z, requires_grad=True)

        self.b_U_z = torch.Tensor(self.num_centers, self.num_centers, 3, 3) # weight : out, in , h, w
        self.b_U_z = init_func(self.b_U_z)
        self.b_U_z = torch.nn.Parameter(self.b_U_z, requires_grad=True)


        self.f_U_h = torch.Tensor(self.num_centers, self.num_centers, 3, 3) # weight : out, in , h, w
        self.f_U_h = init_func(self.f_U_h)
        self.f_U_h = torch.nn.Parameter(self.f_U_h, requires_grad=True)

        self.b_U_h = torch.Tensor(self.num_centers, self.num_centers, 3, 3) # weight : out, in , h, w
        self.b_U_h = init_func(self.b_U_h)
        self.b_U_h = torch.nn.Parameter(self.b_U_h, requires_grad=True)

        self.redu_w = torch.Tensor(self.redu_dim, 1024, 1, 1) # weight : out, in , h, w
        self.redu_w = torch.nn.init.xavier_normal(self.redu_w, gain=1)
        self.redu_w = torch.nn.Parameter(self.redu_w, requires_grad=True)

        self.f_share_w = torch.Tensor(self.num_centers, self.redu_dim, 1, 1) # weight : out, in , h, w
        self.f_share_w = torch.nn.init.xavier_normal(self.f_share_w, gain=1)
        self.f_share_w = torch.nn.Parameter(self.f_share_w, requires_grad=True)


        self.b_share_w = torch.Tensor(self.num_centers, self.redu_dim, 1, 1) # weight : out, in , h, w
        self.b_share_w = torch.nn.init.xavier_normal(self.b_share_w, gain=1)
        self.b_share_w = torch.nn.Parameter(self.b_share_w, requires_grad=True)

        self.centers = torch.Tensor(self.num_centers, self.redu_dim) # weight : out, in , h, w
        self.centers = init_func(self.centers)
        self.centers = torch.nn.Parameter(self.centers, requires_grad=True)

        self.f_share_b = torch.Tensor(self.num_centers,) # weight : out, in , h, w
        self.f_share_b = torch.nn.init.uniform(self.f_share_b)
        self.f_share_b = torch.nn.Parameter(self.f_share_b, requires_grad=True)

        self.b_share_b = torch.Tensor(self.num_centers,) # weight : out, in , h, w
        self.b_share_b = torch.nn.init.uniform(self.b_share_b)
        self.b_share_b = torch.nn.Parameter(self.b_share_b, requires_grad=True)


        self.redu_b = torch.Tensor(self.redu_dim,) # weight : out, in , h, w
        self.redu_b = torch.nn.init.uniform(self.redu_b)
        self.redu_b = torch.nn.Parameter(self.redu_b, requires_grad=True)
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


        ## define the func for compute assignments
        def compute_assignments(forward_h_tm1, backward_h_tm1):


            ## wx_plus_b : N*timesteps, redu_dim, H, W
            forward_wx_plus_b = torch.nn.functional.conv2d(input_tensor, self.f_share_w, bias=self.f_share_b, stride=1, padding=0, dilation=1, groups=1)
            backward_wx_plus_b = torch.nn.functional.conv2d(input_tensor, self.b_share_w, bias=self.b_share_b, stride=1, padding=0, dilation=1, groups=1)

            forward_wx_plus_b = forward_wx_plus_b.view(self.batch_size, self.timesteps, self.num_centers, self.in_shape[2], self.in_shape[3])
            backward_wx_plus_b = backward_wx_plus_b.view(self.batch_size, self.timesteps, self.num_centers, self.in_shape[2], self.in_shape[3])
            ## reshape
            def step_unit(input_t, h_tm1, U_z, U_r, U_h):
                Uz_h = torch.nn.functional.conv2d(h_tm1, U_z, bias=None, stride=1, padding=1)
                z = torch.nn.functional.sigmoid(input_t+Uz_h)

                Ur_h = torch.nn.functional.conv2d(h_tm1, U_r, bias=None, stride=1, padding=1)
                r = torch.nn.functional.sigmoid(input_t+Ur_h)

                Uh_h = torch.nn.functional.conv2d(r*h_tm1, U_h, bias=None, stride=1, padding=1)
                hh = torch.nn.functional.tanh(input_t+Uh_h)

                h = (1 - z) * hh + z*h_tm1
                return h

            forward_assignments_steps = []
            backward_assignments_steps = []
            for i in range(self.timesteps):

                forward_wx_at_t = forward_wx_plus_b[:,i,:,:,:]
                forward_h = step_unit(forward_wx_at_t, forward_h_tm1, self.f_U_z, self.f_U_r, self.f_U_h)
                forward_assignments_steps.append(forward_h)

                backward_wx_at_t = backward_wx_plus_b[:,self.timesteps-1-i,:,:,:]
                backward_h = step_unit(backward_wx_at_t, backward_h_tm1, self.b_U_z, self.b_U_r, self.b_U_h)
                backward_assignments_steps.append(backward_h)
                forward_h_tm1 = forward_h
                backward_h_tm1 = backward_h
            return forward_assignments_steps, backward_assignments_steps


        ## init hidden states
        ## h_tm1 = N, num_centers, H, W
        forward_h_tm1 = torch.autograd.Variable(torch.Tensor(self.batch_size, self.num_centers, self.in_shape[2], self.in_shape[3]), requires_grad=True)
        forward_h_tm1 = torch.nn.init.constant(forward_h_tm1, 0).cuda()

        backward_h_tm1 = torch.autograd.Variable(torch.Tensor(self.batch_size, self.num_centers, self.in_shape[2], self.in_shape[3]), requires_grad=True)
        backward_h_tm1 = torch.nn.init.constant(backward_h_tm1, 0).cuda()

        ## prepare the input tensor shape
        ## output
        # assignments = torch.autograd.Variable(torch.Tensor(self.timesteps,).zeros(), requires_grad=True)
        forward_assignments, backward_assignments = compute_assignments(forward_h_tm1, backward_h_tm1)
        backward_assignments = backward_assignments[::-1]

        ##add forward and backward assginments
        ## timesteps, batch_size , num_centers, h, w
        forward_assignments = torch.transpose(torch.stack(forward_assignments, dim=0), 0, 1).contiguous()
        backward_assignments = torch.transpose(torch.stack(backward_assignments,dim=0), 0, 1).contiguous()
        forward_assignments = forward_assignments.view(self.batch_size*self.timesteps, self.num_centers, self.in_shape[2]*self.in_shape[3])
        backward_assignments = backward_assignments.view(self.batch_size*self.timesteps, self.num_centers, self.in_shape[2]*self.in_shape[3])
        def activate_assignments(temp_assignments):
            if self.activation == 'softmax':
                temp_assignments = torch.transpose(temp_assignments, 1, 2).contiguous()
                temp_assignments = temp_assignments.view(self.batch_size*self.timesteps*self.in_shape[2]*self.in_shape[3], self.num_centers)
                temp_assignments = torch.nn.functional.softmax(temp_assignments) #my_softmax(assignments, dim=1)
                temp_assignments = temp_assignments.view(self.batch_size*self.timesteps, self.in_shape[2]*self.in_shape[3], self.num_centers)
                temp_assignments = torch.transpose(temp_assignments, 1, 2).contiguous()
            else:
                print('TODO implementation ...')
                exit()
            return temp_assignments

        if self.activation is not None:
            forward_assignments = activate_assignments(forward_assignments)
            backward_assignments = activate_assignments(backward_assignments)
        assignments = (forward_assignments + backward_assignments).contiguous()
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
