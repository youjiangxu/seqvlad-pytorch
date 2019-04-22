import os
import numpy as np 

import argparse

parser = argparse.ArgumentParser('merge rgb, flow, and iDT results')


# parser.add_argument('dataset', type=str, choices=['ucf101', 'hmdb51'])
parser.add_argument('--rgb', type=str, default=None)
parser.add_argument('--flow', type=str, default=None)
parser.add_argument('--idt', type=str, default=None)
parser.add_argument('--weight' ,type=float, default=[1., 1.5], nargs="+")
parser.add_argument('--idt_wt', type=float, default=0.25)
parser.add_argument('--split', type=str, default='1')
args = parser.parse_args()


def load_idt_thing():
    re_names = []
    with open('./data/hmdb51_splits/AllVideos_renamed.txt', 'r') as rf:
        for idx, line in enumerate(rf):
            re_names.append(line.strip())
    ori_names = {}
    with open('./data/hmdb51_splits/AllVideos.txt', 'r') as rf:
        for idx, line in enumerate(rf):

            temp = line.strip().split('/')
            ori_names[idx+1] = temp[1][:-4]
    #print(len(ori_names))
    #print(len(re_names))
    #assert len(ori_names)==len(re_names)
    rename2ori = {}
    for idx, re_name in enumerate(re_names):
        #print(re_name, ori_name)
        line_id = int(re_name.split('/')[1])
        rename2ori[re_name] = ori_names[line_id]
            
    test_names = {}
    with open('./data/hmdb51_splits/test_split'+args.split+'.txt', 'r') as rf:
        for idx, line in enumerate(rf):
            temp = line.strip().split(' ')
            test_names[idx] = temp[0]
    #print(test_names)
    return rename2ori, test_names
    #with open('./data/hmdb51_splits/'
def load_idt(idt_path):
    if idt_path is not None:
      import scipy.io as sio
      boxes = sio.loadmat(idt_path)
      #print(boxes.keys())
      #idt_score =  boxes['scores_idt']
      #print(idt_score)
      
      idt_map = {}
      for idx, (name, idt_score) in enumerate(zip(boxes['vn_idt'], boxes['scores_idt'])):
         #print(idt_score.shape)
         #print(name[0][0])
         #print(type(name[0].astype(str))) 
         idt_map[name[0][0]]=idt_score
          
      #print(type(idt_score))
      #print(boxes['vn_idt'][0:10])
      return idt_map, boxes['scores_idt']
      #print(idt.shape)
      #with h5py.File(args['idt_scores'], 'r') as fin:
      #  idt = fin['idt'].value

def load_score(file_path):
    score_info = np.load(file_path)
    return score_info['scores'], score_info['labels']

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def compute_acc(rgb_score, flow_score, labels):

    acc = 0.
    if args.idt is not None:
        idt_map, idt_score = load_idt(args.idt)
        rename2ori, test_names = load_idt_thing()

    for idx, (rgb, flow, label) in enumerate(zip(rgb_score, flow_score, labels)):
        
        # rgb = softmax(rgb[0][0][0])
        # flow = softmax(flow[0][0][0])

        rgb = rgb[0][0][0]
        flow = flow[0][0][0]
        #print(test_names[idx])
        #print(rename2ori[test_names[idx]])
                #idt = idt_score[real_idt_idx]

        final_pred = np.asarray(rgb)*args.weight[0] + np.asarray(flow)*args.weight[1]
        if args.idt is not None:
          idt = idt_map[rename2ori[test_names[idx]]]
          final_pred = (1-args.idt_wt) * (
                      final_pred / np.linalg.norm(final_pred, axis=0, keepdims=True)) + args.idt_wt * (
                      idt / np.linalg.norm(idt, axis=0, keepdims=True))

        acc += 1 if np.argmax(final_pred) == label else 0

    print('Accuracy {:.02f}%'.format(np.mean(acc*1.0/labels.shape[0] * 100)))

    # np.argmax(np.mean(x[0], axis=0))
    # cf = confusion_matrix(video_labels, video_pred).astype(float)
    # cls_cnt = cf.sum(axis=1)
    # cls_hit = np.diag(cf)
    # cls_acc = cls_hit / cls_cnt
    # print(cls_acc)


if __name__ == '__main__':
    
    rgb_score, labels = load_score(args.rgb)
    flow_score, labels = load_score(args.flow)
    
    compute_acc(rgb_score, flow_score, labels)




