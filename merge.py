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
args = parser.parse_args()

def load_score(file_path):
    score_info = np.load(file_path)
    return score_info['scores'], score_info['labels']

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def load_idt(idt_path):
    if idt_path is not None:
      import scipy.io as sio
      boxes = sio.loadmat(idt_path)
      #print(boxes.keys())
      idt =  boxes['scores_idt']
      #print(boxes['vn_idt'][0:10])
      #print(idt.shape)
      #with h5py.File(args['idt_scores'], 'r') as fin:
      #  idt = fin['idt'].value
    else:
      idt = np.zeros(final.shape)
    return idt

def compute_acc(rgb_score, flow_score, labels):

    acc = 0.
    #acc_onlyIDT = np.mean(idt.argmax(axis=1) == labels)

    #idt_wt = args['idt_wt']
    #final = (1-idt_wt) * (
    #  final / np.linalg.norm(final, axis=1, keepdims=True)) + idt_wt * (
    #    idt / np.linalg.norm(idt, axis=1, keepdims=True))
    #acc_withIDT = np.mean(final.argmax(axis=1) == labels)
    #print('Spatial = %0.6f [*%f]\nTemporal = %0.6f [*%f]\nFinal acc = '
    #      '%0.6f.\nonly IDT = %f\nwith IDT = %f' %
    #      (acc_spat, (1.0 - args['temporal_ratio']), acc_temp, args['temporal_ratio'],
    #       acc, acc_onlyIDT, acc_withIDT))

    if args.idt is not None:
        idt_score = load_idt(args.idt)
    for idx, (rgb, flow, label) in enumerate(zip(rgb_score, flow_score, labels)):

        # rgb = softmax(rgb[0][0][0])
        # flow = softmax(flow[0][0][0])
        #print(rgb.shape)
        #print(rgb[0].shape)
        #print(rgb[0][0].shape)
        rgb = rgb[0][0][0]
        flow = flow[0][0][0]
        #print(idt.shape)

        final_pred = np.asarray(rgb)*args.weight[0] + np.asarray(flow)*args.weight[1]
        if args.idt is not None:
            idt = idt_score[idx]
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
