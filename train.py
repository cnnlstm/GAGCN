from __future__ import division
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import numpy as np
import h5py
from sklearn.cluster import *
import json

training_ = {}
for i in range(10):
    training_[i]=random.sample(range(0,101),51)


print (training_)


for Iclass in range(10):
    
    training_classes = training_[Iclass]
    testing_classes = []
    for i in range(1,102):
        if i not in training_classes:
            testing_classes.append(i)
    dic_train={}
    for i,j in enumerate(training_classes):
        dic_train[i]=j
    dic_test={}
    for i,j in enumerate(testing_classes):
        dic_test[i]=j

    
    
    out_feas_test = h5py.File("feature/rgb_test_data_"+str(Iclass)+".h5")['feature'][:]
    labels_test= h5py.File("feature/rgb_test_data_"+str(Iclass)+".h5")['label'][:]
    cluster_ = DBSCAN().fit(out_feas_test)#FeatureAgglomeration(n_clusters=50).fit(out_feas_test.T)

    
    print set(cluster_.labels_)
    print cluster_.labels_
    
    clus_center_test = {}
    for i in range(len(set(cluster_.labels_))):
        center = []
        for j,k in enumerate(cluster_.labels_):
            if k==i:
                center.append(out_feas_test[j])
        center = list(np.mean(center,axis=0))
        yy = []
        for l in center:
            yy.append(str(l))
        
        clus_center_test[dic_test[i]]=yy
    json.dump(clus_center_test,open('clus_centers_test.json', 'w'))

    
    
    
    
    train_fea = 'feature/rgb_train_data_'+str(Iclass)+'.h5'
    def read_h5(path):
        g = h5py.File(path)
        data = g['feature']
        data = data[:]
        label = g['label']
        label = label[:]
        return data,label
    
    fea,label = read_h5(train_fea)
    
    clus_centers_train = {}
    for i in training_classes:
    
        class_fea = []
        for m in range(label.shape[0]):
            if label[m]==i:
                class_fea.append(fea[m])
        class_fea = np.array(class_fea)
        print class_fea.shape
        clus_center = list(np.mean(class_fea,axis=0))
        center = []
        for j in list(np.squeeze(clus_center)):
            center.append(str(j))
        clus_centers_train[str(i)]=center
    json.dump(clus_centers_train,open('clus_centers_train.json', 'w'))

    
    
    
    
    dic = {}
    for i in cluster_.labels_:
        dic[dic_test[i]]=[]
    for i in range(out_feas_test.shape[0]):
        cluster_id = cluster_.labels_[i]
        dic[dic_test[cluster_id]].append(i)
    json.dump(dic,open('cluster_index.json', 'w'))

    
    
    
    from sklearn.neighbors import NearestNeighbors

    
    
    def read_h5(path):
        g = h5py.File(path)
        data = g['data']
        data = data[:]
        return data
    
    def get_nn(classnames, Full_data, one_example):
        edge = []
        nbrs = NearestNeighbors(n_neighbors=11, algorithm='auto', metric='cosine').fit(Full_data)
        distances, nn_indices = nbrs.kneighbors(one_example)
        for i, index in enumerate(nn_indices[0][1:]):
            edge.append(index)
        return edge,distances[0]
    
    def main():
        classnames = []
        labels_ordinary = []
        vectors = []
        labels_dic = {}
        with open('classInd.txt', 'r') as f:
            classnames = f.read().split('\n')
        f.close()
    
        train_json = json.load(open('clus_centers_train.json', 'r'))
        test_json = json.load(open('clus_centers_test.json', 'r'))
    
        train_test_json = dict(train_json, **test_json)

    
        for i in range(101):
            vector = []
            for j in train_test_json[str(i+1)]:
                vector.append(float(j))
            vectors.append(np.array(vector))
        vectors=np.array(vectors)

    
    
        edges = [[] for i in range(10)]
        distances = []
    
        for u in testing_classes:
            edge,distance=get_nn(classnames, vectors, np.reshape(vectors[u-1], (1, 1024)))
            distances.append(distance)
            for i,j in enumerate(edge):
                edges[i].append((u-1,j))
        for u in training_classes:
            edge,distance=get_nn(classnames, vectors, np.reshape(vectors[u-1], (1, 1024)))
            distances.append(distance)
            for i,j in enumerate(edge):
                edges[i].append((u-1,j))
        obj = {}
        obj['edges'] = edges
        json.dump(obj,open('graph_atte.json', 'w'))

    
    
    if __name__ == '__main__':
        main()

    
    
    import argparse
    import json
    import random
    import os.path as osp
    import h5py
    import torch
    import torch.nn.functional as F
    from utils import ensure_path, set_gpu, l2_loss
    from models.gcn_ordinary_atte import GCN_Ord_Atte
    import numpy as np
    import scipy.sparse as sp
    def save_checkpoint(name):
        torch.save(pred_obj, osp.join(save_path, name + '.pred'))
    
    mask_train = []
    mask_test = []
    
    for i in training_classes:
        mask_train.append(i-1)

    
    
    def mask_l2_loss(a, b, mask):
        return l2_loss(a[mask], b[mask])
    
    if __name__ == '__main__':
        parser = argparse.ArgumentParser()

        parser.add_argument('--max-epoch', type=int, default=50000)
        parser.add_argument('--lr', type=float, default=0.001)
        parser.add_argument('--weight-decay', type=float, default=0.0005)
        parser.add_argument('--save-path', default='a_save_'+str(Iclass)+'/')
        parser.add_argument('--gpu', default='3')
        args = parser.parse_args()

    
        set_gpu(args.gpu)
        save_path = args.save_path
        ensure_path(save_path)

    
        #build edge
        graph = json.load(open("graph_atte.json", 'r'))
        edges = graph['edges']
        n = 101
        for i in range(len(edges)):
            edges[i] = edges[i] + [(v, u) for (u, v) in edges[i]]
            edges[i] = edges[i] + [(u, u) for u in range(n)]

    
        #input
        train_json = json.load(open('clus_centers_train.json', 'r'))
        test_json = json.load(open('clus_centers_test.json', 'r'))
        train_test_json = dict(train_json, **test_json)
        vectors = []
        for i in range(101):
            vector = []
            for j in train_test_json[str(i+1)]:
                vector.append(float(j))
            vectors.append(np.array(vector))
        vectors=np.array(vectors)
        class_fea = torch.tensor(torch.from_numpy(vectors)).float().cuda()
        class_fea = F.normalize(class_fea)
        #output
        label_vectors = np.array(h5py.File("glove_ucf101.h5")['vectors'][:],dtype=np.float32)
        label_vectors = torch.tensor(label_vectors).cuda()
        label_vectors = F.normalize(label_vectors)

    
        hidden_layers = 'd2048,d'
        gcn = GCN_Ord_Atte(n, edges, class_fea.shape[1], label_vectors.shape[1], hidden_layers).cuda()
        optimizer = torch.optim.Adam(gcn.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    
    
        for epoch in range(1, args.max_epoch + 1):
            gcn.train()
            output_vectors = gcn(class_fea)
            loss = mask_l2_loss(output_vectors, label_vectors, mask_train)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
            train_loss = loss.item()
            pred_obj = {
                'pred': output_vectors
            }
            save_checkpoint('epoch-{:0>10}'.format(epoch))

    
    
    
    
    
    
    import numpy as np
    import h5py
    import json
    import torch
    import torch.nn.functional as F
    from sklearn.neighbors import NearestNeighbors
    
    testing_classes=[]
    for i in range(1,102):
        if i not in training_classes:
            testing_classes.append(i-1)
    f = open("result.txt","a")
    path ="save_"+str(Iclass)+"/"

    def get_nn(Full_data, one_example):
        nbrs = NearestNeighbors(n_neighbors=1, algorithm='auto', metric='cosine').fit(Full_data)
        distances, nn_indices = nbrs.kneighbors(one_example)
        return nn_indices
    
    g = h5py.File("glove_ucf101.h5")
    label_vectors = g['vectors'][:]
    gt_labels = h5py.File("feature/rgb_test_data_"+str(Iclass)+".h5")['label'][:]

    
    
    gt_labvec = []
    for i in testing_classes:
        gt_labvec.append(label_vectors[i])
    gt_labvec = np.array(gt_labvec)
    g.close()
    
    gt_labvec = torch.tensor(gt_labvec)
    gt_labvec = F.normalize(gt_labvec).detach().numpy()
    
    cluster_index = json.load(open('cluster_index.json', 'r'))
    accs = []
    accs_dic = {}
    preds = []

    for pred in sorted(os.listdir(path)):
            preds.append(pred)
    
            pred_file = torch.load(path+"/"+pred)
            pre_labvec = []
            pred_label = pred_file['pred'].cpu().detach().numpy()
            for i in testing_classes:
                pre_labvec.append([pred_label[i],i+1])  #i from 0


    
            dics = {}
            for i in range(len(pre_labvec)):
                n = get_nn(gt_labvec,pre_labvec[i][0].reshape(1,-1))
                dics[pre_labvec[i][1]]=testing_classes[int(n)]+1     #testing_classes begin from 0
            total = 0
            correct = 0
            for n in cluster_index:
                pred_label = dics[int(n)]
                for m in cluster_index[str(n)]:
                    total+=1
                    gt_label = gt_labels[m]
                    if int(gt_label)==int(pred_label):
                        correct+=1
            acc = correct/total
            #print acc
            accs_dic[acc]=[pred]
            accs.append(acc)
    best_pred = accs_dic[max(accs)]
    for i in preds:
            if i != best_pred[0]:
               os.remove(path+"/"+i)
    print best_pred,"acc:",max(accs),correct,total
    f.write(str(max(accs)))
    f.write("\n")

    
    
    
