import pickle
import scipy
import numpy as np
from sklearn import preprocessing
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.cluster import MiniBatchKMeans

class LAG_UNIT:
    def __init__(self, opt, class_list=None, train_labels=None):
        self.num_cluster = opt.num_cluster
        self.alpha = opt.alpha
        self.SAVE = {}
        self.class_list = class_list
        self.train_labels = train_labels

    def LAG_Unit(self, feature, Mode=True, Unit_num=1, Channel_num=1):
        if Mode:
            #        print('--------Train LAG Unit--------')
            #        print('feature_train shape:', feature.shape)
            #        print("len:", len(feature))
            use_classes = len(np.unique(self.train_labels))

            # Compute output features
            labels_train, clus_labels, centroid = self.compute_target_(feature)
            #                SAVE['train_dis']=cosine_similarity(feature_train,centroid)
            #                SAVE['test_dis']=cosine_similarity(feature_test,centroid)
            # Solve LSR
            scaler = preprocessing.StandardScaler().fit_transform(feature)
            weight_save, bias_save = self.llsr_train(feature, labels_train.astype(int), encode=True, centroid=centroid)

            self.SAVE[str(Unit_num) + ' clus_labels'] = clus_labels
            self.SAVE[str(Unit_num) + ' LLSR weight'] = weight_save
            self.SAVE[str(Unit_num) + ' LLSR bias'] = bias_save
            self.SAVE[str(Unit_num) + ' scaler'] = scaler

            with open(f'./weight/LAG_{Unit_num}-{Channel_num}.pkl', 'wb') as f:
                pickle.dump(self.SAVE, f, protocol=pickle.HIGHEST_PROTOCOL)  # LAG parameter 저장하기 위하여 피클 파일 생성

            feature = self.llsr_test(feature, weight_save, bias_save)
            pred_labels = np.zeros((feature.shape[0], use_classes))
            for km_i in range(use_classes):
                pred_labels[:, km_i] = feature[:, clus_labels == km_i].sum(1)
            pred_labels = np.argmax(pred_labels, axis=1)
            idx = pred_labels == self.train_labels.reshape(-1)
            print(Unit_num, ' Kmean training acc is: {}'.format(1. * np.count_nonzero(idx) / self.train_labels.shape[0]))

            return feature

        else:
            print('--------Testing LAG--------')
            print('feature_train shape:', feature.shape)
            with open(f'./weight/LAG_{Unit_num}-{Channel_num}.pkl', 'rb') as f:
                SAVE = pickle.load(f)
            scaler = SAVE[str(Unit_num) + ' scaler']
            feature = scaler.transform(feature)
            feature_reduced = self.llsr_test(feature, SAVE[str(Unit_num) + ' LLSR weight'], SAVE[str(Unit_num) + ' LLSR bias'])
            print('feature_train shape:', feature_reduced.shape)
            return feature_reduced

    def compute_target_(self, feature):
        use_classes = len(self.class_list)
        train_labels = self.train_labels.reshape(-1)
        num_clusters_sub = int(self.num_cluster / use_classes)
        # batch_size= 1000
        batch_size = 10000  # 데이터 사이즈에 따라 유동적으로 수정 필요함
        labels = np.zeros((feature.shape[0]))
        clus_labels = np.zeros((self.num_cluster,))
        centroid = np.zeros((self.num_cluster, feature.shape[1]))
        for i in range(use_classes):
            # ID=class_list[i]
            ID = i  # class list 형태에 따라 수정 필요함
            feature_train = feature[train_labels == ID]
            kmeans = MiniBatchKMeans(n_clusters=num_clusters_sub, batch_size=batch_size).fit(feature_train)
            # kmeans = KMeans(n_clusters=num_clusters_sub).fit(feature_train)
            labels[train_labels == ID] = kmeans.labels_ + i * num_clusters_sub
            clus_labels[i * num_clusters_sub:(i + 1) * num_clusters_sub] = ID
            centroid[i * num_clusters_sub:(i + 1) * num_clusters_sub] = kmeans.cluster_centers_

        return labels, clus_labels.astype(int), centroid

    def llsr_train(self, feature_train, labels_train, encode=True, centroid=None, clus_labels=None):
        if encode:
            n_sample = labels_train.shape[0]
            labels_train_onehot = np.zeros((n_sample, clus_labels.shape[0]))
            for i in range(n_sample):
                gt = self.train_labels[i]
                idx = clus_labels == gt
                dis = euclidean_distances(feature_train[i].reshape(1, -1), centroid[idx]).reshape(-1)
                dis = dis / (dis.min() + 1e-5)
                p_dis = np.exp(-dis * self.alpha)
                p_dis = p_dis / p_dis.sum()
                labels_train_onehot[i, idx] = p_dis
        else:
            labels_train_onehot = labels_train
        A = np.ones((feature_train.shape[0], 1))
        feature_train = np.concatenate((A, feature_train), axis=1)
        #    print(np.sort(labels_train_onehot[:10],1)[:,::-1])
        weight = scipy.linalg.lstsq(feature_train, labels_train_onehot)[0]
        weight_save = weight[1:weight.shape[0]]
        bias_save = weight[0].reshape(1, -1)
        return weight_save, bias_save

    def llsr_test(self, feature_test, weight_save, bias_save):
        feature_test = np.matmul(feature_test, weight_save)
        feature_test = feature_test + bias_save
        return feature_test

