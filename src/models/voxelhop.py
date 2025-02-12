import pickle
import numpy as np
from sklearn.decomposition import PCA
import time
from numpy import linalg as LA
import msgpack

class VoxelHop:
    def __init__(self, opt, saving_model_path):
        self.mode = opt.Mode
        self.num_unit = opt.NumUnit
        self.weight_name = 'pca_params.pkl'
        self.pca_params = {}
        self.dilate = opt.dilate
        self.pad = opt.pad
        self.Th1 = opt.Th1
        self.Th2 = opt.Th2
        self.model_path = saving_model_path

    def __call__(self, nth_hop, data):
        return self.forward(nth_hop, data)

    def forward(self, nth_hop, data):
        print(f"=========== Start: VoxelHop_{nth_hop}_Unit")
        # Intermediate_node, Leaf_node = self.Transform(feature)
        if self.mode == 'Train':
            Leaf_node = self.fit_transform(nth_hop, data, self.pca_params)
        else:
            Leaf_node = self.Transform(nth_hop, data)

        # print("       <Info>        Output feature shape: %s" % str(Intermediate_node.shape))
        print("       <Info>        Output feature shape: %s" % str(Leaf_node.shape))
        # print("=========== End: PixelHop_Unit -> using %10f seconds" % (time.time() - t0))
        # return Intermediate_node, Leaf_node
        return Leaf_node

    def fit_transform(self, nth_hop, feature, pca_params=None):
        S = list(feature.shape)
        feature = np.moveaxis(feature, -1, 0)
        S[-1] = 1
        Transformed_Leaf = []

        for i in range(feature.shape[0]):
            X = feature[i].reshape(S)
            Neighbor_Constructed = self.Neighbor_construction(X)
            flatten_feature = Neighbor_Constructed.reshape(-1, Neighbor_Constructed.shape[-1])

            mean_removed, DC_mean = self.remove_mean(flatten_feature, axis=1)
            bias = np.max(LA.norm(mean_removed, axis=1))

            training_data, AC_mean = self.remove_mean(mean_removed, axis=0)

            kernels, energy, discard_idx = self.find_kernels_pca(training_data)
            dc_anchor = np.ones((1, training_data.shape[-1])) * 1 / np.sqrt(training_data.shape[-1])
            kernels = np.concatenate((dc_anchor, kernels[:-1]), axis=0)

            flatten_feature += bias
            flatten_feature -= AC_mean
            mul_leaf = np.matmul(flatten_feature, np.transpose(np.array(kernels)))
            Transformed_Leaf.append(mul_leaf)

            print(f"       <Info>        {i + 1}th Kernel shape: {kernels.shape}")

            pca_params[i] = {
                'bias': bias,
                'kernel': kernels,
                'energy': energy,
                'discard_inter': discard_idx,
                'pca_mean': AC_mean
            }

        with open(f'{self.model_path}{nth_hop}hop_params.pkl', 'wb') as f:
            pickle.dump(pca_params, f)

        trns_leaf = np.stack(Transformed_Leaf, axis=-1)
        # leaf_node = trns_leaf.reshape((S[0], S[1], S[2], trns_leaf.shape[1], trns_leaf.shape[2]))
        leaf_node = trns_leaf.reshape((S[0], S[1], S[2], S[3], trns_leaf.shape[1]))

        return leaf_node

    def Transform(self, nth_hop, feature):
        # print("------------------- Start: Pixelhop_fit")
        # print("       <Info>        Using weight: %s" % str(self.weight_name))
        # t0 = time.time()
        with open(f'{self.model_path}{nth_hop}hop_params.pkl', 'rb') as fr:
            pca_params = pickle.load(fr)
        S = list(feature.shape)
        feature = np.moveaxis(feature, -1, 0)
        S[-1] = 1
        Transformed_Leaf = []
        for i in range(feature.shape[0]):
            X = feature[i].reshape(S)
            Neighbor_Constructed = self.Neighbor_construction(X)

            weight = pca_params['kernel'][i]
            bias = pca_params['bias'][i]
            mean = pca_params['pca_mean'][i]
            energy = pca_params['energy']
            discard_idx = pca_params['discard_inter']

            recon = Neighbor_Constructed.reshape(-1, Neighbor_Constructed.shape[-1])
            recon += bias
            recon = recon - mean

            mul_leaf = np.matmul(recon, np.transpose(np.array(weight)))

            # inter_node = np.delete(mul_leaf, discard_idx, axis=-1)

            # Transformed_inter.append(inter_node)
            Transformed_Leaf.append(mul_leaf)

        trns_leaf = np.stack(Transformed_Leaf, axis=-1)
        # trns_inter = np.stack(Transformed_inter, axis=-1)

        Leaf_node = trns_leaf.reshape((S[0], S[1], S[2], trns_leaf.shape[1], trns_leaf.shape[2]))
        # Intermediate_node = trns_i/nter.reshape((S[0], S[1], S[2], trns_inter.shape[1], trns_inter.shape[2]))

        # print("       <Info>        Intermediate feature shape: %s" % str(Intermediate_node.shape))
        # print("       <Info>        Leaf node feature shape: %s" % str(Leaf_node.shape))
        # print("------------------- End: Saab fit -> using %10f seconds" % (time.time() - t0))
        # return Intermediate_node, Leaf_node
        return Leaf_node

    def Neighbor_construction(self, feature):
        S = feature.shape
        Di = self.dilate
        P = self.pad
        if P == 'reflect':
            feature = np.pad(feature, ((0, 0), (Di, Di), (Di, Di), (Di, Di), (0, 0)), 'reflect')
        elif P == 'zeros':
            feature = np.pad(feature, ((0, 0), (Di, Di), (Di, Di), (Di, Di), (0, 0)), 'constant',
                             constant_values=0)
        if P == "none":
            # res = np.zeros((S[1] - 2 * Di, S[2] - 2 * Di, S[0], ((2 * Di + 1) ** 3) * (S[3] - 2 * Di)))
            res = np.zeros((S[1] - 2 * Di, S[2] - 2 * Di, S[0], (S[3] - 2 * Di), (2 * Di + 1) ** 3))
        else:
            # res = np.zeros((S[1], S[2], S[0], ((2 * Di + 1) ** 3) * (S[3] - 2 * Di)))
            res = np.zeros((S[1], S[2], S[3], S[0], (2 * Di + 1) ** 3))


        idx = []
        for N in range(-Di, Di + 1):
            idx.append(int(N / Di))
        feature = np.moveaxis(feature, 0, -2)

        # for i in range(dilate, feature.shape[0] - dilate):
        #     for j in range(dilate, feature.shape[1] - dilate):
        #         tmp = []
        #         for k in range(dilate, feature.shape[2] - dilate):
        #             for ii in idx:
        #                 for jj in idx:
        #                     for kk in idx:
        #                         iii = int(i + ii * dilate)
        #                         jjj = int(j + jj * dilate)
        #                         kkk = int(k + kk * dilate)
        #                         tmp.append(feature[iii, jjj, kkk])
        #         tmp = np.array(tmp)
        #         tmp = np.moveaxis(tmp, 0, 1)
        #         tt= tmp.reshape(S[0],-1)
        #         res[i - dilate, j - dilate] = tt

        offsets = (np.array(np.meshgrid(idx, idx, idx, indexing='ij')) * Di).astype(int)
        offsets = offsets.reshape(3, -1).T  # (27,3)
        for i in range(Di, feature.shape[0] - Di):
            for j in range(Di, feature.shape[1] - Di):
                for k in range(Di, feature.shape[2] - Di):
                    center = np.array([i, j, k])
                    coords = center + offsets
                    tmp_slice = feature[coords[:, 0], coords[:, 1], coords[:, 2]]
                    tmp_slice = np.moveaxis(tmp_slice, 0, 1)
                    tmp_slice = tmp_slice.reshape(S[0], -1)
                    res[i - Di, j - Di, k - Di, :, :] = tmp_slice
                    # res[i - Di, j - Di,:,(k-1)*offsets.shape[0]:k*offsets.shape[0]] = tmp_slice

        res = np.moveaxis(res, 3, 0)

        # print("       <Info>        Output feature shape: %s" % str(res.shape))
        # print("------------------- End: PixelHop_8_Neighbour -> using %10f seconds" % (time.time() - t0))
        return res

    def pca_cal_cov(self, covariance):
        eva, eve = np.linalg.eigh(covariance)
        inds = eva.argsort()[::-1]
        eva = eva[inds]
        kernels = eve.transpose()[inds]
        return kernels, eva

    def pca_cal(self, feature):
        cov = feature.transpose() @ feature
        eva, eve = np.linalg.eigh(cov)
        inds = eva.argsort()[::-1]
        eva = eva[inds]
        kernels = eve.transpose()[inds]
        return kernels, eva / (feature.shape[0] - 1)

    def remove_mean(self, features, axis):
        feature_mean = np.mean(features, axis=axis, keepdims=True)
        feature_remove_mean = features - feature_mean
        return feature_remove_mean, feature_mean

    def find_kernels_pca(self, samples, N=100000):  # , num_kernels
        pca = PCA(n_components=samples.shape[1], svd_solver='full')
        pca.fit(samples)
        energy = pca.explained_variance_ratio_

        energy_sum = np.cumsum(pca.explained_variance_ratio_)
        num_components = np.sum(energy_sum < self.Th1) + 1

        discard_idx = np.argwhere(energy < self.Th2)
        # kernels = pca.components_[:,:]
        kernels = np.delete(pca.components_[:, :], discard_idx, axis=0)
        print("-------Energy sum:", np.sum(energy))

        return kernels, energy, discard_idx
