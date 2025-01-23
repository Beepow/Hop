import pickle
import numpy as np
from sklearn.decomposition import PCA
import time
from numpy import linalg as LA

class VoxelHop:
    def __init__(self, opt):
        self.mode = opt.Mode
        self.num_unit = opt.NumUnit
        self.weight_name = 'pca_params.pkl'
        self.pca_params = []
        self.dilate = opt.dilate
        self.pad = opt.pad
        self.Th1 = opt.Th1
        self.Th2 = opt.Th2

    def __call__(self, data):
        return self.forward(data)

    def forward(self, data):
        print("=========== Start: PixelHop_Unit")
        if self.mode == 'Train':
            self.Saab_Fit(data)
        # Intermediate_node, Leaf_node = self.Transform(feature)
        Leaf_node = self.Saab_Transform(data, self.pca_params)

        # print("       <Info>        Output feature shape: %s" % str(Intermediate_node.shape))
        print("       <Info>        Output feature shape: %s" % str(Leaf_node.shape))
        # print("=========== End: PixelHop_Unit -> using %10f seconds" % (time.time() - t0))
        # return Intermediate_node, Leaf_node
        return Leaf_node

    def Saab_Transform(self, feature, pca_params=None):
        # print("------------------- Start: Pixelhop_fit")
        # print("       <Info>        Using weight: %s" % str(self.weight_name))
        # t0 = time.time()
        fr = open('./weight/' + self.weight_name, 'rb')
        pca_params = pickle.load(fr)
        fr.close()
        S = list(feature.shape)
        feature = np.moveaxis(feature, -1, 0)
        S[-1] = 1
        Transformed_inter = []
        Transformed_Leaf = []
        for i in range(feature.shape[0]):
            X = feature[i].reshape(S)
            Neighbor_Constructed = self.Neighbor_construction(X, dilate=self.dilate, pad=self.pad)

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

    def remove_mean(self, features, axis):
        feature_mean = np.mean(features, axis=axis, keepdims=True)
        feature_remove_mean = features - feature_mean
        return feature_remove_mean, feature_mean

    def find_kernels_pca(self, samples, N=100000): #, num_kernels
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

    def Saab_Fit(self, feature):
        print("------------------- Start: Saab transformation")
        t0 = time.time()
        S = list(feature.shape)
        kernel_cur = []
        bias_cur = []
        AC_Mean = []
        DC_Mean = []
        Covar = []
        discard_idx = []
        pca_params = {}
        print("       <Info>        pixelhop_feature.shape: %s" % str(feature.shape))
        feature = np.moveaxis(feature, -1, 0)

        S[-1] = 1
        for i in range(feature.shape[0]):
            X = feature[i].reshape(S)

            Neighbor_Constructed = self.Neighbor_construction(X, dilate=self.dilate, pad=self.pad)

            flatten_feature = Neighbor_Constructed.reshape(-1, Neighbor_Constructed.shape[-1])
            mean_removed, DC_mean = self.remove_mean(flatten_feature, axis=1)
            pp = np.mean(mean_removed)
            bias = LA.norm(mean_removed, axis=1)
            bias = np.max(bias)
            bias_cur.append(bias)

            training_data, AC_mean = self.remove_mean(mean_removed, axis=0)
            # print(np.mean(training_data, axis=0))
            AC_Mean.append(AC_mean)

            kernels, energy, discard_idx = self.find_kernels_pca(training_data)#, self.num_kernel
            dc_anchor = np.ones((1, training_data.shape[-1])) * 1 / np.sqrt(training_data.shape[-1])
            kernels = np.concatenate((dc_anchor, kernels[:-1]), axis=0)
            kernel_cur.append(kernels)

            pca_params['bias'] = bias_cur
            pca_params['kernel'] = kernel_cur
            pca_params['energy'] = energy
            pca_params['discard_inter'] = discard_idx
            pca_params['pca_mean'] = AC_Mean

            print(f"       <Info>        {i+1}th Kernel shape: %s" % str(kernels.shape))

        fw = open('./weight/' + self.weight_name, 'wb')
        pickle.dump(pca_params, fw)
        fw.close()
        print("       <Info>        Save pca params as name: %s" % str(self.weight_name))
        print("------------------- End: Saab transformation -> using %10f seconds" % (time.time() - t0))


    def Neighbor_construction(self, feature, dilate, pad):

        S = feature.shape
        if pad == 'reflect':
            feature = np.pad(feature, ((0, 0), (dilate, dilate), (dilate, dilate), (0, 0), (0, 0)), 'reflect')
        elif pad == 'zeros':
            feature = np.pad(feature, ((0, 0), (dilate, dilate), (dilate, dilate), (0, 0), (0, 0)), 'constant',
                             constant_values=0)
        if pad == "none":
            res = np.zeros((S[1] - 2 * dilate, S[2] - 2 * dilate, S[0], ((2 * dilate + 1) ** 3)* (S[3] - 2*dilate)))
        else:
            res = np.zeros((S[1], S[2], S[0], ((2 * dilate + 1) ** 3)* (S[3] - 2*dilate)))

        idx = []
        for N in range(-dilate, dilate + 1):
            idx.append(int(N / dilate))
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

        offsets = (np.array(np.meshgrid(idx, idx, idx, indexing='ij')) * dilate).astype(int)
        offsets = offsets.reshape(3, -1).T  # (27,3)
        for i in range(dilate, feature.shape[0] - dilate):
            for j in range(dilate, feature.shape[1] - dilate):
                for k in range(dilate, feature.shape[2] - dilate):
                    center = np.array([i, j, k])
                    coords = center + offsets
                    tmp_slice = feature[coords[:, 0], coords[:, 1], coords[:, 2]]
                    tmp_slice = np.moveaxis(tmp_slice, 0, 1)
                    res[i - dilate, j - dilate,:,(k-1)*offsets.shape[0]:k*offsets.shape[0]] = tmp_slice.reshape(S[0], -1)

        res = np.moveaxis(res, 2, 0)

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