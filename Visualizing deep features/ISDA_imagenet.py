import torch
import torch.nn as nn


class EstimatorCV():
    def __init__(self, feature_num, class_num):
        super(EstimatorCV, self).__init__()
        self.class_num = class_num
        self.CoVariance = torch.zeros(class_num, feature_num).cuda()
        self.Ave = torch.zeros(class_num, feature_num).cuda()
        self.Amount = torch.zeros(class_num).cuda()

    def update_CV(self, features, labels):
        N = features.size(0)
        C = self.class_num
        A = features.size(1)

        NxCxFeatures = features.view(
            N, 1, A
        ).expand(
            N, C, A
        )
        onehot = torch.zeros(N, C).cuda()
        onehot.scatter_(1, labels.view(-1, 1), 1)

        NxCxA_onehot = onehot.view(N, C, 1).expand(N, C, A)

        features_by_sort = NxCxFeatures.mul(NxCxA_onehot)

        Amount_CxA = NxCxA_onehot.sum(0)
        Amount_CxA[Amount_CxA == 0] = 1

        ave_CxA = features_by_sort.sum(0) / Amount_CxA

        var_temp = features_by_sort - \
                   ave_CxA.expand(N, C, A).mul(NxCxA_onehot)

        var_temp = var_temp.pow(2).sum(0).div(Amount_CxA)

        sum_weight_CV = onehot.sum(0).view(C, 1).expand(C, A)

        weight_CV = sum_weight_CV.div(
            sum_weight_CV + self.Amount.view(C, 1).expand(C, A)
        )

        weight_CV[weight_CV != weight_CV] = 0

        additional_CV = weight_CV.mul(1 - weight_CV).mul((self.Ave - ave_CxA).pow(2))

        self.CoVariance = (self.CoVariance.mul(1 - weight_CV) + var_temp
                           .mul(weight_CV)).detach() + additional_CV.detach()

        self.Ave = (self.Ave.mul(1 - weight_CV) + ave_CxA.mul(weight_CV)).detach()

        self.Amount += onehot.sum(0)




class ISDALoss(nn.Module):
    def __init__(self, feature_num, class_num):
        super(ISDALoss, self).__init__()
        self.estimator = EstimatorCV(feature_num, class_num)
        self.class_num = class_num
        self.cross_entropy = nn.CrossEntropyLoss()

    def isda_aug(self, fc, features, y, labels, cv_matrix, ratio):

        N = features.size(0)
        C = self.class_num
        A = features.size(1)

        weight_m = list(fc.parameters())[0]

        NxW_ij = weight_m.expand(N, C, A)

        NxW_kj = torch.gather(NxW_ij,
                              1,
                              labels.view(N, 1, 1)
                              .expand(N, C, A))

        CV_temp = cv_matrix[labels]

        # sigma2 = ratio * \
        #          torch.bmm(torch.bmm(NxW_ij - NxW_kj,
        #                              CV_temp).view(N * C, 1, A),
        #                    (NxW_ij - NxW_kj).view(N * C, A, 1)).view(N, C)

        # sigma2 = ratio * \
        #          torch.bmm(torch.bmm(NxW_ij - NxW_kj,
        #                              CV_temp),
        #                    (NxW_ij - NxW_kj).permute(0, 2, 1))
        # sigma2 = sigma2.mul(torch.eye(C).cuda()
        #                     .expand(N, C, C)).sum(2).view(N, C)

        sigma2 = ratio * (weight_m - NxW_kj).pow(2).mul(
            CV_temp.view(N, 1, A).expand(N, C, A)
        ).sum(2)

        aug_result = y + 0.5 * sigma2

        return aug_result

    def forward(self, model, x, target_x, ratio):

        y, features = model(x, isda=True)

        # y = fc(features)

        self.estimator.update_CV(features.detach(), target_x)

        # isda_aug_y = self.isda_aug(model.module.fc, features, y, target_x, self.estimator.CoVariance.detach(), ratio)
        isda_aug_y = self.isda_aug(model.fc, features, y, target_x, self.estimator.CoVariance.detach(), ratio)
        loss = self.cross_entropy(isda_aug_y, target_x)

        return loss, y
