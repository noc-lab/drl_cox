import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
from scipy import optimize

class linearCoxPH_Regression(nn.Module):
    def __init__(self, inputSize, outputSize):
        super(linearCoxPH_Regression, self).__init__()
        self.linear = nn.Linear(inputSize, outputSize, bias=False)

    def forward(self, x):
        out = self.linear(x)
        return out

class negLogLikelihood_per_sample_for_splitting(nn.Module):
    def __init__(self):
        super(negLogLikelihood_per_sample_for_splitting, self).__init__()

    def forward(self, prediction, targets, time, refer_prediction, refer_time):
        risk = prediction
        E = targets
        hazard_ratio = torch.exp(risk)
        hazard_ratio_refer = torch.exp(refer_prediction[:, 0])
        ones_1 = torch.ones((1, risk.shape[0]), device=risk.device)
        mat_1 = refer_time.view(refer_time.shape[0], 1).matmul(ones_1)
        ones_2 = torch.ones((refer_prediction.shape[0], 1), device=risk.device)
        mat_2 = ones_2.matmul(time.view(time.shape[0], 1).transpose(0, 1))
        mat_all = ((mat_1 - mat_2) >= 0).type(torch.float)

        hazard_ratio_refer_sum = hazard_ratio_refer.view(hazard_ratio_refer.shape[0], 1).transpose(0, 1).matmul(mat_all)
        partial_sum = hazard_ratio + hazard_ratio_refer_sum.transpose(0, 1)
        uncensored_likelihood = risk - torch.log(partial_sum)
        censored_likelihood = -uncensored_likelihood * E.float()

        return censored_likelihood.sum(axis=1)  # Make sure to sum over the correct axis

def threshplus(x):
    y = x.copy()
    y[y < 0] = 0
    return y

def threshplus_tensor(x):
    y = x.clone()
    y[y < 0] = 0
    return y

def loss_map_chi_factory(loss_values, eps):
    return lambda x: np.sqrt(2 * ((1.0 / eps - 1.0)**2.0) + 1) * np.sqrt(np.mean(threshplus(loss_values - x)**2.0)) + x

def loss_map_chi_factory_tensor(loss_values, eps, opt_eta):
    return np.sqrt(2 * ((1.0 / eps - 1.0)**2.0) + 1) * torch.sqrt(torch.mean(threshplus_tensor(loss_values - opt_eta)**2.0)) + opt_eta

def flatten(nested_list):
    return [item for sublist in nested_list for item in sublist]

def Hu_solve(temp,n_epochs=100,epsilon=0.1):
    # Compute Hu and Chen's DRO model. Return estimator of the covariate.
    for ep in range(n_epochs):
        train1, train2 = temp[:int(len(temp) * 0.5)], temp[int(len(temp) * 0.5):]
        np.random.shuffle(train1)
        np.random.shuffle(train2)
        m=len(train1[0])-2

        input_size = m
        output_size = 1
        split_dro_model = linearCoxPH_Regression(input_size, output_size)
        criterion_per_sample_splitting = negLogLikelihood_per_sample_for_splitting()
        optimizer = optim.Adam(split_dro_model.parameters(), lr=0.01)

        data_X_train_1 = Variable(torch.from_numpy(train1[:, :m]).float())
        data_time_train_1 = Variable(torch.from_numpy(train1[:, m]).float())
        data_event_train_1 = Variable(torch.from_numpy(train1[:, m+1]).float())
        data_X_train_2 = Variable(torch.from_numpy(train2[:, :m]).float())
        data_time_train_2 = Variable(torch.from_numpy(train2[:, m]).float())
        data_event_train_2 = Variable(torch.from_numpy(train2[:, m+1]).float())

        outputs_1 = split_dro_model(data_X_train_1)
        outputs_2 = split_dro_model(data_X_train_2)

        per_sample_losses_1 = criterion_per_sample_splitting(outputs_1, data_event_train_1, data_time_train_1, outputs_2, data_time_train_2)
        chi_loss_np_1 = loss_map_chi_factory(per_sample_losses_1.detach().numpy(), epsilon)
        cutpt_1 = optimize.fminbound(chi_loss_np_1, np.min(per_sample_losses_1.detach().numpy()) - 1000.0, np.max(per_sample_losses_1.detach().numpy()))
        loss_1 = loss_map_chi_factory_tensor(per_sample_losses_1, epsilon, cutpt_1)

        per_sample_losses_2 = criterion_per_sample_splitting(outputs_2, data_event_train_2, data_time_train_2, outputs_1, data_time_train_1)
        chi_loss_np_2 = loss_map_chi_factory(per_sample_losses_2.detach().numpy(), epsilon)
        cutpt_2 = optimize.fminbound(chi_loss_np_2, np.min(per_sample_losses_2.detach().numpy()) - 1000.0, np.max(per_sample_losses_2.detach().numpy()))
        loss_2 = loss_map_chi_factory_tensor(per_sample_losses_2, epsilon, cutpt_2)

        loss = loss_1 + loss_2
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    params_list = [param.tolist() for param in split_dro_model.parameters()]
    flattened_params_list = params_list
    while any(isinstance(i, list) for i in flattened_params_list):
        flattened_params_list = flatten(flattened_params_list)
    split_b = flattened_params_list
    return split_b