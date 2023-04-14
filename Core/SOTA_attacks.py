import torch
from torch.distributions.normal import Normal

def label_flipping(user, ratio_flip):

    idx0 = (user.ytr == 0)
    idx1 = (user.ytr == 1)
    x0 = user.xtr[idx0,:]
    y0 = user.ytr[idx0]
    x1 = user.xtr[idx1,:]
    y1 = user.ytr[idx1]
    num1_flip = x1.size(0) * ratio_flip
    num0_flip = x0.size(0) * ratio_flip

    x0_mean = torch.mean(x0, 0).unsqueeze(0)
    x1_mean = torch.mean(x1, 0).unsqueeze(0)

    dis1_list = torch.zeros_like(y1)
    dis0_list = torch.zeros_like(y0)

    for ids, x in enumerate(x1):
        x = x.unsqueeze(0)
        dis = torch.cdist(x, x0_mean)
        dis1_list[ids] = dis

    for ids, x in enumerate(x0):
        x = x.unsqueeze(0)
        dis = torch.cdist(x, x1_mean)
        dis0_list[ids] = dis


    sortlist_1, indices_1 = torch.sort(dis1_list, 0, descending=True)
    sortlist_0, indices_0 = torch.sort(dis0_list, 0, descending=True)

    fliped_y1 = torch.zeros_like(y1)
    fliped_y1.copy_(y1)
    fliped_y0 = torch.zeros_like(y0)
    fliped_y0.copy_(y0)
    it = 0
    for i in indices_1:
        fliped_y1[i] = 0
        it = it + 1
        if it > num1_flip:
            break

    its = 0
    for i in indices_0:
        fliped_y0[i] = 1
        its = its + 1
        if its > num0_flip:
            break

    fliped_ytr = torch.cat((fliped_y0, fliped_y1))

    return fliped_ytr

def flip_users(users, num_ad):
     var = []
     for u in users:
         var.append(u.v.item())
     sort_var = sorted(var, key=float, reverse=True)

     for i in range(num_ad):
         ad_id = var.index(sort_var[i])
         fliped = label_flipping(users[ad_id], 0.95)
         users[ad_id].ytr.copy_(fliped)
         users[ad_id].adv_flag = True

    return users


def adding_noise(user, add_ratio):

    num_add = int(user.xtr.size(0)*add_ratio)
    x_mean = torch.mean(user.xtr, 0).unsqueeze(0)
    x_std = torch.std(user.xtr, 0, unbiased=False).unsqueeze(0)

    sampler = Normal(x_mean, x_std)
    noise = sampler.sample([num_add]).squeeze(1)

    N_y0 = int(num_add/2)
    N_y1 = num_add - N_y0

    noise_y = [0]*N_y0 + [1]*N_y1
    random.shuffle(noise_y)
    noise_y = torch.FloatTensor(noise_y)
    noised_xtr = torch.cat((user.xtr, noise))
    noised_ytr = torch.cat((user.ytr, noise_y))


    return noised_xtr, noised_ytr

def noise_users(users, num_ad):
    var = []
    for u in users:
        var.append(u.v.item())
    sort_var = sorted(var, key=float, reverse=True)

    for i in range(num_ad):
        ad_id = var.index(sort_var[i])
        noised_xtr, noised_ytr = adding_noise(users[ad_id], 2)
        users[ad_id].xtr = noised_xtr
        users[ad_id].ytr = noised_ytr
        users[ad_id].adv_flag = True

    return users

def parameter_poison(user):
    params = user.model.named_parameters()
    dict_params = dict(params)
    for name, param in dict_params.items():
        arb_param = torch.randn(param.size())
        dict_params[name].data.copy_(arb_param)

def perform_byzantine(users, num_ad):
    var = []
    for u in users:
        var.append(u.v.item())
    sort_var = sorted(var, key=float, reverse=True)

    for i in range(num_ad):
        ad_id = var.index(sort_var[i])
        parameter_poison(users[ad_id])
        users[ad_id].adv_flag = True


 def obj_poison(users, num_ad):
     var = []
     for u in users:
         var.append(u.v.item())
     sort_var = sorted(var, key=float, reverse=True)

     for i in range(num_ad):
         ad_id = var.index(sort_var[i])
         users[ad_id].poison_flag = True
         users[ad_id].adv_flag = True

    return users
