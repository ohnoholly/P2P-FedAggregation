import torch
import time

def trimmed_mean(users, adj_list, compressor, shared):

    # First transform all parameters to the dict.
    all_weights = {}
    all_masks = {}
    start_trim_time = time.time()
    for u in users:
        start_trim_time = time.time()
        params = u.model.named_parameters()
        all_weights[u.id] = dict(params)
        grads= []
        with torch.no_grad():
            for name, param in all_weights[u.id].items():
                if name != "out.weight":
                    if name !="out.bias":
                        grad = torch.flatten(param.grad)
                        grads.append(grad)

        ids = 0
        for ad in adj_list[u.id-1]:
            if ad == 1:
                if len(users[ids].xsyn)!=0 and users[ids].share < sum(adj_list[ids]) and shared==True:
                    r = torch.randperm(math.floor(u.nos/(sum(adj_list[ids]))))
                    o_xsample = users[ids].xsyn[r]
                    o_ysample = users[ids].ysyn[r]

                    if len(u.o_xsyn) == 0:
                        u.o_xsyn = o_xsample
                        u.o_ysyn = o_ysample
                    else:
                        u.o_xsyn = torch.cat((u.o_xsyn, o_xsample), dim=0)
                        u.o_ysyn = torch.cat((u.o_ysyn, o_ysample), dim=0)

                    users[ids].share = users[ids].share + 1

                ids = ids + 1



        g_o = torch.cat(grads, 0)
        mask = compressor.compress(g_o, u.id, 1)
        all_masks[u.id] = mask


    for u in users:

        if u.adv_flag == False:
            it = 0
            for name, param in all_weights[u.id].items():
                if name != "out.weight":
                    if name !="out.bias":
                        weights = []
                        #Count the number of the parameters
                        num = torch.numel(param)
                        #get the size of the parameters
                        size = param.detach().cpu().numpy().shape

                        ids = 0
                        for ad in adj_list[u.id-1]:
                            if ad == 1:
                                weight_layer = all_weights[ids+1][name]
                                mask_layer = all_masks[u.id][it:it+num]
                                mask_layer = torch.reshape(mask_layer, size)
                                sparse_w = weight_layer * mask_layer
                                weights.append(sparse_w)

                            ids = ids + 1

                        tensor_weights = torch.stack(weights)

                        # If there are more than one neighbors, do...
                        if sum(adj_list[u.id-1]) > 1:
                            maxi, max_id = torch.max(tensor_weights, dim=0)
                            mini, min_id = torch.min(tensor_weights, dim=0)

                            if tensor_weights.dim() > 2:
                                for row, lists in enumerate(max_id):
                                    for col, value in enumerate(lists):
                                        tensor_weights[value][row][col] = 0.


                                for row, lists in enumerate(min_id):
                                    for col, value in enumerate(lists):
                                        tensor_weights[value][row][col] = 0.
                            else:
                                for col, rows in enumerate(max_id):
                                    tensor_weights[rows][col] = 0.


                        #Do not apply sparsification to the current node which is going to be updated
                        tensor_weights = torch.cat((tensor_weights, param.unsqueeze(0)), 0)
                        mask_trimmed = tensor_weights!=0
                        trimmed_weight = (tensor_weights*mask_trimmed).sum(dim=0)/mask_trimmed.sum(dim=0)
                        all_weights[u.id][name].data.copy_(trimmed_weight)
                        it = it + num

    end_trim_time = time.time()
    aggregate_time = end_trim_time - start_trim_time
    print("Trimmed_aggregate_time:",aggregate_time )

def get_median(users, adj_list, compressor, shared):

    # First transform all parameters to the dict.
    all_weights = {}
    all_masks = {}
    start_median_time = time.time()
    for u in users:
        params = u.model.named_parameters()
        all_weights[u.id] = dict(params)
        grads= []
        with torch.no_grad():
            for name, param in all_weights[u.id].items():
                if name != "out.weight":
                    if name !="out.bias":
                        grad = torch.flatten(param.grad)
                        grads.append(grad)


        ids = 0
        for ad in adj_list[u.id-1]:
            if ad == 1:
                if users[ids].adv_flag == False and len(users[ids].xsyn)!=0 and users[ids].share < sum(adj_list[ids]) and shared==True:
                    r = torch.randperm(math.floor(u.nos/(sum(adj_list[ids]))))
                    o_xsample = users[ids].xsyn[r]
                    o_ysample = users[ids].ysyn[r]

                    if len(u.o_xsyn) == 0:
                        u.o_xsyn = o_xsample
                        u.o_ysyn = o_ysample
                    else:
                        u.o_xsyn = torch.cat((u.o_xsyn, o_xsample), dim=0)
                        u.o_ysyn = torch.cat((u.o_ysyn, o_ysample), dim=0)

                    users[ids].share = users[ids].share + 1

                ids = ids + 1



        g_o = torch.cat(grads, 0)
        mask = compressor.compress(g_o, u.id, 1)
        all_masks[u.id] = mask

    for u in users:
        if u.adv_flag == False:
            it = 0
            for name, param in all_weights[u.id].items():
                if name != "out.weight":
                    if name !="out.bias":
                        weights = []
                        #Do not apply sparsification to the current node which is going to be updated
                        weights.append(param)
                        #Count the number of the parameters
                        num = torch.numel(param)
                        #get the size of the parameters
                        size = param.detach().cpu().numpy().shape

                        ids = 0
                        for ad in adj_list[u.id-1]:
                            if ad == 1:
                                weight_layer = all_weights[ids+1][name]
                                mask_layer = all_masks[u.id][it:it+num]
                                mask_layer = torch.reshape(mask_layer, size)
                                sparse_w = weight_layer * mask_layer
                                weights.append(sparse_w)

                            ids = ids + 1

                        tensor_weights = torch.stack(weights)

                        median_weight, index = torch.median(tensor_weights, 0)
                        all_weights[u.id][name].data.copy_(median_weight)
                        it = it + num

        end_median_time = time.time()
        median_time = end_median_time - start_median_time
        print("Median_time:",  median_time)


def krum(users, adj_list, compressor, shared, num_ad):

    # First transform all parameters to the dict.
    all_weights = {}
    all_masks = {}
    update_weight={}
    start_krum_time = time.time()
    for u in users:
        params = u.model.named_parameters()
        all_weights[u.id] = dict(params)
        update_weight[u.id] = dict(params)
        grads= []
        with torch.no_grad():
            for name, param in all_weights[u.id].items():
                if name != "out.weight":
                    if name !="out.bias":
                        grad = torch.flatten(param.grad)
                        grads.append(grad)


        ids = 0
        for ad in adj_list[u.id-1]:
            if ad == 1:
                if users[ids].adv_flag == False and len(users[ids].xsyn)!=0 and users[ids].share < sum(adj_list[ids]) and shared==True:
                    r = torch.randperm(math.floor(u.nos/(sum(adj_list[ids]))))
                    o_xsample = users[ids].xsyn[r]
                    o_ysample = users[ids].ysyn[r]

                    if len(u.o_xsyn) == 0:
                        u.o_xsyn = o_xsample
                        u.o_ysyn = o_ysample
                    else:
                        u.o_xsyn = torch.cat((u.o_xsyn, o_xsample), dim=0)
                        u.o_ysyn = torch.cat((u.o_ysyn, o_ysample), dim=0)

                    users[ids].share = users[ids].share + 1

                ids = ids + 1



        g_o = torch.cat(grads, 0)
        mask = compressor.compress(g_o, u.id, 1)
        all_masks[u.id] = mask


    for u in users:
        if u.adv_flag == False:
            it = 0
            for name, param in all_weights[u.id].items():
                if name != "out.weight":
                    if name !="out.bias":
                        #Count the number of the parameters
                        num = torch.numel(param)
                        #get the size of the parameters
                        size = param.detach().cpu().numpy().shape

                        peers  = [i for i, e in enumerate(adj_list[u.id-1]) if e != 0]
                        # Get the number for sampling
                        #tao = sum(adj_list[u.id-1]) - num_ad - 2
                        # Random sample the peers
                        #sample_peers = random.choices(peers, k=tao)

                        ecu = []
                        for ad in peers:
                            weight_layer = all_weights[ad+1][name]
                            mask_layer = all_masks[u.id][it:it+num]
                            mask_layer = torch.reshape(mask_layer, size)
                            sparse_w = weight_layer * mask_layer
                            distance = ((weight_layer-param)**2).norm()
                            ecu.append(distance.item())

                        min_ids = ecu.index(min(ecu))
                        selected_peer = peers[min_ids]
                        update_weight[u.id][name] = all_weights[selected_peer+1][name]
                        it = it + num

    for u in users:
        if u.adv_flag == False:
            for name, param in all_weights[u.id].items():
                if name != "out.weight":
                    if name !="out.bias":
                        all_weights[u.id][name].data.copy_(update_weight[u.id][name])

    end_krum_time = time.time()
    krum_time = end_krum_time - start_krum_time
    print("Krum_time:",  krum_time)
