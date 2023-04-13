import numpy as np
import torch


class TopKCompressor():
    """
    Sparse Communication for Distributed Gradient Descent, Alham Fikri Aji et al., 2017
    """
    res = {}


    @staticmethod
    def clear():
        TopKCompressor.res = {}


    @staticmethod
    def layer_wise_compress(tensor, name, id_node, beta, ratio=0.01):
        grad = tensor.grad
        residuals = {}


        with torch.no_grad():
            mask = torch.zeros_like(tensor.data)
            #rest = torch.ones_like(tensor.data)

            if id_node not in TopKCompressor.res:
                TopKCompressor.res[id_node] = residuals
                #TopKCompressor.momen[id_node] = momentum
                #Check whether it is empty
            if name not in TopKCompressor.res[id_node]:
                TopKCompressor.res[id_node][name] = torch.zeros_like(grad.data)
                #TopKCompressor.momen[id_node][name] = torch.zeros_like(grad.data)

            #Normalize grad
            grad = torch.abs(grad.data)
            if grad.dim() == 1:
                grad = nn.functional.normalize(grad, dim=0)
            else:
                grad = nn.functional.normalize(grad)

            #TopKCompressor.momen[id_node][name].data =  beta*TopKCompressor.momen[id_node][name].data + (1-beta)*grad

            TopKCompressor.res[id_node][name].data =  beta*TopKCompressor.res[id_node][name].data + (1-beta)*grad

            # top-k solution
            # torch.numel returns the total number in the input tensor
            numel =  TopKCompressor.res[id_node][name].data.numel()
            k = max(int(numel * ratio), 1)
            #print("k=",k)


            values, indexes = torch.topk(torch.abs(TopKCompressor.res[id_node][name].data).flatten(), k=k)
            indexes = np.array(np.unravel_index(indexes.cpu().numpy(), TopKCompressor.res[id_node][name].shape)).T


            for index in indexes:
                if len(index) < 2:
                    mask[index] = 1

                else:
                    mask[index[0], index[1]] = 1


            TopKCompressor.res[id_node][name].data = TopKCompressor.res[id_node][name].data - mask * grad



            return mask



    @staticmethod
    def compress(tensor, id_node, beta, ratio=0.5):

        with torch.no_grad():
            mask = torch.zeros_like(tensor)


            if id_node not in TopKCompressor.res:
                TopKCompressor.res[id_node] = torch.zeros_like(tensor)


            #Normalize grad
            tensor = torch.abs(tensor)
            #tensor = nn.functional.normalize(tensor, dim=0)

            TopKCompressor.res[id_node] =  beta*TopKCompressor.res[id_node] + (1-beta)*tensor

            # top-k solution
            # torch.numel returns the total number in the input tensor
            numel =  TopKCompressor.res[id_node].numel()
            k = max(int(numel * ratio), 1)
            #print("k=",k)


            values, indexes = torch.topk(torch.abs(TopKCompressor.res[id_node]).flatten(), k=k)
            indexes = np.array(np.unravel_index(indexes.cpu().numpy(), TopKCompressor.res[id_node].shape)).T


            for index in indexes:
                mask[index] = 1


            TopKCompressor.res[id_node] = TopKCompressor.res[id_node] - mask * tensor


            return mask


class model_update:

    similarity={}
    divergence={}
    params = {}

    def __init__(self):
        model_update.similarity = {}
        model_update.divergence = {}
        model_update.params = {}


    @staticmethod
    def sparse_merge_rand(u_or, u_dest,comb, w, masks_list):
        params_dest = u_dest.model.named_parameters()
        dict_params_dest = dict(params_dest)
        params_or = u_or.model.named_parameters()

        for name1, param1 in params_or:
            if name1 in dict_params_dest:
                param2 = dict_params_dest[name1]
                update = torch.zeros_like(param1.data)

                compressed_o = masks_list[name1] * param1
                compressed_d = masks_list[name1] * param2
                update = w*compressed_o.data + comb*compressed_d.data

                with torch.no_grad():
                    #Pick out the indices which are not zero
                    indices = torch.nonzero(update)
                    for index in indices:
                        index = index.cpu().numpy()
                        if len(index) < 2:
                            param2[index] = update[index]
                        else:
                            param2[index[0], index[1]] = update[index[0], index[1]]

                    dict_params_dest[name1].data.copy_(param2)


    @staticmethod
    def sparse_merge_top(u_or, u_dest, comb, w, adj_list, compressor, shared=False):
        grad_o = []
        para_o = []
        para_d = []

        if u_dest.id not in model_update.similarity:
            model_update.similarity[u_dest.id] = {}
        if u_or.id not in model_update.similarity[u_dest.id]:
            model_update.similarity[u_dest.id][u_or.id] = []

        if u_dest.id not in model_update.divergence:
            model_update.divergence[u_dest.id] = {}
        if u_or.id not in model_update.divergence[u_dest.id]:
            model_update.divergence[u_dest.id][u_or.id] = []

        if u_dest.id not in model_update.params:
            model_update.params[u_dest.id] = {}
        if u_or.id not in model_update.params[u_dest.id]:
            model_update.params[u_dest.id][u_or.id] = {}

        params_dest = u_dest.model.named_parameters()
        dict_params_dest = dict(params_dest)
        params_or = u_or.model.named_parameters()
        dict_params_or = dict(params_or)


        with torch.no_grad():
            for name1, param1 in dict_params_or.items():
                if name1 in dict_params_dest:
                    if name1 != "out.weight":
                        if name1 !="out.bias":
                            param2 = dict_params_dest[name1]
                            grad = torch.flatten(param1.grad)
                            grad_o.append(grad)
                            param1 = torch.flatten(param1)
                            param2 = torch.flatten(param2)
                            para_o.append(param1)
                            para_d.append(param2)

        g_o = torch.cat(grad_o, 0)
        p_o = torch.cat(para_o, 0)
        p_d = torch.cat(para_d, 0)
        sim = F.cosine_similarity(p_o, p_d, dim=0)
        div = torch.norm(p_o-p_d)/torch.norm(p_d)
        model_update.similarity[u_dest.id][u_or.id].append(sim.item())
        model_update.divergence[u_dest.id][u_or.id].append(div.item())
        mask = compressor.compress(g_o, u_or.id, 0.95)


        with torch.no_grad():
            start_share_time = time.time()
            it = 0
            for name, param1 in dict_params_or.items():
                if name in dict_params_dest:
                    if name != "out.weight":
                        if name !="out.bias":
                            param2 = dict_params_dest[name]
                            num = torch.numel(param1)
                            size = param1.cpu().numpy().shape
                            mask_layer = mask[it:it+num]
                            mask_layer = torch.reshape(mask_layer, size)
                            it = it + num
                            compressed_o = mask_layer * param1
                            compressed_d = mask_layer * param2

                            model_update.params[u_dest.id][u_or.id][name] = compressed_o
                            update = w*compressed_o.data + comb*compressed_d.data

                            #Pick out the indices which are not zero
                            indices = torch.nonzero(update)
                            for index in indices:
                                index = index.cpu().numpy()
                                if len(index) < 2:
                                    param2[index] = update[index]
                                else:
                                    param2[index[0], index[1]] = update[index[0], index[1]]

                            dict_params_dest[name].data.copy_(param2)

            end_share_time = time.time()
            share_time = end_share_time - start_share_time
            print("share_time:", share_time)

        if u_or.adv_flag==False and len(u_or.xsyn)!=0 and u_or.share < sum(adj_list[u_or.id-1]) and shared==True:
            r = torch.randperm(math.floor(u_dest.nos/(sum(adj_list[u_or.id-1]))))
            o_xsample = u_or.xsyn[r]
            o_ysample = u_or.ysyn[r]

            if len(u_dest.o_xsyn) == 0:
                u_dest.o_xsyn = o_xsample
                u_dest.o_ysyn = o_ysample
            else:
                u_dest.o_xsyn = torch.cat((u_dest.o_xsyn, o_xsample), dim=0)
                u_dest.o_ysyn = torch.cat((u_dest.o_ysyn, o_ysample), dim=0)

            u_or.share = u_or.share + 1


    @staticmethod
    def merge_models_global(m_or,m_dest,alpha_or,alpha_dest):

        params_dest = m_dest.named_parameters()
        dict_params_dest = dict(params_dest)
        params_or = m_or.named_parameters()
        for name1, param1 in params_or:
            if name1 in dict_params_dest:
                dict_params_dest[name1].data.copy_(alpha_or*param1.data +
                            alpha_dest*dict_params_dest[name1].data)
