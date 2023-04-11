
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
