
import torch
import mindspore
import mindspore.nn
import mindspore.ops as ops
from mindspore.ops import composite as C
import numpy as np
import pandas as pd

def covert_model(torch_file_path):
    torch_param_dict=torch.load(torch_file_path)
    param_dict_list=[]
    param_dict_list_value=[]
    for key in torch_param_dict:
        value_dict=torch_param_dict[key]
        for value_key in value_dict:
            value=value_dict[value_key]
            value=value.cpu()
            value=mindspore.Tensor(value.numpy())
            param_dict_list_value.append({'name': value_key, 'data': value})
        mindspore.save_checkpoint(param_dict_list_value,'/home/luoxuewei/Project/Code3_local/model/umls_complex_ms/model_best.ckpt')
        break
    return

def weight_rename(weight_dict):
    new_dict = {}
    for value_key in weight_dict:
        if value_key == 'kg.entity_embeddings.weight':
            new_dict.update({'fn_kg.entity_embeddings.embedding_table': mindspore.Parameter(weight_dict[value_key].data,
                                                                                        name='fn_kg.entity_embeddings.embedding_table')})

        elif value_key == 'kg.relation_embeddings.weight':
            new_dict.update(
                {'fn_kg.relation_embeddings.embedding_table': mindspore.Parameter(weight_dict[value_key].data,
                                                                                  name='fn_kg.relation_embeddings.embedding_table')})

        elif value_key == 'kg.entity_img_embeddings.weight':
            new_dict.update(
                {'fn_kg.entity_img_embeddings.embedding_table': mindspore.Parameter(weight_dict[value_key].data,
                                                                                    name='fn_kg.entity_img_embeddings.embedding_table')})

        elif value_key == 'kg.relation_img_embeddings.weight':
            new_dict.update(
                {'fn_kg.relation_img_embeddings.embedding_table': mindspore.Parameter(weight_dict[value_key].data,
                                                                                      name='fn_kg.relation_img_embeddings.embedding_table')})

    return new_dict

def parameters_to_vector(parameters):
    r"""Convert parameters to one vector

    Args:
        parameters (Iterable[Tensor]): an iterator of Tensors that are the
            parameters of a model.

    Returns:
        The parameters represented by a single vector
    """
    # Flag for the device where the parameter is located
    #param_device = None

    vec = []
    for param in parameters:
        # Ensure the parameters are located in the same device
        #param_device = _check_param_device(param, param_device)

        vec.append(param.view((-1)))
    #return torch.cat(vec)
    return mindspore.ops.Concat()(vec)

class MyWithLossCell_PG(mindspore.nn.Cell):
    """定义损失网络
        backbone和loss_fn都是pg的成员函数
    """
    def __init__(self, pg):
        super(MyWithLossCell_PG, self).__init__(auto_prefix=False)
        self.pg = pg
        self.loss_t = dict()

    def construct(self, mini_batch):
        #out = self.pg.rollout_hrl(mini_batch)
        loss = self.pg.loss_hrl(mini_batch)
        self.loss_t = loss
        return loss['model_loss_high'] + loss['model_loss_low']

    def consturc_complete(self):
        if self.loss_t is not None:
            return self.loss_t


    # def backbone_network(self):
    #     return self.backbone




class MyTrainStepCell(mindspore.nn.TrainOneStepCell):
    """定义训练流程"""
    def __init__(self, network, optimizer):
        """参数初始化"""
        super(MyTrainStepCell, self).__init__(network, optimizer)
        self.grad = ops.GradOperation(get_by_list=True)

    def construct(self, mini_batch):
        """构建训练过程"""
        weights = self.weights
        #loss = self.network(mini_batch)
        grads = self.grad(self.network, weights)(mini_batch)
        # grads = mindspore.Tensor(grads, mindspore.float32)
        #fill grads
        new_grads=[]
        for i in range(len(grads)):
            if (mindspore.ops.IsNan()(grads[i])).any():
                array = mindspore.Tensor.asnumpy(grads[i]).astype(np.float32)
                array[mindspore.Tensor.asnumpy(mindspore.ops.IsNan()(grads[i]))] = 1.0
                # df.fillna(10.0, inplace=True)
                # array = np.asarray(df).astype(np.float32)
                tensor_grad = mindspore.ops.Cast()(mindspore.Tensor.from_numpy(array), mindspore.float32)
                new_grads.append(mindspore.nn.ClipByNorm()(tensor_grad, ops.cast(ops.tuple_to_array((0.01,)), mindspore.float32)))
            else:
                new_grads.append(grads[i])
        new_grads = tuple(new_grads)
        #clip_grad
        # for i in grads:

            #i = mindspore.nn.DistributedGradReducer(self.get_parameters())(i)
        #grads = C.clip_by_global_norm(grads, clip_norm=5)
        #计算grads[0], grads[1]的众数
        # for i in grads:
        #     if (i.abs()>10).any(i):
        #         new_val = self.most_num(i)
        #         self.instead_abnormal(i, new_val)
        # grads_0 = self.most_num(grads[0])
        # grads_1 = self.most_num(grads[1])
        # self.instead_abnormal(grads[0], grads_0)
        # self.instead_abnormal(grads[1], grads_1)

        self.optimizer(new_grads)
        return new_grads

    def most_num(self, tensor_grads):
        numpy_grads=np.asarray(tensor_grads)
        value, count = np.unique(numpy_grads, return_counts=True)
        res = np.argmax(count)
        return value[res]

    def instead_abnormal(self, grads, new_val):
        grads[grads.abs()>10.0] = new_val
        return grads
