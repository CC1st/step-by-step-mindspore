"""
 Copyright (c) 2018, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
 
 Base learning framework.
"""

import os
import random
import shutil
from tqdm import tqdm

import logging
import numpy as np
import gc

import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_

import mindspore
import mindspore.nn
from mindspore import Model
from src.utils.ms_load_checkpoint import covert_model, MyWithLossCell_PG, MyTrainStepCell

import src.eval
from src.utils.ops import var_cuda, zeros_var_cuda
import src.utils.ops as ops

import warnings
from typing import Union, Iterable
import math

_tensor_or_tensors = Union[mindspore.Tensor, Iterable[mindspore.Tensor]]
inf = math.inf

#overwrite clip_grap_norm_
def clip_grad_norm_(
        parameters: _tensor_or_tensors, max_norm: float, norm_type: float = 2.0,
        error_if_nonfinite: bool = False):
    r"""Clips gradient norm of an iterable of parameters.

    The norm is computed over all gradients together, as if they were
    concatenated into a single vector. Gradients are modified in-place.

    Args:
        parameters (Iterable[Tensor] or Tensor): an iterable of Tensors or a
            single Tensor that will have gradients normalized
        max_norm (float or int): max norm of the gradients
        norm_type (float or int): type of the used p-norm. Can be ``'inf'`` for
            infinity norm.
        error_if_nonfinite (bool): if True, an error is thrown if the total
            norm of the gradients from :attr:``parameters`` is ``nan``,
            ``inf``, or ``-inf``. Default: False (will switch to True in the future)

    Returns:
        Total norm of the parameters (viewed as a single vector).
    """
    ms_max = mindspore.ops.ArgMaxWithValue()
    ms_stack = mindspore.ops.Stack()
    ms_isnan=mindspore.ops.IsNan()
    ms_isinf=mindspore.ops.IsInf()
    if isinstance(parameters, mindspore.Tensor):
        parameters = [parameters]
    parameters = [p for p in parameters if p.grad is not None]
    max_norm = float(max_norm)
    norm_type = float(norm_type)
    if len(parameters) == 0:
        return mindspore.Tensor(0.)
    device = parameters[0].grad.device
    if norm_type == inf:
        norms = [p.grad.detach().abs().max().to(device) for p in parameters]
        total_norm = norms[0] if len(norms) == 1 else ms_max(ms_stack(norms)) #[1]?
    else:
        ms_norm=mindspore.nn.Norm()
        #norm_type
        total_norm = ms_norm(ms_stack([torch.norm(p.grad.detach(), norm_type).to(device) for p in parameters]), norm_type)
    if ms_isnan(total_norm) or ms_isinf(total_norm):
        if error_if_nonfinite:
            raise RuntimeError(
                f'The total norm of order {norm_type} for gradients from '
                '`parameters` is non-finite, so it cannot be clipped. To disable '
                'this error and scale the gradients by the non-finite norm anyway, '
                'set `error_if_nonfinite=False`')
        else:
            warnings.warn("Non-finite norm encountered in torch.nn.utils.clip_grad_norm_; continuing anyway. "
                          "Note that the default behavior will change in a future release to error out "
                          "if a non-finite total norm is encountered. At that point, setting "
                          "error_if_nonfinite=false will be required to retain the old behavior.",
                          FutureWarning, stacklevel=2)
    clip_coef = max_norm / (total_norm + 1e-6)
    if clip_coef < 1:
        for p in parameters:
            p.grad.detach().mul_(clip_coef.to(p.grad.device))
    return total_norm

#class LFramework(nn.Module):
class LFramework(mindspore.nn.Cell):
    def __init__(self, args, kg, mdl):
        super(LFramework, self).__init__()
        self.args = args
        self.data_dir = args.data_dir
        self.model_dir = args.model_dir
        self.model = args.model

        # Training hyperparameters
        self.batch_size = args.batch_size
        self.train_batch_size = args.train_batch_size
        self.dev_batch_size = args.dev_batch_size
        self.start_epoch = args.start_epoch
        self.num_epochs = args.num_epochs
        self.num_wait_epochs = args.num_wait_epochs
        self.num_peek_epochs = args.num_peek_epochs
        self.learning_rate = args.learning_rate
        self.grad_norm = args.grad_norm
        self.adam_beta1 = args.adam_beta1
        self.adam_beta2 = args.adam_beta2
        self.optim = None

        self.inference = not args.train
        self.run_analysis = args.run_analysis

        self.kg = kg
        self.mdl = mdl

        self.rl_module = args.rl_module
        logging.info('{} module created'.format(self.model))

        #mindspore->torch.cat()
        self.concat=mindspore.ops.Concat()
        self.ms_cast=mindspore.ops.Cast()

    def print_all_model_parameters(self):
        logging.info('Model Parameters')
        logging.info('--------------------------')
        #for name, param in self.named_parameters():
            #logging.info('Parameter %s: %s, require_grad = %s' % (name, str(param.size()), str(param.requires_grad)))
        for param in self.get_parameters():
            logging.info('Parameter %s: %s, require_grad = %s' % (param.name, str(param.shape), str(param.requires_grad)))
        #param_sizes = [param.numel() for param in self.parameters()]
        param_sizes = [np.prod(param.data.shape) for param in self.get_parameters()]
        logging.info('Total # parameters = {}'.format(sum(param_sizes)))
        logging.info('--------------------------')

    def run_train(self, train_data, dev_data, few_shot=False, adaptation=False, adaptation_relation=None):
        self.print_all_model_parameters()

        if self.optim is None:
            #self.optim = optim.Adam(
            #    filter(lambda p: p.requires_grad, self.parameters()), lr=self.learning_rate)
            self.optim = mindspore.nn.Adam(
                params=filter(lambda p: p.requires_grad, self.trainable_params()), learning_rate=self.learning_rate)

        # Track dev metrics changes
        best_dev_metrics = 0
        best_epoch_id = 0
        dev_metrics_history = []

        for epoch_id in range(self.start_epoch, self.num_epochs):
            logging.info('Epoch {}'.format(epoch_id))

            # Update model parameters
            #self.train()
            self.set_train()
            if self.rl_variation_tag.startswith('rs'):
                # fn => fact network
                #self.fn.eval()
                #self.fn_kg.eval()
                self.fn.set_train(False)
                self.fn_kg.set_train(False)
                if self.model.endswith('hypere'):
                    #self.fn_secondary_kg.eval()
                    self.fn_secondary_kg.set_train(False)
            self.batch_size = self.train_batch_size
            batch_losses = []
            entropies = []
            batch_losses_high = []
            entropies_high = []
            batch_losses_low = []
            entropies_low = []
            if self.run_analysis:
                rewards = None
                fns = None
                rewards_high = None
                rewards_low = None


            # 1. shuffle data
            random.shuffle(train_data)

            #2.train(MS)
            for example_id in tqdm(range(0, len(train_data), self.batch_size)):

                # gc.disable()    #禁止垃圾回收
                mini_batch = train_data[example_id:example_id + self.batch_size]

                #loss = self.loss_hrl(mini_batch)

                net_with_criterion = MyWithLossCell_PG(self)
                # manager = mindspore.nn.DynamicLossScaleUpdateCell(loss_scale_value=2 ** 12, scale_factor=2,
                #                                                   scale_window=10)
                # train_net = mindspore.nn.TrainOneStepWithLossScaleCell(net_with_criterion, optimizer=self.optim,
                #                                                        scale_sense=manager)
                train_net = MyTrainStepCell(net_with_criterion, self.optim)

                #loss_back = net_with_criterion(mini_batch)
                train_net(mini_batch)
                loss = net_with_criterion.consturc_complete()


                #ms_model=Model(network=self.rollout_hrl,loss_fn=self.loss_hrl,optimizer=self.optim)
                #ms_model.train(1,train_dataset=mini_batch)

                batch_losses_high.append(loss['print_loss_high'])
                batch_losses_low.append(loss['print_loss_low'])

                if 'entropy_high' in loss and 'entropy_low' in loss:
                    entropies_high.append(loss['entropy_high'])
                    entropies_low.append(loss['entropy_low'])

                if self.run_analysis:
                    if rewards_high is None:
                        rewards_high = loss['reward_high']
                    else:
                        #rewards_high = torch.cat([rewards_high, loss['reward_high']])
                        rewards_high = self.concat((rewards_high, loss['reward_high']))
                    if rewards_low is None:
                        rewards_low = loss['reward_low']
                    else:
                        #rewards_low = torch.cat([rewards_low, loss['reward_low']])
                        rewards_low = self.concat((rewards_low, loss['reward_low']))
                    if fns is None:
                        fns = loss['fn']
                    else:
                        #fns = torch.cat([fns, loss['fn']])
                        fns = self.concat((fns, loss['fn']))
                gc.collect()
            # # 2. train
            # # use tqdm to display process bar
            # for example_id in tqdm(range(0, len(train_data), self.batch_size)):
            #
            #     mini_batch = train_data[example_id:example_id + self.batch_size]
            #
            #     self.optim.zero_grad()
            #     loss = self.loss_hrl(mini_batch)
            #
            #     (loss['model_loss_high'] + loss['model_loss_low']).backward()
            #
            #     if self.grad_norm > 0:
            #         clip_grad_norm_(self.parameters(), self.grad_norm)
            #     self.optim.step()
            #
            #     batch_losses_high.append(loss['print_loss_high'])
            #     batch_losses_low.append(loss['print_loss_low'])
            #
            #     if 'entropy_high' in loss and 'entropy_low' in loss:
            #         entropies_high.append(loss['entropy_high'])
            #         entropies_low.append(loss['entropy_low'])
            #
            #     if self.run_analysis:
            #         if rewards_high is None:
            #             rewards_high = loss['reward_high']
            #         else:
            #             rewards_high = torch.cat([rewards_high, loss['reward_high']])
            #         if rewards_low is None:
            #             rewards_low = loss['reward_low']
            #         else:
            #             rewards_low = torch.cat([rewards_low, loss['reward_low']])
            #         if fns is None:
            #             fns = loss['fn']
            #         else:
            #             fns = torch.cat([fns, loss['fn']])

            # Check training statistics
            # batch_losses_high = mindspore.ops.Concat(1)(batch_losses_high)
            # batch_losses_low = mindspore.ops.Concat(1)(batch_losses_low)
            # batch_losses = mindspore.Tensor(batch_losses, mindspore.float32)
            # entropies = mindspore.Tensor(entropies, mindspore.float32)
            # entropies_high = mindspore.Tensor(entropies_high, mindspore.float32)
            # entropies_low = mindspore.Tensor(entropies_low, mindspore.float32)
            if self.rl_module == 'original':
                stdout_msg = 'Epoch {}: average training loss = {}'.format(epoch_id, np.mean(batch_losses))
                if entropies:
                    stdout_msg += ' entropy = {}'.format(np.mean(entropies))
            elif self.rl_module == 'hrl':
                stdout_msg = 'Epoch {}: average training loss_high = {}, loss_low = {}'. \
                    format(epoch_id, np.mean(batch_losses_high), np.mean(batch_losses_low))
                    #format(epoch_id, np.mean(batch_losses_high), np.mean(batch_losses_low))
                if entropies_high and entropies_low:
                    stdout_msg += '|| entropies_high = {}, entropies_low = {}'. \
                        format(np.mean(entropies_high), np.mean(entropies_low))
                        #format(np.mean(entropies_high), np.mean(entropies_low))
            logging.info(stdout_msg)
            if adaptation:
                if epoch_id % self.num_wait_epochs == 0 or epoch_id == self.num_epochs - 1:
                    self.save_checkpoint(checkpoint_id=epoch_id, epoch_id=epoch_id, relation=adaptation_relation)
            elif few_shot:
                if epoch_id % self.num_wait_epochs == 0 or epoch_id == self.num_epochs - 1:
                    self.save_checkpoint(checkpoint_id=epoch_id, epoch_id=epoch_id)
            else:
                self.save_checkpoint(checkpoint_id=epoch_id, epoch_id=epoch_id)

            if self.run_analysis:
                logging.info('* Analysis: # path types seen = {}'.format(self.num_path_types))
                if self.rl_module == 'original':
                    num_hits = float(rewards.sum())
                    hit_ratio = num_hits / len(rewards)
                elif self.rl_module == 'hrl':
                    #num_hits = float(rewards_high.sum())
                    #hit_ratio = num_hits / len(rewards_high)
                    num_hits = mindspore.ops.Cast()(rewards_high.sum(), mindspore.float32)
                    hit_ratio = num_hits / rewards_high.size
                logging.info('* Analysis: # hits = {} ({})'.format(num_hits, hit_ratio))
                #num_fns = float(fns.sum())
                #fn_ratio = num_fns / len(fns)
                num_fns = mindspore.ops.Cast()(fns.sum(), mindspore.float32)
                fn_ratio = num_fns / fns.size
                logging.info('* Analysis: false negative ratio = {}'.format(fn_ratio))

            # Check dev set performance
            if self.run_analysis or (epoch_id > 0 and epoch_id % self.num_peek_epochs == 0):
                #self.eval()
                self.set_train(False)
                self.batch_size = self.dev_batch_size
                #ms.constract
                dev_scores = self.forward(dev_data, verbose=False)  # run dev data
                logging.info('Dev set performance: (correct evaluation)')
                _, _, _, _, mrr = src.eval.hits_and_ranks(dev_data, dev_scores, self.kg.dev_objects, verbose=True)
                metrics = mrr
                logging.info('Dev set performance: (include test set labels)')
                src.eval.hits_and_ranks(dev_data, dev_scores, self.kg.all_objects, verbose=True)
                # Action dropout anneaking
                if self.model.startswith('point'):
                    eta = self.action_dropout_anneal_interval
                    if len(dev_metrics_history) > eta and metrics < min(dev_metrics_history[-eta:]):
                        old_action_dropout_rate = self.action_dropout_rate
                        self.action_dropout_rate *= self.action_dropout_anneal_factor
                        logging.info('Decreasing action dropout rate: {} -> {}'.format(
                            old_action_dropout_rate, self.action_dropout_rate))
                # Save checkpoint
                if metrics > best_dev_metrics:
                    self.save_checkpoint(checkpoint_id=epoch_id, epoch_id=epoch_id, is_best=True)
                    best_dev_metrics = metrics
                    best_epoch_id = epoch_id
                    with open(os.path.join(self.model_dir, 'best_dev_iteration.dat'), 'w') as o_f:
                        o_f.write('{}'.format(epoch_id))
                else:
                    # Early stopping
                    if epoch_id >= self.num_wait_epochs and metrics < np.mean(
                            dev_metrics_history[-self.num_wait_epochs:]):
                        print("early stopping")
                        logging.info("early stopping")
                        break

                logging.info(
                    "best dev iteration: {} => best_metrics: {:.3f}".format(best_epoch_id, best_dev_metrics))
                dev_metrics_history.append(metrics)
                if self.run_analysis:
                    num_path_types_file = os.path.join(self.model_dir, 'num_path_types.dat')
                    dev_metrics_file = os.path.join(self.model_dir, 'dev_metrics.dat')
                    hit_ratio_file = os.path.join(self.model_dir, 'hit_ratio.dat')
                    fn_ratio_file = os.path.join(self.model_dir, 'fn_ratio.dat')
                    if epoch_id == 0:
                        with open(num_path_types_file, 'w') as o_f:
                            o_f.write('{}\n'.format(self.num_path_types))
                        with open(dev_metrics_file, 'w') as o_f:
                            o_f.write('{}\n'.format(metrics))
                        with open(hit_ratio_file, 'w') as o_f:
                            o_f.write('{}\n'.format(hit_ratio))
                        with open(fn_ratio_file, 'w') as o_f:
                            o_f.write('{}\n'.format(fn_ratio))
                    else:
                        with open(num_path_types_file, 'a') as o_f:
                            o_f.write('{}\n'.format(self.num_path_types))
                        with open(dev_metrics_file, 'a') as o_f:
                            o_f.write('{}\n'.format(metrics))
                        with open(hit_ratio_file, 'a') as o_f:
                            o_f.write('{}\n'.format(hit_ratio))
                        with open(fn_ratio_file, 'a') as o_f:
                            o_f.write('{}\n'.format(fn_ratio))

    def forward(self, examples, verbose=False):
        pred_scores = []
        for example_id in tqdm(range(0, len(examples), self.batch_size)):
            mini_batch = examples[example_id:example_id + self.batch_size]
            mini_batch_size = len(mini_batch)
            if len(mini_batch) < self.batch_size:
                self.make_full_batch(mini_batch, self.batch_size)
            pred_score = self.predict(mini_batch, verbose=verbose)  # size([self.mini_batch_size, kg.num_entities])
            pred_scores.append(pred_score[:mini_batch_size])
        #scores = torch.cat(pred_scores)
        scores = self.concat(pred_scores)
        return scores

    #ms
    def construct(self, examples, verbose=False):
        out=self.forward(examples, verbose)
        return out

    def format_batch(self, batch_data, num_labels=-1, num_tiles=1):
        """
        Convert batched tuples to the tensors accepted by the NN.
        """

        def convert_to_binary_multi_subject(e1):
            #e1_label = zeros_var_cuda([len(e1), num_labels])
            e1_label = mindspore.Parameter((mindspore.ops.Zeros()(len(e1), num_labels), mindspore.float32),
                                             requires_grad=False)
            for i in range(len(e1)):
                e1_label[i][e1[i]] = 1
            return e1_label

        def convert_to_binary_multi_object(e2):
            #e2_label = zeros_var_cuda([len(e2), num_labels])
            e2_label = mindspore.Parameter((mindspore.ops.Zeros()(len(e2), num_labels), mindspore.float32),
                                           requires_grad=False)
            for i in range(len(e2)):
                e2_label[i][e2[i]] = 1
            return e2_label

        batch_e1, batch_e2, batch_r = [], [], []
        for i in range(len(batch_data)):
            e1, e2, r = batch_data[i]
            batch_e1.append(e1)
            batch_e2.append(e2)
            batch_r.append(r)
        #batch_e1 = var_cuda(torch.LongTensor(batch_e1), requires_grad=False)
        #batch_r = var_cuda(torch.LongTensor(batch_r), requires_grad=False)
        batch_e1 = mindspore.Tensor(batch_e1, mindspore.int32)
        batch_r = mindspore.Tensor(batch_r, mindspore.int32)
        # if tail entity is a list
        #if type(batch_e2[0]) is list:
        if type(batch_e2[0]) is list:
            # convert to a 2-dimension ones array
            batch_e2 = convert_to_binary_multi_object(batch_e2)
        elif type(batch_e1[0]) is list:
            batch_e1 = convert_to_binary_multi_subject(batch_e1)
        else:
            #batch_e2 = var_cuda(torch.LongTensor(batch_e2), requires_grad=False)
            batch_e2 = mindspore.Tensor(batch_e2, mindspore.int32)
        # Rollout multiple times for each example
        if num_tiles > 1:
            batch_e1 = ops.tile_along_beam(batch_e1, num_tiles)
            batch_r = ops.tile_along_beam(batch_r, num_tiles)
            batch_e2 = ops.tile_along_beam(batch_e2, num_tiles)
        return batch_e1, batch_e2, batch_r

    def make_full_batch(self, mini_batch, batch_size, multi_answers=False):
        dummy_e = self.kg.dummy_e
        dummy_r = self.kg.dummy_r
        if multi_answers:
            dummy_example = (dummy_e, [dummy_e], dummy_r)
        else:
            dummy_example = (dummy_e, dummy_e, dummy_r)
        for _ in range(batch_size - len(mini_batch)):
            mini_batch.append(dummy_example)

    def save_checkpoint(self, checkpoint_id, epoch_id=None, is_best=False, relation=False):
        """
        Save model checkpoint.
        :param checkpoint_id: Model checkpoint index assigned by training loop.
        :param epoch_id: Model epoch index assigned by training loop.
        :param is_best: if set, the model being saved is the best model on dev set.
        """
        checkpoint_dict = dict()
        #checkpoint_dict['state_dict'] = self.state_dict()
        checkpoint_dict['state_dict'] = self.parameters_dict()
        checkpoint_dict['epoch_id'] = epoch_id
        checkpoint_dict_list=[checkpoint_dict]

        if relation:
            #out_tar = os.path.join(self.model_dir, 'checkpoint-{}-{}.tar'.format(checkpoint_id, relation))
            out_ckpt = os.path.join(self.model_dir, 'checkpoint-{}-{}.ckpt'.format(checkpoint_id, relation))
        else:
            #out_tar = os.path.join(self.model_dir, 'checkpoint-{}.tar'.format(checkpoint_id))
            out_ckpt = os.path.join(self.model_dir, 'checkpoint-{}.ckpt'.format(checkpoint_id))
        if is_best:
            #best_path = os.path.join(self.model_dir, 'model_best.tar')
            #shutil.copyfile(out_tar, best_path)
            best_path = os.path.join(self.model_dir, 'model_best.ckpt')
            shutil.copyfile(out_ckpt, best_path)
            print('=> best model updated \'{}\''.format(best_path))
        else:
            #torch.save(checkpoint_dict, out_tar)
            #print('=> saving checkpoint to \'{}\''.format(out_tar))
            mindspore.save_checkpoint(self, out_ckpt)
            print('=> saving checkpoint to \'{}\''.format(out_ckpt))

    def load_checkpoint(self, input_file, adaptation=True):
        """
        Load model checkpoint.
        :param n: Neural network module.
        :param kg: Knowledge graph module.
        :param input_file: Checkpoint file path.
        """
        if os.path.isfile(input_file):
            print('=> loading checkpoint \'{}\''.format(input_file))
            #checkpoint = torch.load(input_file, map_location="cuda:{}".format(self.args.gpu))
            #print(checkpoint['state_dict'].keys())
            #self.load_state_dict(checkpoint['state_dict'])
            covert_model(input_file)
            checkpoint_state_dict_value = mindspore.load_checkpoint(input_file)
            print(checkpoint_state_dict_value)
            mindspore.load_param_into_net(self,checkpoint_state_dict_value)
            if not self.inference and not adaptation:
                #self.start_epoch = checkpoint['epoch_id'] + 1
                assert (self.start_epoch <= self.num_epochs)
        else:
            print('=> no checkpoint found at \'{}\''.format(input_file))

    def export_to_embedding_projector(self):
        """
        Export knowledge base embeddings into .tsv files accepted by the Tensorflow Embedding Projector.
        """
        vector_path = os.path.join(self.model_dir, 'vector.tsv')
        meta_data_path = os.path.join(self.model_dir, 'metadata.tsv')
        v_o_f = open(vector_path, 'w')
        m_o_f = open(meta_data_path, 'w')
        for r in self.kg.relation2id:
            if r.endswith('_inv'):
                continue
            r_id = self.kg.relation2id[r]
            R = self.kg.relation_embeddings.weight[r_id]
            r_print = ''
            for i in range(len(R)):
                r_print += '{}\t'.format(float(R[i]))
            v_o_f.write('{}\n'.format(r_print.strip()))
            m_o_f.write('{}\n'.format(r))
            #print(r, '{}'.format(float(R.norm())))
            print(r, '{}'.format(float(mindspore.nn.Norm()(R))))
        v_o_f.close()
        m_o_f.close()
        print('KG embeddings exported to {}'.format(vector_path))
        print('KG meta data exported to {}'.format(meta_data_path))

    @property
    def rl_variation_tag(self):
        parts = self.model.split('.')
        if len(parts) > 1:
            return parts[1]
        else:
            return ''
