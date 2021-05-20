# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Functions and classes related to optimization (weight updates)."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import re
import tensorflow as tf
import horovod.tensorflow as hvd
import numpy as np

def create_optimizer(loss, init_lr, num_train_steps, num_warmup_steps, num_gpus, use_fp16=False):
  """Creates an optimizer training op."""
  global_step = tf.train.get_or_create_global_step()

  decayed_learning_rate_at_crossover_point = init_lr * (1.0 - float(num_warmup_steps) / float(num_train_steps))
  adjusted_init_lr = init_lr * (init_lr / decayed_learning_rate_at_crossover_point)
  tf.logging.info('decayed_learning_rate_at_crossover_point = %s, adjusted_init_lr = %s' % (
  str(decayed_learning_rate_at_crossover_point), str(adjusted_init_lr)))

  #learning_rate = tf.constant(value=init_lr, shape=[], dtype=tf.float32)
  learning_rate = tf.constant(value=adjusted_init_lr, shape=[], dtype=tf.float32)
  
  # Implements linear decay of the learning rate.
  learning_rate = tf.train.polynomial_decay(
      learning_rate,
      global_step,
      num_train_steps,
      end_learning_rate=0.0,
      power=1.0,
      cycle=False)

  # Implements linear warmup. I.e., if global_step < num_warmup_steps, the
  # learning rate will be `global_step/num_warmup_steps * init_lr`.
  if num_warmup_steps:
    global_steps_int = tf.cast(global_step, tf.int32)
    warmup_steps_int = tf.constant(num_warmup_steps, dtype=tf.int32)

    global_steps_float = tf.cast(global_steps_int, tf.float32)
    warmup_steps_float = tf.cast(warmup_steps_int, tf.float32)

    warmup_percent_done = global_steps_float / warmup_steps_float
    warmup_learning_rate = init_lr * warmup_percent_done

    is_warmup = tf.cast(global_steps_int < warmup_steps_int, tf.float32)
    learning_rate = (
        (1.0 - is_warmup) * learning_rate + is_warmup * warmup_learning_rate)

  learning_rate = learning_rate * float(np.sqrt(hvd.size()))
  # create tensor name for logging
  tf.identity(learning_rate, name='learning_rate')
  tf.summary.scalar('learning_rate', learning_rate)

  # It is recommended that you use this optimizer for fine tuning, since this
  # is how the model was trained (note that the Adam m/v variables are NOT
  # loaded from init_checkpoint.)
  optimizer = AdamWeightDecayOptimizer(
      learning_rate=learning_rate,
      weight_decay_rate=0.01,
      beta_1=0.9,
      beta_2=0.999,
      epsilon=1e-6,
      exclude_from_weight_decay=["LayerNorm", "layer_norm", "bias"])

  tvars = tf.trainable_variables()

  optimizer = hvd.DistributedOptimizer(optimizer, sparse_as_dense=True)
  #optimizer = hvd.DistributedOptimizer(optimizer, sparse_as_dense=True, backward_passes_per_step=4) # compression=hvd.Compression.fp16)

  grads_vars = optimizer.compute_gradients(loss, tvars, use_fp16=use_fp16)

  # we should call the clip_by_global_norm after the allreduece
  grads = [grad_var[0] for grad_var in grads_vars]
  vars  = [grad_var[1] for grad_var in grads_vars]

  grads_fp32 = []
  for grad in grads:
    if grad is not None:
      grads_fp32.append(tf.cast(grad, tf.float32))
    else:
      grads_fp32.append(None)

  (grads, _) = tf.clip_by_global_norm(grads_fp32, clip_norm=1.0)

  grads_vars = zip(grads, vars)

  train_op = optimizer.apply_gradients(
      grads_and_vars=grads_vars, global_step=global_step, use_fp16=use_fp16)

  new_global_step = global_step + 1
  new_global_step = tf.identity(new_global_step, name='step_update')
  train_op = tf.group(train_op, [global_step.assign(new_global_step)])
  return train_op

class AdamWeightDecayOptimizer(tf.train.Optimizer):
  """A basic Adam optimizer that includes "correct" L2 weight decay."""

  GATE_OP = 1
  def __init__(self,
               learning_rate,
               weight_decay_rate=0.0,
               beta_1=0.9,
               beta_2=0.999,
               epsilon=1e-6,
               exclude_from_weight_decay=None,
               name="AdamWeightDecayOptimizer"):
    """Constructs a AdamWeightDecayOptimizer."""
    super(AdamWeightDecayOptimizer, self).__init__(False, name)

    self.learning_rate = learning_rate
    self.weight_decay_rate = weight_decay_rate
    self.beta_1 = beta_1
    self.beta_2 = beta_2
    self.epsilon = epsilon
    self.exclude_from_weight_decay = exclude_from_weight_decay

  def compute_gradients(self, loss, var_list=None,
                        gate_gradients=GATE_OP,
                        aggregation_method=None,
                        colocate_gradients_with_ops=False,
                        grad_loss=None,
                        use_fp16=False):
    # loss scale may need adjustment for new datasets
    loss_scale = 128.0 if use_fp16 else 1.0
    if loss_scale != 1.0:
      grads = tf.gradients(loss * loss_scale, var_list)
      grads = [tf.math.scalar_mul(1.0 / loss_scale, grad) for grad in grads]
    else:
      grads = tf.gradients(loss, var_list)

    grads_and_vars = zip(grads, var_list)
    return grads_and_vars

  def apply_gradients(self, grads_and_vars, global_step=None, name=None,
      use_fp16=False):
    """See base class."""
    assignments = []
    for (grad, param) in grads_and_vars:
      if grad is None or param is None:
        continue

      param_name = self._get_variable_name(param.name)
      has_shadow = use_fp16 and param.dtype.base_dtype != tf.float32
      if has_shadow:
        # create shadow fp32 weights for fp16 variable
        param_fp32 = tf.get_variable(
            name=param_name + "/shadow",
            dtype=tf.float32,
            trainable=False,
            initializer=tf.cast(param.initialized_value(),tf.float32))
      else:
        param_fp32 = param

      m = tf.get_variable(
          name=param_name + "/adam_m",
          shape=param.shape.as_list(),
          dtype=tf.float32,
          trainable=False,
          initializer=tf.zeros_initializer())
      v = tf.get_variable(
          name=param_name + "/adam_v",
          shape=param.shape.as_list(),
          dtype=tf.float32,
          trainable=False,
          initializer=tf.zeros_initializer())

      # Standard Adam update.
      next_m = (
          tf.multiply(self.beta_1, m) + tf.multiply(1.0 - self.beta_1, grad))
      next_v = (
          tf.multiply(self.beta_2, v) + tf.multiply(1.0 - self.beta_2,
                                                    tf.square(grad)))

      update = next_m / (tf.sqrt(next_v) + self.epsilon)

      # Just adding the square of the weights to the loss function is *not*
      # the correct way of using L2 regularization/weight decay with Adam,
      # since that will interact with the m and v parameters in strange ways.
      #
      # Instead we want ot decay the weights in a manner that doesn't interact
      # with the m/v parameters. This is equivalent to adding the square
      # of the weights to the loss with plain (non-momentum) SGD.
      if self._do_use_weight_decay(param_name):
        update += self.weight_decay_rate * param_fp32

      update_with_lr = self.learning_rate * update

      next_param = param_fp32 - update_with_lr

      if has_shadow:
        # cast shadow fp32 weights to fp16 and assign to trainable variable
        param.assign(tf.cast(next_param, param.dtype.base_dtype))
      assignments.extend(
          [param_fp32.assign(next_param),
           m.assign(next_m),
           v.assign(next_v)])
    return tf.group(*assignments, name=name)

  def _do_use_weight_decay(self, param_name):
    """Whether to use L2 weight decay for `param_name`."""
    if not self.weight_decay_rate:
      return False
    if self.exclude_from_weight_decay:
      for r in self.exclude_from_weight_decay:
        if re.search(r, param_name) is not None:
          return False
    return True

  def _get_variable_name(self, param_name):
    """Get the variable name from the tensor name."""
    m = re.match("^(.*):\\d+$", param_name)
    if m is not None:
      param_name = m.group(1)
    return param_name
