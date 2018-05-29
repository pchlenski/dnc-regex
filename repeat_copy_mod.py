# Copyright 2017 Google Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""A repeat copy task."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import numpy as np
import sonnet as snt
import tensorflow as tf
import re     # PC: For regular expression matching
import random # PC: To sample from all_strings

DatasetTensors = collections.namedtuple('DatasetTensors', ('observations',
                                                           'target', 'mask'))


def masked_sigmoid_cross_entropy(logits,
                                 target,
                                 mask,
                                 time_average=False,
                                 log_prob_in_bits=False):
  """Adds ops to graph which compute the (scalar) NLL of the target sequence.

  The logits parametrize independent bernoulli distributions per time-step and
  per batch element, and irrelevant time/batch elements are masked out by the
  mask tensor.

  Args:
    logits: `Tensor` of activations for which sigmoid(`logits`) gives the
        bernoulli parameter.
    target: time-major `Tensor` of target.
    mask: time-major `Tensor` to be multiplied elementwise with cost T x B cost
        masking out irrelevant time-steps.
    time_average: optionally average over the time dimension (sum by default).
    log_prob_in_bits: iff True express log-probabilities in bits (default nats).

  Returns:
    A `Tensor` representing the log-probability of the target.
  """
  xent = tf.nn.sigmoid_cross_entropy_with_logits(labels=target, logits=logits)
  loss_time_batch = tf.reduce_sum(xent, axis=2)
  loss_batch = tf.reduce_sum(loss_time_batch * mask, axis=0)

  batch_size = tf.cast(tf.shape(logits)[1], dtype=loss_time_batch.dtype)

  if time_average:
    mask_count = tf.reduce_sum(mask, axis=0)
    loss_batch /= (mask_count + np.finfo(np.float32).eps)

  loss = tf.reduce_sum(loss_batch) / batch_size
  if log_prob_in_bits:
    loss /= tf.log(2.)

  return loss


def bitstring_readable(data, batch_size, model_output=None, whole_batch=False):
  """Produce a human readable representation of the sequences in data.

  Args:
    data: data to be visualised
    batch_size: size of batch
    model_output: optional model output tensor to visualize alongside data.
    whole_batch: whether to visualise the whole batch. Only the first sample
        will be visualized if False

  Returns:
    A string used to visualise the data batch
  """

  def _readable(datum):
    return '+' + ' '.join(['-' if x == 0 else '%d' % x for x in datum]) + '+'

  obs_batch = data.observations
  targ_batch = data.target

  iterate_over = range(batch_size) if whole_batch else range(1)

  batch_strings = []
  for batch_index in iterate_over:
    obs = obs_batch[:, batch_index, :]
    targ = targ_batch[:, batch_index, :]

    obs_channels = range(obs.shape[1])
    targ_channels = range(targ.shape[1])
    obs_channel_strings = [_readable(obs[:, i]) for i in obs_channels]
    targ_channel_strings = [_readable(targ[:, i]) for i in targ_channels]

    readable_obs = 'Observations:\n' + '\n'.join(obs_channel_strings)
    readable_targ = 'Targets:\n' + '\n'.join(targ_channel_strings)
    strings = [readable_obs, readable_targ]

    if model_output is not None:
      output = model_output[:, batch_index, :]
      output_strings = [_readable(output[:, i]) for i in targ_channels]
      strings.append('Model Output:\n' + '\n'.join(output_strings))

    batch_strings.append('\n\n'.join(strings))

  return '\n' + '\n\n\n\n'.join(batch_strings)


class RepeatCopy(snt.AbstractModule):
  """Sequence data generator for the task of repeating a random binary pattern.

  When called, an instance of this class will return a tuple of tensorflow ops
  (obs, targ, mask), representing an input sequence, target sequence, and
  binary mask. Each of these ops produces tensors whose first two dimensions
  represent sequence position and batch index respectively. The value in
  mask[t, b] is equal to 1 iff a prediction about targ[t, b, :] should be
  penalized and 0 otherwise.

  For each realisation from this generator, the observation sequence is
  comprised of I.I.D. uniform-random binary vectors (and some flags).

  The target sequence is comprised of this binary pattern repeated
  some number of times (and some flags). Before explaining in more detail,
  let's examine the setup pictorially for a single batch element:

  ```none
  Note: blank space represents 0.

  time ------------------------------------------>

                +------------+
  observation:  |  1011001   |
                | 2          | 'regular expression' indicator
                |         3  | terminal
                +------------+
  Note that values are actually one-hot vectors
  Inputs are of size [batch size]x[max string length + 1 (for regex indicator)] + [one-hot vector length]

                +------------+
  mask:         | 0000000001 |
                +------------+

                +------------+
  target:       |          1 |
                +------------+

  The length of the random pattern and the number of times it is repeated
  in the target are both discrete random variables distributed according to
  uniform distributions whose parameters are configured at construction time.

  The obs sequence has two extra channels (components in the trailing dimension)
  which are used for flags. One channel is marked with a 1 at the first time
  step and is otherwise equal to 0. The other extra channel is zero until the
  binary pattern to be repeated ends. At this point, it contains an encoding of
  the number of times the observation pattern should be repeated. Rather than
  simply providing this integer number directly, it is normalised so that
  a neural network may have an easier time representing the number of
  repetitions internally. To allow a network to be readily evaluated on
  instances of this task with greater numbers of repetitions, the range with
  respect to which this encoding is normalised is also configurable by the user.

  As in the diagram, the target sequence is offset to begin directly after the
  observation sequence; both sequences are padded with zeros to accomplish this,
  resulting in their lengths being equal. Additional padding is done at the end
  so that all sequences in a minibatch represent tensors with the same shape.
  """

  def __init__(
      self,
      num_bits=6,
      batch_size=1,
      min_length=1,
      max_length=1,
      min_repeats=1,
      max_repeats=2,
      norm_max=10,
      log_prob_in_bits=False,
      time_average_cost=False,
      name='repeat_copy',):
    """Creates an instance of RepeatCopy task.

    Args:
      name: A name for the generator instance (for name scope purposes).
      num_bits: The dimensionality of each random binary vector.
      batch_size: Minibatch size per realization.
      min_length: Lower limit on number of random binary vectors in the
          observation pattern.
      max_length: Upper limit on number of random binary vectors in the
          observation pattern.
      min_repeats: Lower limit on number of times the obervation pattern
          is repeated in targ.
      max_repeats: Upper limit on number of times the observation pattern
          is repeated in targ.
      norm_max: Upper limit on uniform distribution w.r.t which the encoding
          of the number of repetitions presented in the observation sequence
          is normalised.
      log_prob_in_bits: By default, log probabilities are expressed in units of
          nats. If true, express log probabilities in bits.
      time_average_cost: If true, the cost at each time step will be
          divided by the `true`, sequence length, the number of non-masked time
          steps, in each sequence before any subsequent reduction over the time
          and batch dimensions.
    """
    super(RepeatCopy, self).__init__(name=name)

    self._batch_size = batch_size
    self._num_bits = num_bits
    self._min_length = min_length
    self._max_length = max_length
    self._min_repeats = min_repeats
    self._max_repeats = max_repeats
    self._norm_max = norm_max
    self._log_prob_in_bits = log_prob_in_bits
    self._time_average_cost = time_average_cost

  def _normalise(self, val):
    return val / self._norm_max

  def _unnormalise(self, val):
    return val * self._norm_max

  @property
  def time_average_cost(self):
    return self._time_average_cost

  @property
  def log_prob_in_bits(self):
    return self._log_prob_in_bits

  @property
  def num_bits(self):
    """The dimensionality of each random binary vector in a pattern."""
    return self._num_bits

  @property
  def target_size(self):
    """The dimensionality of the target tensor."""
    return 8 # PC: Needs to be set manually

  @property
  def batch_size(self):
    return self._batch_size

  def _build(self):
    """Implements build method which adds ops to graph."""

    # PC: This is the generating function of interest - from data.py
    def gen_all(n, alph):
      n = n - 1
      all_strings = list(alph)
      for i in range(n):
        # Generate all sequences of size up to i+1:
        new_sets = []
        for letter in list(alph): # list prevents mere referencing
          # Create a copy of all_strings with letter appended
          new_sets += [ letter + element for element in all_strings ]
        # Add new_sets back to all_strings
        all_strings += new_sets
        # Loop again
      return set(all_strings) # Set eliminates necessary redundancies

    # PC: This is the regex matching function - from data.py
    def match_bin(regex, string):
      value = None
      # Catches the cases where regex doesn't match at all
      try: 
        match = re.match(regex, string).group()
      except AttributeError:
        return 0
      # Ensure exact rather than partial matching
      if match == string: 
        return 1
      else: 
        return 0

    # Set parameters
    n          = 8
    alphabet   = ['a','b']
    regex_list = ['a*', 'b*', '.*']

    # PC: Added this for convenience. 
    # PC: A vector of [alphabet length] + [regex list length] + [terminal length]
    one_hot_key   = regex_list + alphabet + ['BLANK'] + ['END']
    l             = (len(one_hot_key))     # PC: Blank = l-2, End = l-1
    one_hot_blank = [0] * (l + 1)          # PC: Last row is for target only

    # Generate data
    all_strings = gen_all(n, alphabet)

    # short-hand for private fields.
    min_length, max_length = 1, n + 2                             # PC: old values are self._min_length, self._max_length
    min_reps, max_reps     = self._min_repeats, self._max_repeats # PC: deprecated 
    num_bits               = l + 1                                # PC: old value is self.num_bits
    batch_size             = self.batch_size

    # PC: We reserve one dimension for regex indicator, padding, and terminal character
    full_obs_size           = num_bits
    # PC: Just a 1 or a 0, but in a vector
    full_targ_size          = num_bits
    start_end_flag_idx      = full_obs_size - 2
    num_repeats_channel_idx = full_obs_size - 1

    # PC: Not needed for this code
    # # Samples each batch index's sequence length and the number of repeats.
    # sub_seq_length_batch = tf.random_uniform(
    #     [batch_size], minval=min_length, maxval=max_length + 1, dtype=tf.int32)
    # num_repeats_batch = tf.random_uniform(
    #     [batch_size], minval=min_reps, maxval=max_reps + 1, dtype=tf.int32)

    # Pads all the batches to have the same total sequence length.
    # total_length_batch = sub_seq_length_batch * (num_repeats_batch + 1) + 3
    # max_length_batch = tf.reduce_max(total_length_batch)
    # residual_length_batch = max_length_batch - total_length_batch

    obs_batch_shape        = [max_length + 1, batch_size, full_obs_size]  # PC: Changed max_length_batch to max_length, added 1
    targ_batch_shape       = [max_length + 1, batch_size, full_targ_size] # PC: Ditto above
    mask_batch_trans_shape = [batch_size, max_length + 1]                 # PC: Ditto

    obs_tensors  = []
    targ_tensors = []
    mask_tensors = []

    # Generates patterns for each batch element independently.
    for batch_index in range(batch_size):
      # PC: Don't need this anymore
      # sub_seq_len = sub_seq_length_batch[batch_index]
      # num_reps = num_repeats_batch[batch_index]

      # The observation pattern is a sequence of random binary vectors.
      # obs_pattern_shape = [sub_seq_len, num_bits]
      # obs_pattern = tf.cast(
      #     tf.random_uniform(
      #         obs_pattern_shape, minval=0, maxval=2, dtype=tf.int32),
      #     tf.float32)

      # PC: The new observation pattern is a matrix of the following form:
      '''
      OBSERVATIONS: #########
              +-------------+
      regex 1 | 1           |
      regex 2 |             |
      ...     |             |
      alpha 1 |  1 11       |
      alpha 2 |   1  111    |
      ...     |             |
      padding |         1   |
      end     |          1  |
      target  |             |
              +-------------+

      TARGET: ###############
              +-------------+
      ...     |             |
      target  |           1 |
              +-------------+      
      '''

      # # The target pattern is the observation pattern repeated n times.
      # # Some reshaping is required to accomplish the tiling.
      # targ_pattern_shape = [sub_seq_len * num_reps, num_bits]
      # flat_obs_pattern = tf.reshape(obs_pattern, [-1])
      # flat_targ_pattern = tf.tile(flat_obs_pattern, tf.stack([num_reps]))
      # targ_pattern = tf.reshape(flat_targ_pattern, targ_pattern_shape)

      # # Expand the obs_pattern to have two extra channels for flags.
      # # Concatenate start flag and num_reps flag to the sequence.
      # obs_flag_channel_pad = tf.zeros([sub_seq_len, 2])
      # obs_start_flag = tf.one_hot(
      #     [start_end_flag_idx], full_obs_size, on_value=1., off_value=0.)
      # num_reps_flag = tf.one_hot(
      #     [num_repeats_channel_idx],
      #     full_obs_size,
      #     on_value=self._normalise(tf.cast(num_reps, tf.float32)),
      #     off_value=0.)

      # # note the concatenation dimensions.
      # obs = tf.concat([obs_pattern, obs_flag_channel_pad], 1)
      # obs = tf.concat([obs_start_flag, obs], 0)
      # obs = tf.concat([obs, num_reps_flag], 0)

      # PC: Pick a string and a regex
      this_str  = random.sample(all_strings, 1)[0] # PC: Use sample for sets, returns a list so take [0] index
      this_re   = random.choice(regex_list)

      # PC: Determine match
      targ      = match_bin(this_re, this_str)
      targ_list = [0] * l + [targ]
      targ_arry = np.array(targ_list).T
      targ_tens = tf.convert_to_tensor(targ_arry, dtype=tf.float32)
      targ_tens = tf.reshape(targ_tens, [full_obs_size, 1]) # PC: Fixing a shape mismatch error

      # PC: Generate the array of zeroes
      obs_array = np.array([one_hot_blank] * (max_length)).T
      # PC: Each observation is an array of [full_obs_size] x [max_length]

      # PC: Populate the array
      m = one_hot_key.index(this_re)

      obs_array[m][0]     = 1 # Set regex marker
      obs_array[l-1][n+1] = 1 # Set end marker

      # PC: Get a vectorized string
      str_vectorized      = [ one_hot_key.index(x) for x in this_str ]

      # PC: Pad with blanks
      str_vectorized      = str_vectorized + [l-2]*(max_length - len(str_vectorized)-2)

      for k in range(len(str_vectorized)):
        obs_array[str_vectorized[k]][k+1] = 1 # PC: Populate the middle of the array

      # PC: Pass to tensorflow
      obs = tf.convert_to_tensor(obs_array, dtype=tf.float32)

      # PC: Our target is much simpler, so we can skip all this stuff
      # # Now do the same for the targ_pattern (it only has one extra channel).
      # targ_flag_channel_pad = tf.zeros([full_obs_size, 1]) # PC: used to be tf.zeros([sub_seq_len * num_reps, 1])
      # targ_end_flag = tf.one_hot(
      #     [start_end_flag_idx], full_targ_size, on_value=1., off_value=0.)
      # targ = tf.concat([targ_pattern, targ_flag_channel_pad], 1)
      # targ = tf.concat([targ, targ_end_flag], 0)

      # Concatenate zeros at end of obs and begining of targ.
      # This aligns them s.t. the target begins as soon as the obs ends.
      obs_end_pad    = tf.zeros([l + 1, 1],  dtype=tf.float32)         # PC: Was tf.zeros([sub_seq_len * num_reps + 1, full_obs_size])
      targ_start_pad = tf.zeros([l + 1, max_length], dtype=tf.float32) # PC: Was tf.zeros([sub_seq_len + 2, full_targ_size])

      # The mask is zero during the obs and one during the targ.
      mask_off = tf.zeros([max_length], dtype=tf.float32) # PC: Was tf.zeros([sub_seq_len + 2])
      mask_on  = tf.ones([1], dtype=tf.float32)           # PC: Was tf.ones([sub_seq_len * num_reps + 1])

      obs  = tf.concat([obs, obs_end_pad], 1) # PC: Changed axis from 0 to 1
      targ = tf.concat([targ_start_pad, targ_tens], 1)
      mask = tf.concat([mask_off, mask_on], 0)

      # PC: Not needed?
      obs_tensors.append(obs)
      targ_tensors.append(targ)
      mask_tensors.append(mask)

    # PC: I don't think we need any of this, since we don't have variable lengths across batches.
    # End the loop over batch index.
    # Compute how much zero padding is needed to make tensors sequences
    # the same length for all batch elements.
    # residual_obs_pad = [
    #     tf.zeros([residual_length_batch[i], full_obs_size], dtype=tf.float32)  # PC: Made float32
    #     for i in range(batch_size)
    # ]
    # residual_targ_pad = [
    #     tf.zeros([residual_length_batch[i], full_targ_size], dtype=tf.float32) # PC: Made float32
    #     for i in range(batch_size)
    # ]
    # residual_mask_pad = [
    #     tf.zeros([residual_length_batch[i]], dtype=tf.float32) for i in range(batch_size) # PC: Made float32
    # ]

    # # Concatenate the pad to each batch element.
    # obs_tensors = [
    #     tf.concat([o, p], 0) for o, p in zip(obs_tensors, residual_obs_pad)
    # ]
    # targ_tensors = [
    #     tf.concat([t, p], 0) for t, p in zip(targ_tensors, residual_targ_pad)
    # ]
    # mask_tensors = [
    #     tf.concat([m, p], 0) for m, p in zip(mask_tensors, residual_mask_pad)
    # ]

    # Concatenate each batch element into a single tensor.
    obs  = tf.reshape(tf.concat(obs_tensors, 1), obs_batch_shape) # PC: Changed axis to 0
    targ = tf.reshape(tf.concat(targ_tensors, 1), targ_batch_shape)
    mask = tf.transpose(
        tf.reshape(tf.concat(mask_tensors, 0), mask_batch_trans_shape))
    return DatasetTensors(obs, targ, mask)

  def cost(self, logits, targ, mask):
    return masked_sigmoid_cross_entropy(
        logits,
        targ,
        mask,
        time_average=self.time_average_cost,
        log_prob_in_bits=self.log_prob_in_bits)

  def to_human_readable(self, data, model_output=None, whole_batch=False):
    obs = data.observations
    unnormalised_num_reps_flag = self._unnormalise(obs[:,:,-1:]).round()
    obs = np.concatenate([obs[:,:,:-1], unnormalised_num_reps_flag], axis=2)
    data = data._replace(observations=obs)
    return bitstring_readable(data, self.batch_size, model_output, whole_batch)
