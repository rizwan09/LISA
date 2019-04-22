import tensorflow as tf
import json
import re
import sys
# import dataset_ori as dataset
import dataset
import constants
from pathlib import Path


def load_hparams(args, model_config):
  # Create a HParams object specifying the names and values of the model hyperparameters
  hparams = tf.contrib.training.HParams(**constants.hparams)

  # First get default hyperparams from the model config
  if 'hparams' in model_config:
    hparams.override_from_dict(model_config['hparams'])

  if args.debug:
    hparams.set_hparam('shuffle_buffer_multiplier', 10)
    hparams.set_hparam('eval_throttle_secs', 60)
    hparams.set_hparam('eval_every_steps', 100)

  # Override those with command line hyperparams
  if args.hparams:
    hparams.parse(args.hparams)

  tf.logging.log(tf.logging.INFO, "Using hyperparameters: %s" % str(hparams.values()))

  return hparams


def get_input_fn(vocab, data_config, data_files, parse_tree_files, batch_size, num_epochs, shuffle,
                 shuffle_buffer_multiplier=1, embedding_files=None, input_context=None):
  # this needs to be created from here (lazily) so that it ends up in the same tf.Graph as everything else
  vocab_lookup_ops = vocab.create_vocab_lookup_ops(embedding_files)
  # print("In train_utils.get_input_fn")
  # #for my idea
  tuple_data_itr = dataset.get_data_iterator(data_files, parse_tree_files, data_config, vocab_lookup_ops, batch_size, num_epochs, shuffle,
                                   shuffle_buffer_multiplier)
  return {'features': tuple_data_itr[0], 'parse_tree':tuple_data_itr[1]}
  # #for baseline
  # return dataset.get_data_iterator(data_files, data_config, vocab_lookup_ops, batch_size, num_epochs, shuffle,
  #                                  shuffle_buffer_multiplier)


def load_json_configs(config_file_list, args=None):
  """
  Loads a list of json configuration files into one combined map. Configuration files
  at the end of the list take precedece over earlier configuration files (so they will
  overwrite earlier configs!)

  If args is passed, then this function will attempt to replace

  :param config_file_list: list of json configuration files to load
  :param args: command line args to replace special strings in json
  :return: map containing combined configurations
  """
  combined_config = {}
  config_files = config_file_list.split(',')
  for config_file in config_files:
    if args:
      # read the json in as a string so that we can run a replace on it
      json_str = Path(config_file).read_text()
      matches = re.findall(r'.*##(.*)##.*', json_str)
      for match in matches:
        try:
          value = getattr(args, match)
          json_str = json_str.replace('##%s##' % match, value)
        except AttributeError:
          tf.logging.log(tf.logging.ERROR, 'Could not find "%s" attribute in command line args when parsing: %s' %
                         (match, config_file))
          sys.exit(1)
      try:
        config = json.loads(json_str)
      except json.decoder.JSONDecodeError as e:
        tf.logging.log(tf.logging.ERROR, 'Error reading json: "%s"' % config_file)
        tf.logging.log(tf.logging.ERROR, e.msg)
        sys.exit(1)
    else:
      with open(config_file) as f:
        try:
          config = json.load(f)
        except json.decoder.JSONDecodeError as e:
          tf.logging.log(tf.logging.ERROR, 'Error reading json: "%s"' % config_file)
          tf.logging.log(tf.logging.ERROR, e.msg)
          sys.exit(1)
    combined_config = {**combined_config, **config}
  return combined_config


def copy_without_dropout(hparams):
  new_hparams = {k: (1.0 if 'dropout' in k else v) for k, v in hparams.values().items()}
  return tf.contrib.training.HParams(**new_hparams)


def get_vars_for_moving_average(average_norms):
  vars_to_average = tf.trainable_variables()
  if not average_norms:
    vars_to_average = [v for v in tf.trainable_variables() if 'norm' not in v.name]
  tf.logging.log(tf.logging.INFO, "Creating moving averages for %d variables." % len(vars_to_average))
  return vars_to_average


def learning_rate(hparams, global_step):
  lr = hparams.learning_rate
  warmup_steps = hparams.warmup_steps
  decay_rate = hparams.decay_rate
  if warmup_steps > 0:

    # add 1 to global_step so that we start at 1 instead of 0
    global_step_float = tf.cast(global_step, tf.float32) + 1.
    lr *= tf.minimum(tf.rsqrt(global_step_float),
                     tf.multiply(global_step_float, warmup_steps ** -decay_rate))
    return lr
  else:
    decay_steps = hparams.decay_steps
    if decay_steps > 0:
      return lr * decay_rate ** (global_step / decay_steps)
    else:
      return lr


def best_model_compare_fn(best_eval_result, current_eval_result, key):
  """Compares two evaluation results and returns true if the second one is greater.
    Both evaluation results should have the value for key, used for comparison.
    Args:
      best_eval_result: best eval metrics.
      current_eval_result: current eval metrics.
      key: key to value used for comparison.
    Returns:
      True if the loss of current_eval_result is smaller; otherwise, False.
    Raises:
      ValueError: If input eval result is None or no loss is available.
    """

  if not best_eval_result or key not in best_eval_result:
    raise ValueError('best_eval_result cannot be empty or key "%s" is not found.' % key)

  if not current_eval_result or key not in current_eval_result:
    raise ValueError('best_eval_result cannot be empty or key "%s" is not found.' % key)

  return best_eval_result[key] < current_eval_result[key]


def serving_input_receiver_fn():
  inputs = tf.placeholder(tf.int32, [None, None, None])
  pinputs = tf.placeholder(tf.int32, [None, None])
  return tf.estimator.export.TensorServingInputReceiver(inputs,inputs)


# Called once when the model is saved. This function produces a Tensorflow
# graph of operations that will be prepended to your model graph. When
# your model is deployed as a REST API, the API receives data in JSON format,
# parses it into Tensors, then sends the tensors to the input graph generated by
# this function. The graph can transform the data so it can be sent into your
# model input_fn. You can do anything you want here as long as you do it with
# tf.* functions that produce a graph of operations.
def serving_input_receiver_fn2():
    # placeholder for the data received by the API (already parsed, no JSON decoding necessary,
    # but the JSON must contain one or multiple 'image' key(s) with 28x28 greyscale images  as content.)
    inputs = {"features": tf.placeholder(tf.int32, [None, None, None]), "labels": tf.placeholder(tf.int32, [None, None])}  # the shape of this dict should match the shape of your JSON
    # features = {inputs['features'],  inputs['labels']} # no transformation needed
    return tf.estimator.export.TensorServingInputReceiver(inputs, inputs)  # features are the features needed by your model_fn
    # Return a ServingInputReceiver if your features are a dictionary of Tensors, TensorServingInputReceiver if they are a straight Tensor

