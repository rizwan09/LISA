import tensorflow as tf
import constants
from data_generator import conll_data_generator, serialized_tree_generator
from tensorflow.contrib.data.python.ops import grouping
from tensorflow.python.ops import array_ops
import pdb

def map_strings_to_ints(vocab_lookup_ops, data_config, feature_label_names):
  # def _mapper(d,t):
  def _mapper(d):
    intmapped = []
    inttree=None
    for i, datum_name in enumerate(feature_label_names):
      if 'vocab' in data_config[datum_name]:
        if 'type' in data_config[datum_name] and data_config[datum_name]['type'] == 'range':
          idx = data_config[datum_name]['conll_idx']
          if idx[1] == -1:
            intmapped.append(vocab_lookup_ops[data_config[datum_name]['vocab']].lookup(d[:, i:]))
          else:
            last_idx = i + idx[1]
            intmapped.append(vocab_lookup_ops[data_config[datum_name]['vocab']].lookup(d[:, i:last_idx]))
        elif 'parse_tree_type' in datum_name:
          inttree=tf.reshape(vocab_lookup_ops[data_config[datum_name]['vocab']].lookup(d), [-1])
        else:
          intmapped.append(tf.expand_dims(vocab_lookup_ops[data_config[datum_name]['vocab']].lookup(d[:, i]), -1))
      else:
        intmapped.append(tf.expand_dims(tf.string_to_number(d[:, i], out_type=tf.int64), -1))

    if len(intmapped)>0: return tf.cast(tf.concat(intmapped, axis=-1), tf.int32)
    if inttree is not None: 
      return tf.cast(inttree, tf.int32)

  return _mapper


def get_data_iterator(data_filenames, parse_tree_filenames, data_config, vocab_lookup_ops, batch_size, num_epochs, shuffle,
                      shuffle_buffer_multiplier):

  bucket_boundaries = constants.DEFAULT_BUCKET_BOUNDARIES
  bucket_batch_sizes = [batch_size] * (len(bucket_boundaries) + 1)

  # todo do something smarter with multiple files + parallel?


  with tf.device('/cpu:0'): 
  # with constants.MIRRORED_STRATEGY.scope():
    
  
    # get the names of data fields in data_config that correspond to features or labels,
    # and thus that we want to load into batches

    feature_label_names = [d for d in data_config.keys() if 'parse_tree' not in d and\
                           ('feature' in data_config[d] and data_config[d]['feature']) or
                           ('label' in data_config[d] and data_config[d]['label'])]
    dataset = tf.data.Dataset.from_generator(lambda: conll_data_generator(data_filenames, data_config),
                                             output_shapes=[None, None], output_types=tf.string)
    # intmap the dataset
    dataset = dataset.map(map_strings_to_ints(vocab_lookup_ops, data_config, feature_label_names), num_parallel_calls=8)
    


    parse_feature_names = [d for d in data_config.keys() if 'parse_tree_type' in d]
    parseset = tf.data.Dataset.from_generator(lambda: serialized_tree_generator(parse_tree_filenames, 
      data_config),output_shapes=[None], output_types=tf.string)                                   
    # intmap the dataset
    parseset = parseset.map(map_strings_to_ints(vocab_lookup_ops, data_config, parse_feature_names), num_parallel_calls=8)
    
    pdb.set_trace()
    
    zippedDatatset = tf.data.Dataset.zip((dataset, parseset))
    zippedDatatset = zippedDatatset.cache()
    
    itd = zippedDatatset.make_initializable_iterator()
    eld = itd.get_next()
    with tf.Session() as sess:
      pdb.set_trace()
      sess.run(tf.global_variables_initializer())
      sess.run([itd.initializer, tf.tables_initializer()]) #[itd.initializer, itp.initializer, tf.tables_initializer()]
      aa = sess.run(eld)
      pdb.set_trace()  

    # zippedDatatset = zippedDatatset.filter(lambda d, t: tf.math.less_equal(tf.shape(d)[0], 40)) #empirical for now
    zippedDatatset = zippedDatatset.apply(tf.contrib.data.bucket_by_sequence_length(element_length_func=lambda d, t: tf.shape(d)[0]+tf.shape(t)[0], \
                                                              bucket_boundaries=bucket_boundaries,
                                                              bucket_batch_sizes=bucket_batch_sizes,
                                                              padded_shapes=zippedDatatset.output_shapes,
                                                              padding_values=(constants.PAD_VALUE, constants.PAD_VALUE)))
    if shuffle:
      zippedDatatset = zippedDatatset.apply(tf.contrib.data.shuffle_and_repeat(buffer_size=batch_size*shuffle_buffer_multiplier,
                                                                 count=num_epochs))

    # todo should the buffer be bigger?
    zippedDatatset.prefetch(buffer_size=1)

    # create the iterator
    # it has to be initializable due to the lookup tables
    iterator = zippedDatatset.make_initializable_iterator()
    tf.add_to_collection(tf.GraphKeys.TABLE_INITIALIZERS, iterator.initializer)
    
    return iterator.get_next()

    
