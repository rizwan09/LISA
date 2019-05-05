import data_converters
import numpy as np
import pdb


def conll_data_generator(filenames, data_config):
  print('in conll_data_generator', filenames)
  lengths = []
  lower_lengths = []
  for filename in filenames :
    with open(filename, 'r') as f:
      sents = 0
      toks = 0
      buf = []
      for line in f:
        line = line.strip()
        if line:
          toks += 1
          split_line = line.split()
          data_vals = []
          for d in data_config.keys():
            # only return the data that we're actually going to use as inputs or outputs
            if ('feature' in data_config[d] and data_config[d]['feature']) or \
               ('label' in data_config[d] and data_config[d]['label']):
              if 'parse_tree' not in d: 
                datum_idx = data_config[d]['conll_idx']
                converter_name = data_config[d]['converter']['name'] if 'converter' in data_config[d] else 'default_converter'
                converter_params = data_converters.get_params(data_config[d], split_line, datum_idx)
                data = data_converters.dispatch(converter_name)(**converter_params)
                data_vals.extend(data)    
          buf.append(tuple(data_vals))
          '''
          buf = [('0', 'In', 'in', 'IN', '44', 'prep', 'False', 'IN/False', 'O', 'O', 'O', 'B-AM-LOC'), ('0', 'an', 'an', 'DT', '4', 'det', 'False', 'DT/False', 'O', 'O', 'O', 'I-AM-LOC'), ('0', 'Oct.', 'oct.', 'NNP', '4', 'nn', 'False', 'NNP/False', 'O', 'O', 'O', 'I-AM-LOC'), ('0', '19', '19', 'CD', '4', 'num', 'False', 'CD/False', 'O', 'O', 'O', 'I-AM-LOC'), ('0', 'review', 'review', 'NN', '0', 'pobj', 'False', 'NN/False', 'O', 'O', 'O', 'I-AM-LOC'), ('0', 'of', 'of', 'IN', '4', 'prep', 'False', 'IN/False', 'O', 'O', 'O', 'I-AM-LOC'), ('0', '``', '``', '``', '8', 'punct', 'False', '``/False', 'O', 'O', 'O', 'I-AM-LOC'), ('0', 'The', 'the', 'DT', '8', 'det', 'False', 'DT/False', 'O', 'O', 'O', 'I-AM-LOC'), ('0', 'Misanthrope', 'misanthrope', 'NN', '5', 'pobj', 'False', 'NN/False', 'O', 'O', 'O', 'I-AM-LOC'), ('0', "''", "''", "''", '8', 'punct', 'False', "''/False", 'O', 'O', 'O', 'I-AM-LOC'), ('0', 'at', 'at', 'IN', '8', 'prep', 'False', 'IN/False', 'O', 'O', 'O', 'I-AM-LOC'), ('0', 'Chicago', 'chicago', 'NNP', '14', 'poss', 'False', 'NNP/False', 'O', 'O', 'O', 'I-AM-LOC'), ('0', "'s", "'s", 'POS', '11', 'possessive', 'False', 'POS/False', 'O', 'O', 'O', 'I-AM-LOC'), ('0', 'Goodman', 'goodman', 'NNP', '14', 'nn', 'False', 'NNP/False', 'O', 'O', 'O', 'I-AM-LOC'), ('0', 'Theatre', 'theatre', 'NNP', '10', 'pobj', 'False', 'NNP/False', 'O', 'O', 'O', 'I-AM-LOC'), ('0', '(', '(', '-LRB-', '19', 'punct', 'False', '-LRB-/False', 'O', 'O', 'O', 'I-AM-LOC'), ('0', '``', '``', '``', '19', 'punct', 'False', '``/False', 'O', 'O', 'O', 'I-AM-LOC'), ('0', 'Revitalized', 'revitalized', 'VBN', '18', 'amod', 'True', 'VBN/True', 'B-V', 'B-A0', 'O', 'I-AM-LOC'), ('0', 'Classics', 'classics', 'NNS', '19', 'nsubj', 'False', 'NNS/False', 'B-A1', 'I-A0', 'O', 'I-AM-LOC'), ('0', 'Take', 'take', 'VBP', '4', 'dep', 'True', 'VBP/True', 'O', 'B-V', 'O', 'I-AM-LOC'), ('0', 'the', 'the', 'DT', '21', 'det', 'False', 'DT/False', 'O', 'B-A1', 'O', 'I-AM-LOC'), ('0', 'Stage', 'stage', 'NN', '19', 'dobj', 'False', 'NN/False', 'O', 'I-A1', 'O', 'I-AM-LOC'), ('0', 'in', 'in', 'IN', '19', 'prep', 'False', 'IN/False', 'O', 'B-AM-LOC', 'O', 'I-AM-LOC'), ('0', 'Windy', 'windy', 'NNP', '24', 'nn', 'False', 'NNP/False', 'O', 'I-AM-LOC', 'O', 'I-AM-LOC'), ('0', 'City', 'city', 'NNP', '22', 'pobj', 'False', 'NNP/False', 'O', 'I-AM-LOC', 'O', 'I-AM-LOC'), ('0', ',', ',', ',', '19', 'punct', 'False', ',/False', 'O', 'O', 'O', 'I-AM-LOC'), ('0', "''", "''", "''", '19', 'punct', 'False', "''/False", 'O', 'O', 'O', 'I-AM-LOC'), ('0', 'Leisure', 'leisure', 'NN', '19', 'dep', 'False', 'NN/False', 'O', 'O', 'O', 'I-AM-LOC'), ('0', '&', '&', 'CC', '27', 'cc', 'False', 'CC/False', 'O', 'O', 'O', 'I-AM-LOC'), ('0', 'Arts', 'arts', 'NNS', '27', 'conj', 'False', 'NNS/False', 'O', 'O', 'O', 'I-AM-LOC'), ('0', ')', ')', '-RRB-', '19', 'punct', 'False', '-RRB-/False', 'O', 'O', 'O', 'I-AM-LOC'), ('0', ',', ',', ',', '44', 'punct', 'False', ',/False', 'O', 'O', 'O', 'O'), ('0', 'the', 'the', 'DT', '33', 'det', 'False', 'DT/False', 'O', 'O', 'B-A1', 'B-A1'), ('0', 'role', 'role', 'NN', '44', 'nsubjpass', 'False', 'NN/False', 'O', 'O', 'I-A1', 'I-A1'), ('0', 'of', 'of', 'IN', '33', 'prep', 'False', 'IN/False', 'O', 'O', 'I-A1', 'I-A1'), ('0', 'Celimene', 'celimene', 'NNP', '34', 'pobj', 'False', 'NNP/False', 'O', 'O', 'I-A1', 'I-A1'), ('0', ',', ',', ',', '33', 'punct', 'False', ',/False', 'O', 'O', 'O', 'I-A1'), ('0', 'played', 'played', 'VBN', '33', 'vmod', 'True', 'VBN/True', 'O', 'O', 'B-V', 'I-A1'), ('0', 'by', 'by', 'IN', '37', 'prep', 'False', 'IN/False', 'O', 'O', 'B-A0', 'I-A1'), ('0', 'Kim', 'kim', 'NNP', '40', 'nn', 'False', 'NNP/False', 'O', 'O', 'I-A0', 'I-A1'), ('0', 'Cattrall', 'cattrall', 'NNP', '38', 'pobj', 'False', 'NNP/False', 'O', 'O', 'I-A0', 'I-A1'), ('0', ',', ',', ',', '33', 'punct', 'False', ',/False', 'O', 'O', 'O', 'I-A1'), ('0', 'was', 'was', 'VBD', '44', 'auxpass', 'False', 'VBD/False', 'O', 'O', 'O', 'O'), ('0', 'mistakenly', 'mistakenly', 'RB', '44', 'advmod', 'False', 'RB/False', 'O', 'O', 'O', 'B-AM-MNR'), ('0', 'attributed', 'attributed', 'VBN', '44', 'root', 'True', 'VBN/True', 'O', 'O', 'O', 'B-V'), ('0', 'to', 'to', 'TO', '44', 'prep', 'False', 'TO/False', 'O', 'O', 'O', 'B-A2'), ('0', 'Christina', 'christina', 'NNP', '47', 'nn', 'False', 'NNP/False', 'O', 'O', 'O', 'I-A2'), ('0', 'Haag', 'haag', 'NNP', '45', 'pobj', 'False', 'NNP/False', 'O', 'O', 'O', 'I-A2'), ('0', '.', '.', '.', '44', 'punct', 'False', './False', 'O', 'O', 'O', 'O')]
          '''
        else:
          if buf:
            yield buf
            lengths.append(len(buf))
            if len(buf)<=42: lower_lengths.append(len(buf))
            sents += 1
            buf = []
      if buf:
        yield buf
    
  print('max sentence lengths: ', max(lengths),len(lengths))#, " less than = 42:", len(lower_lengths), len(lower_lengths)/len(lengths))


def serialized_tree_generator(parse_tree_filenames, data_config):
  s_lengths = []
  print('serialized_tree_gen', parse_tree_filenames)
  for filename in parse_tree_filenames:
    with open(filename, 'r') as f:
      sents = 0
      toks = 0
      for line in f:
        line = line.strip()
        if line:
          toks += 1
          split_line = line.split()
          yield split_line
          sents += 1
          s_lengths.append(len(split_line))
  print('max tree lengths: ', max(s_lengths),len(s_lengths))

