from nltk import Tree
import pdb
import benepar
''' For the Allennlp constituency parser 
# from allennlp.predictors.predictor import Predictor
# p = predictor.predict(
#   sentence="If I bring 10 dollars tomorrow, can you buy me lunch?"
# )
# print(p['trees'])'''


def traverse_tree(tree):
    st = ''
    if len(tree)==1:
    	st = ' '.join(tree.leaves())+' '+(tree.label())#str(tree).replace('(', '').replace(')','')
    elif len(tree)>1: 
    	st += traverse_tree(tree[0])
    	st += ' ' +tree.label()+ ' '
    	for subtree in tree[1:]:
    		st += traverse_tree(subtree) + ' '

    return st.strip()

parser = benepar.Parser("benepar_en2")

files = ['dev-set',  'train-set', 'test.wsj', 'test.brown']
for file in files:
	print(' FILE: ', file)
	with open(file+'.gz.parse.sdeps', 'r') as f, \
		open(file+'.pred.serialized.txt', 'w') as wf: 
		new_line = ''
		lc  = 0
		for line in f:
			line = line.strip().split()
			if len(line)>0:
				new_line += line[1]+ ' '
			else:
				p = parser.parse(new_line)
				wf.write(traverse_tree(p)+'\n')
				new_line = ''
				lc+=1
				print(' line: ', lc)
				
				
				
