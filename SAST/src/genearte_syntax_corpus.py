import os
from nltk import Tree
import pdb
import benepar


datasets= [ 'training']
source_data_base_path = '../../../../local/harold/1-billion-word-language-modeling-benchmark-r13output/'
folder_suffix = '-monolingual.tokenized.shuffled'

output_base_path = 'SyntaxCorpus/'

def traverse_tree_no_leaf(tree):
    st = ''
    if len(tree)==1:
    	st = tree.label()#str(tree).replace('(', '').replace(')','')
    elif len(tree)>1: 
    	st += traverse_tree_no_leaf(tree[0])
    	st += ' ' +tree.label()+ ' '
    	for subtree in tree[1:]:
    		st += traverse_tree_no_leaf(subtree) + ' '

    return st.strip()

parser = benepar.Parser("benepar_en2")

i = 0
for dataset in datasets:
	mydir = source_data_base_path+dataset+folder_suffix
	my_output_dir = output_base_path+dataset 
	for file in os.listdir(mydir):
	    if file.startswith("news"):
	    	i+=1
	    	if i not in [70,86]: continue
	    	with open(os.path.join(mydir, file), 'r') as rf, open(os.path.join(my_output_dir, file), 'w') as wf:
	        	print(i, dataset, file)
	        	for line in rf:
	        		# wf.write(line.strip()+"\n")
	        		if len(line.split())<= 100:
	        			p = parser.parse(line)
	        			# print(line)
	        			wf.write(traverse_tree_no_leaf(p)+'\n')
	        		# if i==44:
	        		# 	if len(line.split())<= 65:
		        	# 		p = parser.parse(line)
		        	# 		# print(line)
		        	# 		wf.write(traverse_tree_no_leaf(p)+'\n')

	        		
	# break
