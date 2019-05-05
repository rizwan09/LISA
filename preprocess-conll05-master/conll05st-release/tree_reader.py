from nltk import Tree
import pdb
trees = []

#Brown './test.brown.gz.parse' has an issue of ending with (. ?), fixed by considering it inside the tree

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

files = ['./dev-set.gz.parse', './train-set.gz.parse', './test.wsj.gz.parse', './test.brown.gz.parse']
for file in files[0:]:
	with open(file, 'r') as f, open(file+'.no.leaf.serialized.txt', 'w') as wf:
		print('Reading file: ', file, "...")
		trees = []
		s = ''
		i=0
		for line in f:
			if(len(line.strip())==0): 
				# if 'brown' in file: print('line#: ', i, ' readed: ', line, ' s: ', s)
				trees.append(Tree.fromstring(s))
				s=''
			else: 
				i+=1
				s += line.strip() + ' '
		for tree in trees:
			# wf.write(traverse_tree(tree)+'\n')
			wf.write(traverse_tree_no_leaf(tree)+'\n')
	
