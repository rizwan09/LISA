import pdb, math
import numpy as np
np.random.seed(695338114)

file_names = ['train-set.gz.parse.sdeps.combined', 'dev-set.gz.parse.sdeps.combined', 'test.wsj.gz.parse.sdeps.combined', 'test.brown.gz.parse.sdeps.combined']

# pfile_names = ['train-set.gz.parse.pred.serialized', 'dev-set.gz.parse.pred.serialized', 'test.wsj.gz.parse.pred.serialized','test.brown.gz.parse.pred.serialized']
pfile_names = ['train-set.pred.no.leaf.serialized', 'dev-set.pred.no.leaf.serialized', 'test.wsj.pred.no.leaf.serialized','test.brown.pred.no.leaf.serialized']
# pfile_names = ['train-set.gz.parse.serialized', 'dev-set.gz.parse.serialized', 'test.wsj.gz.parse.serialized','test.brown.gz.parse.serialized']
# pfile_names = ['train-set.gz.parse.no.leaf.serialized', 'dev-set.gz.parse.no.leaf.serialized', 'test.wsj.gz.parse.no.leaf.serialized','test.brown.gz.parse.no.leaf.serialized']



trim_length = 40
fraction=0.7

file_tag_s = '.short.'
file_tag = '.no.leaf.short.'
# file_tag = '.with.leaf.short.'
rnd_tag = '.rnd_'

def write_conll_file(file, sent_ids, ext='txt', f=1):

	if f==1: 
		wfile = file+file_tag_s+ext
	else:
		wfile = file+rnd_tag+str(f)+file_tag_s+ext
	# pdb.set_trace()
	with open(file+'.'+ext,'r') as f, open(wfile, 'w') as wf:
		instance = ''
		i=0
		for line in f:
			if len(line.strip().split())>0:
				instance+=line
			else:
				if i in sent_ids or f==1:
					# if 'train' in file:
						# pdb.set_trace()
					wf.write(instance+"\n")
				instance=''
				i+=1
	print(' Done writting: ', len(sent_ids),'/',i, ' (p = '+str(len(sent_ids)/i) +') sentences from ', file+'.'+ext, ' to ', wfile)


def write_txt_file(file, sent_ids, ext='txt', f=1):
	if f==1: 
		wfile = file+file_tag+ext
	else:
		wfile = file+rnd_tag+str(f)+file_tag+ext
	with open(file+'.'+ext,'r') as f, open(wfile, 'w') as wf:
		sentences = f.readlines()
		for i, sen in enumerate(sentences):
			if i in sent_ids or f==1:
				wf.write(sen.strip()+"\n")
	print(' Done writting: ', str(len(sent_ids)) + ' sentences from ', file+'.'+ext, ' to ', wfile)

 


for file, pfile in zip(file_names, pfile_names): 
	with open(file+'.bio', 'r') as f, open(pfile+".txt", 'r') as pf, open(file+file_tag_s+'bio', 'w') as wf, open(pfile+file_tag+'txt', 'w') as wpf:
		i = 0
		parse_sentences = pf.readlines()
		instance = ''
		sen_len = 0
		new_sent_ids = []
		for line in f:
			if len(line.strip().split())>0:
				instance+=line
				sen_len+=1
			else:
				if sen_len>0 and sen_len<=trim_length:
					parse_instance = parse_sentences[i]
					wf.write(instance+"\n")
					wpf.write(parse_instance)
					new_sent_ids.append(i)
					# if i==18 and 'dev-set' in file: pdb.set_trace()

				instance=''
				sen_len=0
				i+=1

		# print('FIle: ', file, ' sentence# ', i, ' trimmed# ', len(new_sent_ids), ' percentage: ', len(new_sent_ids)/i)
		print(' Done writting: ', len(new_sent_ids),'/',i, ' (p = '+str(len(new_sent_ids)/i) +') sentences from ', f.name, ' to ', wf.name)
		print(' Done writting: ', len(new_sent_ids),'/',i, ' (p = '+str(len(new_sent_ids)/i) +') sentences from ', pf.name, ' to ', wpf.name)

		

		l = len(new_sent_ids)
		sent_ids = np.random.choice(np.array(new_sent_ids), int(math.ceil(fraction*l)), replace=False)
		# if 'train' in file: pdb.set_trace()
		write_conll_file(file, sent_ids, ext='bio', f=fraction)
		write_txt_file(pfile, sent_ids, ext='txt', f=fraction)

		if 'dev-set' in file:

			# pdb.set_trace()
			# print('to check: 100 ids:', sent_ids[:100])

			write_conll_file('conll2005-dev-gold-parse', new_sent_ids, ext='txt')
			write_conll_file('conll2005-dev-gold-props', new_sent_ids, ext='txt')

			write_conll_file('conll2005-dev-gold-parse', sent_ids, ext='txt', f=fraction)
			write_conll_file('conll2005-dev-gold-props', sent_ids, ext='txt', f=fraction)







