import pdb
file_names = ['train-set.gz.parse.sdeps.combined', 'dev-set.gz.parse.sdeps.combined', 'test.wsj.gz.parse.sdeps.combined', 'test.brown.gz.parse.sdeps.combined']

pfile_names = ['train-set.gz.parse.pred.serialized', 'dev-set.gz.parse.pred.serialized', 'test.wsj.gz.parse.pred.serialized','test.brown.gz.parse.pred.serialized']

props_parse_file_only_dev = ['dev-set.gz.parse.serialized']

trim_length = 42


def write_conll_file(file, sent_ids, ext='txt'):
	with open(file+'.'+ext,'r') as f, open(file+'.short.'+ext, 'w') as wf:
		instance = ''
		i=0
		for line in f:
			if len(line.strip().split())>0:
				instance+=line
			else:
				if i in sent_ids:
					wf.write(instance)
				instance=''
				i+=1
	print(' Done writting: ', len(sent_ids),'/',i, ' (p = '+str(len(sent_ids)/i) +') sentences from ', file+'.'+ext, ' to ', file+'.short.'+ext)



for file, pfile in zip(file_names, pfile_names): 
	with open(file+'.bio', 'r') as f, open(pfile+".txt", 'r') as pf, open(file+'.short.bio', 'w') as wf, open(pfile+'.short.txt', 'w') as wpf:
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
					wf.write(instance)
					wpf.write(parse_instance)
					new_sent_ids.append(i)
				instance=''
				sen_len=0
				i+=1

		# print('FIle: ', file, ' sentence# ', i, ' trimmed# ', len(new_sent_ids), ' percentage: ', len(new_sent_ids)/i)
		print(' Done writting: ', len(new_sent_ids),'/',i, ' (p = '+str(len(new_sent_ids)/i) +') sentences from ', f.name, ' to ', wf.name)
		print(' Done writting: ', len(new_sent_ids),'/',i, ' (p = '+str(len(new_sent_ids)/i) +') sentences from ', pf.name, ' to ', wpf.name)

		if 'dev-set' in file:
			write_conll_file('conll2005-dev-gold-parse', new_sent_ids, ext='txt')
			write_conll_file('conll2005-dev-gold-props', new_sent_ids, ext='txt')






