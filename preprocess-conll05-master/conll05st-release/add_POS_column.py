files=['wsj','brown']

def make_str(words):
	t = ''
	for tt in words[:5]:
		t+='\t'+tt
	for tt in words[4:]:
		t+='\t'+tt
	# print(len(words),len(t.split('\t')))
	return t.strip()

for file in files:
	with open('test.'+file+'.gz.parse.sdeps.combined', 'r') as ff,\
	open('test.'+file+'.gz.parse.sdeps.combined.temp', 'w') as tf:
		for line in ff:
			words = line.split('\t')
			tf.write(make_str(words)+'\n')
	break;