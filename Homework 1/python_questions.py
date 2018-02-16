import re

def check_for_foo_or_bar(text):
	if not re.search(r'\bfoo\b',text):
		return False
	elif not re.search(r'\bbar\b',text):
		return False
	return True
	pass

def replace_rgb(text):
	#replace hex

	text=re.sub('((^)#[A-Fa-f0-9](\s))|((^)#[A-Fa-f0-9]{3}(\s))|((^)#[A-Fa-f0-9]{6}(\s))','COLOR ',text)
	text=re.sub('((^)#[A-Fa-f0-9]($))|((^)#[A-Fa-f0-9]{3}($))|((^)#[A-Fa-f0-9]{6}($))','COLOR',text)
	text=re.sub('((\s)#[A-Fa-f0-9](\s))|((\s)#[A-Fa-f0-9]{3}(\s))|((\s)#[A-Fa-f0-9]{6}(\s))',' COLOR ',text)
	text=re.sub('((\s)#[A-Fa-f0-9]($))|((\s)#[A-Fa-f0-9]{3}($))|((\s)#[A-Fa-f0-9]{6}($))',' COLOR',text)
	
	text=re.sub('(^)rgb\(\s*((((\d*\.)?\d+\s*,\s*)|((\d+\.)\s*,\s*))){2}((\d*\.)?\d+\s*\)|\d+\.\s*\))(\s)','COLOR ',text)
	text=re.sub('(^)rgb\(\s*((((\d*\.)?\d+\s*,\s*)|((\d+\.)\s*,\s*))){2}((\d*\.)?\d+\s*\)|\d+\.\s*\))($)','COLOR',text)
	text=re.sub('(\s)rgb\(\s*((((\d*\.)?\d+\s*,\s*)|((\d+\.)\s*,\s*))){2}((\d*\.)?\d+\s*\)|\d+\.\s*\))(\s)',' COLOR ',text)
	text=re.sub('(\s)rgb\(\s*((((\d*\.)?\d+\s*,\s*)|((\d+\.)\s*,\s*))){2}((\d*\.)?\d+\s*\)|\d+\.\s*\))($)',' COLOR',text)

	return text
	pass

def edit_distance(str1,str2):
	source=str1
	target=str2
	src_length=len(source)+1
	tar_length=len(target)+1
	dp=[0]*src_length

	for i in range(src_length):
		dp[i]=[0]*tar_length

	for i in range(1, src_length):
		dp[i][0]=dp[i-1][0]+1

	for j in range(1, tar_length):
		dp[0][j]=dp[0][j-1]+1

	for i in range(1,src_length):
		for j in range(1,tar_length):
			if source[i-1]==target[j-1]:
				dp[i][j]=min(dp[i-1][j]+1,dp[i][j-1]+1,dp[i-1][j-1])
			else:
				dp[i][j]=min(dp[i-1][j]+1,dp[i][j-1]+1,dp[i-1][j-1]+1)
	return dp[src_length-1][tar_length-1]
	pass

def wine_text_processing(wine_file_path, stopwords_file_path):
	word_count=dict()
	with open(wine_file_path) as w:
		for line in w:
			words=line.split()
			for word in words:
				if word in word_count:
					word_count[word]+=1
				else:
					word_count[word]=1

	#Answer 1:
	print ( '******', '\t', word_count['******'] )
	print ( '*****', '\t', word_count['*****'] )
	print ( '****', '\t', word_count['****'] )
	print ( '***', '\t', word_count['***'] )
	print ( '**', '\t', word_count['**'] )
	print ( '*', '\t', word_count['*'] )

	print ( '\n' )

	#Answer 2:
	iterations=1
	#for key, value in sorted(word_count.items(), key=lambda k,v: (v,k), reverse=True):
	for key in sorted(word_count, key=word_count.get, reverse=True):
		value=word_count[key]
		if iterations>10:
			break
		if not re.match(r'\*{1,6}', key):
			iterations+=1
			print ( key, '\t', value )

	print ( '\n' )

	#Answer 3:
	print ( word_count['a'] )

	print ( '\n' )

	#Answer 4:
	print ( word_count['fruit'] )

	print ( '\n' )
	
	#Answer 5:
	print ( word_count['mineral'] )

	print ( '\n' )

	#Answer 6:
	#Converting to lowercase
	word_count_preprocessed=dict()
	for key in word_count.keys():
		if not key.lower() in word_count_preprocessed:
			word_count_preprocessed[key.lower()]=word_count[key]
		else:
			word_count_preprocessed[key.lower()]+=word_count[key]

	#Remove stopwords
	with open(stopwords_file_path) as s:
		for line in s:
			word=re.sub(r'\n$','',line)
			word_count_preprocessed.pop(word,None)

	iterations=1
	#for key, value in sorted(word_count_preprocessed.items(), key=lambda k,v: (v,k), reverse=True):
	for key in sorted(word_count_preprocessed, key=word_count_preprocessed.get, reverse=True):
		value=word_count_preprocessed[key]
		if iterations>10:
			break
		if not re.match(r'\*{1,6}', key):
			iterations+=1
			print ( key, '\t', value )

	print ( '\n' )

	#Answer 7
	word_count=dict()
	with open(wine_file_path) as w:
		for line in w:
			if re.search(r'\t\*{5}$',line):
				words=line.split()
				for word in words:
					if word in word_count:
						word_count[word]+=1
					else:
						word_count[word]=1

	word_count_preprocessed=dict()
	for key in word_count.keys():
		if not key.lower() in word_count_preprocessed:
			word_count_preprocessed[key.lower()]=word_count[key]
		else:
			word_count_preprocessed[key.lower()]+=word_count[key]

	#Remove stopwords
	with open(stopwords_file_path) as s:
		for line in s:
			word=re.sub(r'\n$','',line)
			word_count_preprocessed.pop(word,None)

	iterations=1
	#for key, value in sorted(word_count_preprocessed.items(), key=lambda k,v: (v,k), reverse=True):
	for key in sorted(word_count_preprocessed,key=word_count_preprocessed.get,reverse=True):
		value=word_count_preprocessed[key]
		if iterations>10:
			break
		if not re.match(r'\*{5}', key):
			iterations+=1
			print ( key, '\t', value )

	print ( '\n' )
 
	#Answer 8
	word_count=dict()
	with open(wine_file_path) as w:
		for line in w:
			if re.search(r'\t\*$',line):
				words=line.split()
				for word in words:
					if word in word_count:
						word_count[word]+=1
					else:
						word_count[word]=1

	word_count_preprocessed=dict()
	for key in word_count.keys():
		if not key.lower() in word_count_preprocessed:
			word_count_preprocessed[key.lower()]=word_count[key]
		else:
			word_count_preprocessed[key.lower()]+=word_count[key]

	#Remove stopwords
	with open(stopwords_file_path) as s:
		for line in s:
			word=re.sub(r'\n$','',line)
			word_count_preprocessed.pop(word,None)

	iterations=1
	#for key, value in sorted(word_count_preprocessed.items(), key=lambda k,v: (v,k), reverse=True):
	for key in sorted(word_count_preprocessed,key=word_count_preprocessed.get,reverse=True):
		value=word_count_preprocessed[key]
		if iterations>10:
			break
		if not re.match(r'\*', key):
			iterations+=1
			print ( key, '\t', value )

	print ( '\n' )
 

	#Starting answer 9, 10
	word_count_red=dict()
	word_count_white=dict()
	with open(wine_file_path) as w:
		for line in w:
			if re.search(r'\b[rR][eE][dD]\b',line):
				words=line.split()
				for word in words:
					if word in word_count_red:
						word_count_red[word]+=1
					else:
						word_count_red[word]=1
			if re.search(r'\b[wW][hH][iI][tT][eE]\b',line):
				words=line.split()
				for word in words:
					if word in word_count_white:
						word_count_white[word]+=1
					else:
						word_count_white[word]=1


	word_count_preprocessed_red=dict()
	for key in word_count_red.keys():
		if not key.lower() in word_count_preprocessed_red:
			word_count_preprocessed_red[key.lower()]=word_count_red[key]
		else:
			word_count_preprocessed_red[key.lower()]+=word_count_red[key]

	#Remove stopwords
	with open(stopwords_file_path) as s:
		for line in s:
			word=re.sub(r'\n$','',line)
			word_count_preprocessed_red.pop(word,None)

	word_count_preprocessed_white=dict()
	for key in word_count_white.keys():
		if not key.lower() in word_count_preprocessed_white:
			word_count_preprocessed_white[key.lower()]=word_count_white[key]
		else:
			word_count_preprocessed_white[key.lower()]+=word_count_white[key]

	#Remove stopwords
	with open(stopwords_file_path) as s:
		for line in s:
			word=re.sub(r'\n$','',line)
			word_count_preprocessed_white.pop(word,None)

	iterations=1
	popular_white_words=[]
	#for key, value in sorted(word_count_preprocessed_white.items(), key=lambda k,v: (v,k), reverse=True):
	for key in sorted(word_count_preprocessed_white,key=word_count_preprocessed_white.get,reverse=True):
		value=word_count_preprocessed_white[key]
		if not re.match(r'\*{1,6}', key):
			iterations+=1
			popular_white_words.append(key)


	iterations=1
	popular_red_words=[]
	#for key, value in sorted(word_count_preprocessed_red.items(), key=lambda k,v: (v,k), reverse=True):
	for key in sorted(word_count_preprocessed_red,key=word_count_preprocessed_red.get,reverse=True):
		value=word_count_preprocessed_red[key]
		if not re.match(r'\*{1,6}', key):
			iterations+=1
			popular_red_words.append(key)

	#Answer 9
	iterations=1
	for elem in popular_red_words:
		if iterations>10:
			break
		if not elem in popular_white_words:
			iterations+=1
			print ( elem, '\t', word_count_preprocessed_red[elem] )

	print ( '\n' )

	#Answer 10
	iterations=1
	for elem in popular_white_words:
		if iterations>10:
			break
		if not elem in popular_red_words:
			iterations+=1
			print ( elem, '\t', word_count_preprocessed_white[elem] )

	print ( '\n' )

	pass	
