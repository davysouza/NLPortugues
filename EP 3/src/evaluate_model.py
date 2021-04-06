import nltk

hypothesis = ['This', 'is', 'cat'] 
reference  = ['This', 'is', 'a', 'cat']
BLEUscore  = nltk.translate.bleu_score.corpus_bleu([reference], [hypothesis], weights = [1])
print(BLEUscore)
