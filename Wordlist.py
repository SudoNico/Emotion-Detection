import csv

def start():
    wordlist=[]
    row=0
    with open('German-NRC-EmoLex.csv', 'r') as data:
        for line in data:
            word = line.split()
            wordlist.append(word[len(word)-1])      
    wordlist.sort()
    text = [[0 for i in range(12)] for j in range(len(wordlist))]
    with open('German-NRC-EmoLex.csv', 'r') as data:
        for line in data:
            word = line.split() 
            for i in range(12):
                text[row][i]=word[i]
            row+=1
    Labels = [[0 for i in range(11)] for j in range(len(wordlist))]
    for i in range(len(wordlist)):
        Labels[i][0]=wordlist[i]
        for j in range(len(text)):
            if wordlist[i]==text[j][11]:
                for k in range(1,11):
                    Labels[i][k]=text[j][k]
                text.pop(j)                    
                break
    with open('German-NRC-EmoLex-sorted.csv','w') as output:
        csvwriter = csv.writer(output)
        csvwriter.writerows(Labels)    
       
#Word	anger	anticipation	disgust	fear	joy	negative	positive	sadness	surprise	trust	



start()
