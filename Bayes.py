import csv

# Ich werde noch schauen ob ich eine gute Bibliotek mit fertigen Funktionen finde ansonsten hier erst mal ein commplett selbst aufgesetzter Bayes
# Quelle Big Data Vorlesung zu NaiveBayes ;)

# words from the input text
text=[]
text.append("Sunny")
text.append("Overcast")

# for which Label? 
Label=1

# Variables for Summ off all with and without Label
totalyes=0
totalno=0

# Calculate total Sum
with open('FrequenzyBSP.csv', 'r') as data:
    reader = csv.reader(data, delimiter=',')
    for line in reader:
        totalyes += int(line[Label])
        for i in range(1,len(line)):
            if i != Label:
                totalno += int(line[i])
absolute = totalyes + totalno

# Probability for existance of Label and not Label -- p(Label) and p(notLabel)
PLabel = totalyes/absolute
PnotLabel = totalno/absolute

# Variables for probability -- p(words|Label) and p(words|notLabel)
WordLabel = 1
NotWordLabel = 1

# Calculate p(words|Label) and p(words|notLabel)
with open('FrequenzyBSP.csv', 'r') as data:
    reader = csv.reader(data, delimiter=',')
    for line in reader:
        for i in range(len(text)):
            if text[i] == line[0]:
                isWord = int(line[Label])/totalyes
                WordLabel = WordLabel * isWord
                no=0
                for i in range(1,len(line)):
                    if i != Label:
                        no += int(line[i])
                notword = no/totalno
                NotWordLabel = NotWordLabel * notword

# calculate final Score
Bayes = (WordLabel*PLabel)/(NotWordLabel*PnotLabel)

print(Bayes)

# there must be one value for all labels at least else error
# for testing i used the FrequenzyBSP.csv

