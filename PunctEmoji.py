import re
import emoji


def removeText (text):

    ohne = re.sub(r"[^\w\s?!]", "", text)
    return ohne

def removeEmoji (text):
   return emoji.replace_emoji(text, replace= 'HierEmojiEntfernt' )


#testing here 
tweet="Hallo!!! ++**~ Geb@ 😊 das aus?" 
test= removeEmoji(removeText(tweet))
print (test)

    