#   Converts the text present in label file to speech
#   Check label file before running this code
from gtts import gTTS

LABEL_PATH = "Model/labels.txt"
#   insert label path here

folder = open(LABEL_PATH, 'r')
label = folder.read().splitlines()
print(label)

FOLDER = "Speech"
#   audio samples stored here

for i in range(len(label)):
    text = label[i]
    print(f"\n Converting {label[i]}...")
    tts = gTTS(text, slow=False, lang='en')
    #   converts the given text to speech

    if not tts.save(f'{FOLDER}/{text}.mp3'):
        print(f"\nSaving {text} to {FOLDER}/{text}.mp3")
        print("\n Success!")
    else:
        #   if fails to convert
        print(f"!!{text} ERROR!!")