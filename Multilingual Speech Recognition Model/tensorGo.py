import whisper
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
import json

model = whisper.load_model('base')

#This function convert audio(.wav or mp3) file to text and return text with language
def audio_to_text(path):
    result = model.transcribe(path)
    return result["text"],result["language"]


#Translate the text
model_name = "facebook/m2m100_418M"
translate_model = M2M100ForConditionalGeneration.from_pretrained(model_name)
tokenizer = M2M100Tokenizer.from_pretrained(model_name)

#This function translate the text of any language to english
def translate_text(text, src_lang, tgt_lang):
  inputs = tokenizer(text, return_tensors="pt", src_lang=src_lang)
  outputs = translate_model.generate(**inputs, 
                                     forced_bos_token_id=tokenizer.lang_code_to_id[tgt_lang])
  translation = tokenizer.decode(outputs[0], skip_special_tokens=True)
  return translation

# Example usage
audio_path = "Sample audios/Red Fort.mp3"
transcribe_text,language = audio_to_text(audio_path)
translation = translate_text(transcribe_text, src_lang=language, tgt_lang="en")
print("Text after translation:",translation)


#Creating dummy RAG using json file
with open('dummy.json', 'r') as file:
    data = json.load(file)
    
def query_rag_document(query, data):
    for entry in data['document']:
        if query.lower() in entry['title'].lower() or query.lower() in entry['content'].lower():
            return entry['content']
    return "No relevant information found."


x=query_rag_document(translation,data)
print(x)


