from transformers import AutoTokenizer, TFAutoModelForQuestionAnswering
import os

model_dir = os.path.join(os.getcwd(), 'model')
tokenizer_dir = os.path.join(os.getcwd(), 'tokenizer')
checkpoint = 'wdevinsp/indobert-base-uncased-finetuned-digestive-qna'

model = TFAutoModelForQuestionAnswering.from_pretrained(checkpoint)
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

model.save_pretrained(model_dir)
tokenizer.save_pretrained(tokenizer_dir)