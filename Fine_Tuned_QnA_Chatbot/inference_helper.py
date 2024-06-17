from transformers import AutoTokenizer, TFAutoModelForQuestionAnswering
import numpy as np
import os

def preprocess_input(question, context) :
  MAX_LENGTH = 512
  STRIDE = 256
  tokenized_data = tokenizer(question,
                             context,
                             max_length=MAX_LENGTH,
                             truncation='only_second',
                             return_overflowing_tokens=True,
                             return_offsets_mapping=True,
                             stride=STRIDE,
                             padding='max_length',
                             return_tensors='np')
  return tokenized_data
 
 def predict_long_context(question, context, model) :
  MAX_LENGTH = 512
  STRIDE = 256
  scores = []
  starts = []
  ends = []
  answers = []
  input_data = preprocess_input(question, context)
  sequence_ids = input_data.sequence_ids()

  ctx_start = 0
  ctx_end = len(sequence_ids) - 1

  while sequence_ids[ctx_start] != 1 :
    ctx_start += 1
  
  while sequence_ids[ctx_end] != 1 :
    ctx_end -= 1
  
  num_samples = len(input_data['input_ids'])
  for i in range(num_samples) :
    input_cleaned = {'input_ids':np.expand_dims(input_data['input_ids'][i], axis=0),
                     'token_type_ids':np.expand_dims(input_data['token_type_ids'][i], axis=0),
                     'attention_mask':np.expand_dims(input_data['attention_mask'][i], axis=0)}
    outputs = model(input_cleaned)
    start_pred = np.argmax(outputs.start_logits[0])
    end_pred = np.argmax(outputs.end_logits[0])
    answers.append(tokenizer.decode(input_cleaned['input_ids'][0, start_pred:end_pred+1]))

    if start_pred < ctx_start or end_pred > ctx_end :
      scores.append(0)
      starts.append(0)
      ends.append(0)
    else :
      scores.append(outputs.start_logits[0][start_pred] + outputs.end_logits[0][end_pred])
      starts.append(start_pred)
      ends.append(end_pred)
  
  highest_score = np.argmax(np.array(scores))
  start_max = starts[highest_score]
  end_max = ends[highest_score]

  if scores[highest_score] > 0 :
    answer = answers[highest_score]
    return answer
  else :
    return "Maaf, kami tidak bisa menjawab pertanyaan yang diberikan."