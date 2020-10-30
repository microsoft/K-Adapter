# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
"""
Input：raw files of T_Rex dataset
output："T_REx_example.json"

example：
{'docid': 'Q3343462',
 'token': ['Nora','Zaïdi','(','Nora','Mebrak','-','Zaïdi','),','born','on','July','6',',','1965','in','Bethoncourt','(','French','département','of',
  'Doubs','),','daughter','of','an','Algerian','textile','toiler',',','is','a','French','activist','who','seated','in','the','European','Parliament','from',
  '1989','to','1994','.'],
  'relation': 'P31',
  'subj_start': 20,
  'subj_end': 20,
  'obj_start': 18,
  'obj_end': 18,
  'subj_label': 'Q3361',
  'obj_label': 'Q6465'}
"""
import json
import multiprocessing
import time
import os

def is_beginning_of_word(token, tokenizer):
    i = tokenizer._convert_token_to_id(token)
    if i < 4:  # self.source_dictionary.nspecial: vocab_list --> 4
        # special elements are always considered beginnings
        return True
    tok = token
    if tok.startswith('madeupword'):
        return True
    try:
        return tokenizer.decode(i, clean_up_tokenization_spaces=False).startswith(' ')
    except ValueError:
        return True

def cal_entity_start_index(is_start, start):
    count = 0  # the first world in is_start is always False.
    for index, i in enumerate(is_start):
        if i :
            count += 1
        if count == start + 1:
            break
    result = index + 1  # for the <s> token.
    return result

def cal_entity_end_index(is_start, end):
    count = 0
    for index, i in enumerate(is_start):
        if i:
            count += 1
        if count == end + 1 + 1:
            break
    result = index - 1 + 1  # -1 if for finding the last word, +1 is for <s>
    return result

def func(file):
    print(file)
    examples = []
    char_index_2_token_index_errors_counts = 0
    with open(file) as fin:
        elements = json.load(fin)
        for i, ele in enumerate(elements):
            # if i > 100:
            #   break
            text = ele["text"]
            triples = ele["triples"]
            sentences_boundaries = ele["sentences_boundaries"]
            word_boundaries = ele["words_boundaries"]
            for tri in triples:
                sent_bound = sentences_boundaries[tri['sentence_id']]
                entity1_index = tri["subject"]["boundaries"]
                entity1_uri = tri["subject"]["uri"]
                entity2_index = tri["object"]["boundaries"]
                entity2_uri = tri["object"]["uri"]
                predicate = tri["predicate"]["uri"]
                predicate = predicate.split("/")[-1]
                entity1_label = entity1_uri.split("/")[-1]
                entity2_label = entity2_uri.split("/")[-1]
                if entity1_index and entity2_index:
                    sentence = []
                    word_index_in_sentence = []
                    word_index = []
                    for word_bound in word_boundaries:
                        if word_bound[0] >= sent_bound[0] and word_bound[1] <= sent_bound[1]:
                            sentence.append(text[word_bound[0]: word_bound[1]])
                            word_index_in_sentence.append(word_bound)
                            word_index.append([word_bound[0]-sent_bound[0],word_bound[1]-sent_bound[0]])
                    count = 0
                    for j, word_bound in enumerate(word_index_in_sentence):
                        if word_bound[0] == entity1_index[0]:
                            entity1_index[0] = j
                            count += 1
                        if word_bound[1] == entity1_index[1]:
                            entity1_index[1] = j
                            count += 1
                        if word_bound[0] == entity2_index[0]:
                            entity2_index[0] = j
                            count += 1
                        if word_bound[1] == entity2_index[1]:
                            entity2_index[1] = j
                            count += 1
                    if count != 4:
                        char_index_2_token_index_errors_counts += 1
                        continue  # drop
                    example= {}
                    example['docid'] = ele['docid'].split("/")[-1]
                    example['token'] = sentence
                    example['relation'] = predicate
                    example['subj_start'] = entity1_index[0]
                    example['subj_end'] = entity1_index[1]
                    example['obj_start'] = entity2_index[0]
                    example['obj_end'] = entity2_index[1]
                    example['subj_label'] = entity1_label
                    example['obj_label'] = entity2_label
                    examples.append(example)
                else:
                    pass
    return [examples]

if __name__ == "__main__":
    start_time = time.time()
    pool = multiprocessing.Pool(processes=os.cpu_count())
    data_path = "data/TREx/re-nlg_data" # the path of raw T_Rex dataset
    file_list = os.listdir(data_path)

    results = []
    for file in file_list:
        results.append(pool.apply_async(func, (os.path.join(data_path,file),)))

    pool.close()
    pool.join()

    print('cost time: ', time.time()-start_time)

    examples = []
    skip_examples = []
    char_index_2_token_index_errors_counts = 0
    wrong_res_count = 0
    for res in results:
        try:
            res = res.get()
            example = res[0]
            examples += example
        except BaseException as e:
            print("Error found in res.get()!")
            wrong_res_count +=1
            print(res)

    # predicate->id
    pred2id = {}
    for example in examples:
        if example["relation"] in pred2id:
            example["relation"] = pred2id[example["relation"]]
        else:
            pred2id[example["relation"]] = len(pred2id)
            example["relation"] = pred2id[example["relation"]]

    print("wrong_res_count: ", wrong_res_count)
    print("totoal examples :", len(examples))
    print("total considered predicates: ", len(pred2id))
    print("wrong_res_count: ", wrong_res_count)

    # Saving
    save_path = "./data/cleaned_T_REx"
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    with open(os.path.join(save_path,"T_REx_pred2ic.json"), "w") as fout_pred2id:
        json.dump(pred2id, fout_pred2id)
    with open(os.path.join(save_path, "T_REx_example.json"), "w") as fout_ex:
        json.dump(examples, fout_ex)