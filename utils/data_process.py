import os
from pytorch_transformers import BertTokenizer
from tqdm import tqdm
import pandas as pd
import pickle
import random
import numpy as np
import collections
from collections import Counter
import sys
sys.stdin.reconfigure(encoding='utf-8')
sys.stdout.reconfigure(encoding='utf-8')
def _is_chinese_char(cp):
    """Checks whether CP is the codepoint of a CJK character."""
    # This defines a "chinese character" as anything in the CJK Unicode block:
    #   https://en.wikipedia.org/wiki/CJK_Unified_Ideographs_(Unicode_block)
    #
    # Note that the CJK Unicode block is NOT all Japanese and Korean characters,
    # despite its name. The modern Korean Hangul alphabet is a different block,
    # as is Japanese Hiragana and Katakana. Those alphabets are used to write
    # space-separated words, so they are not treated specially and handled
    # like the all of the other languages.
    if ((cp >= 0x4E00 and cp <= 0x9FFF) or  #
        (cp >= 0x3400 and cp <= 0x4DBF) or  #
        (cp >= 0x20000 and cp <= 0x2A6DF) or  #
        (cp >= 0x2A700 and cp <= 0x2B73F) or  #
        (cp >= 0x2B740 and cp <= 0x2B81F) or  #
        (cp >= 0x2B820 and cp <= 0x2CEAF) or
        (cp >= 0xF900 and cp <= 0xFAFF) or  #
        (cp >= 0x2F800 and cp <= 0x2FA1F)):  #
      return True

    return False

def bert4token(tokenizer, title, attribute, value):
    # title = tokenizer.tokenize(title)
    # attribute = tokenizer.tokenize(attribute)
    # value = tokenizer.tokenize(value)

    title = tokenizer.basic_tokenizer.tokenize(title)
    attribute = tokenizer.basic_tokenizer.tokenize(attribute)
    value = tokenizer.basic_tokenizer.tokenize(value)

    tag = ['O']*len(title)

    for i in range(0,len(title)-len(value)):
        if title[i:i+len(value)] == value:
            for j in range(len(value)):
                if j==0:
                    tag[i+j] = 'B'
                else:
                    tag[i+j] = 'I'
    title_id = tokenizer.convert_tokens_to_ids(title)
    attribute_id = tokenizer.convert_tokens_to_ids(attribute)
    value_id = tokenizer.convert_tokens_to_ids(value)
    tag_id = [TAGS[_] for _ in tag]
    return title_id, attribute_id, value_id, tag_id

def nobert4token(tokenizer, title, attribute, value):

    def get_char(sent):
        tmp = []
        s = ''
        for char in sent.strip():
            if char.strip():
                cp = ord(char)
                if _is_chinese_char(cp):
                    if s:
                        tmp.append(s)
                    tmp.append(char)
                    s = ''
                else:
                    s += char
            elif s:
                tmp.append(s)
                s = ''
        if s:
            tmp.append(s)
        return tmp

    title_list = get_char(title)
    attribute_list = get_char(attribute)
    value_list = get_char(value)

    tag_list = ['O']*len(title_list)
    for i in range(0,len(title_list)-len(value_list)):
        if title_list[i:i+len(value_list)] == value_list:
            for j in range(len(value_list)):
                if j==0:
                    tag_list[i+j] = 'B'
                else:
                    tag_list[i+j] = 'I'

    title_list = tokenizer.convert_tokens_to_ids(title_list)
    attribute_list = tokenizer.convert_tokens_to_ids(attribute_list)
    value_list = tokenizer.convert_tokens_to_ids(value_list)
    tag_list = [TAGS[i] for i in tag_list]

    return title_list, attribute_list, value_list, tag_list


max_len = 40
def X_padding(ids):
    if len(ids) >= max_len:  
        return ids[:max_len]
    ids.extend([0]*(max_len-len(ids))) 
    return ids

tag_max_len = 6
def tag_padding(ids):
    if len(ids) >= tag_max_len: 
        return ids[:tag_max_len]
    ids.extend([0]*(tag_max_len-len(ids))) 
    return ids

def rawdata2pkl4nobert(path):
    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
    titles = []
    attributes = []
    values = []
    tags = []
    with open(path, 'r', encoding='utf-8') as f:
        for index, line in enumerate(tqdm(f.readlines())):
            line = line.strip('\n')
            if line:
                title, attribute, value = line.split('<$$$>')
                if attribute in ['适用季节','品牌'] and value in title and _is_chinese_char(ord(value[0])):
                    title, attribute, value, tag = nobert4token(tokenizer, title, attribute, value)
                    titles.append(title)
                    attributes.append(attribute)
                    values.append(value)
                    tags.append(tag)
    print([tokenizer.convert_ids_to_tokens(i) for i in titles[:3]])
    print([[id2tags[j] for j in i] for i in tags[:3]])
    print([tokenizer.convert_ids_to_tokens(i) for i in attributes[:3]])
    print([tokenizer.convert_ids_to_tokens(i) for i in values[:3]])

    df = pd.DataFrame({'titles': titles, 'attributes': attributes, 'values': values, 'tags': tags},
                      index=range(len(titles)))
    print(df.shape)
    df['x'] = df['titles'].apply(X_padding)
    df['y'] = df['tags'].apply(X_padding)
    df['att'] = df['attributes'].apply(tag_padding)

    index = list(range(len(titles)))
    random.shuffle(index)
    train_index = index[:int(0.9 * len(index))]
    valid_index = index[int(0.9 * len(index)):int(0.96 * len(index))]
    test_index = index[int(0.96 * len(index)):]

    train = df.loc[train_index, :]
    valid = df.loc[valid_index, :]
    test = df.loc[test_index, :]

    train_x = np.asarray(list(train['x'].values))
    train_att = np.asarray(list(train['att'].values))
    train_y = np.asarray(list(train['y'].values))

    valid_x = np.asarray(list(valid['x'].values))
    valid_att = np.asarray(list(valid['att'].values))
    valid_y = np.asarray(list(valid['y'].values))

    test_x = np.asarray(list(test['x'].values))
    test_att = np.asarray(list(test['att'].values))
    test_value = np.asarray(list(test['values'].values))
    test_y = np.asarray(list(test['y'].values))

    with open('../data/中文_适用季节.pkl', 'wb') as outp:
        pickle.dump(train_x, outp)
        pickle.dump(train_att, outp)
        pickle.dump(train_y, outp)
        pickle.dump(valid_x, outp)
        pickle.dump(valid_att, outp)
        pickle.dump(valid_y, outp)
        pickle.dump(test_x, outp)
        pickle.dump(test_att, outp)
        pickle.dump(test_value, outp)
        pickle.dump(test_y, outp)



def rawdata2pkl4bert(path, att_list):
    print("hiiiiiii")
    tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
    with open(path, 'r', encoding='unicode_escape') as f:
        lines = f.readlines()
        print ("length of lines is ", len(lines))
        for att_name in tqdm(att_list):
            print('#'*20+att_name+'#'*20)
            titles = []
            attributes = []
            values = []
            tags = []

            counter = 0
            for index, line in enumerate(lines):
                print ("line is", line)
                line = line.strip('\n')
                
                if line:
                    
                    title, attribute, value = line.split('<$$$>')
                    qwertytitles = title.split('qwerty')
                    print ("title strings are ", qwertytitles[::2])
                    print ("title location are ", qwertytitles[1::2])
                    print ("STRIPPED VALEU IS ", value.replace(" ", ""))
                    print ("JOINED QWERTY IS ", "".join(qwertytitles[::2]).replace(" ", ""))
                    if attribute in [att_name] and  value.replace(" ", "") in ("".join(qwertytitles[::2]).replace(" ", "")): #and _is_chinese_char(ord(value[0])):
                        counter += 1
                        
                        title, attribute, value, tag = bert4token(tokenizer, "".join(qwertytitles[::2]), attribute, value)
                        
                        # title, attribute, value, tag = bert4token(tokenizer, title, attribute, value)
                        for coordinate in qwertytitles[1::2]:
                            x, y = coordinate.split()
                            title.append(int(float(x)))
                            title.append(int(float(y)))


                        # print ('coordinate locations are ', qwertytitles[1::2])
                        # print ("title token is", title )
                        # print ("\n")
                        # print ("size of title token is ", len(title))
                        # print ("abracadabra\n")
                        # print ("attribute token is ", attribute)
                        # print ("\n")
                        # print ("size of attribute token is ", len(attribute))
                        # print ("\n")
                        # print ("value token is ", value)
                        # print ("\n")
                        # print ("size of value token is ", len(value))
                        # print ("\n")
                        titles.append(title)
                        attributes.append(attribute)
                        values.append(value)
                        tags.append(tag)
                    else:
                        print ("NOOOOOMATTTTTCHHHHH________________________")
                print ("endd_________________________________________________")
                print ("endd_________________________________________________")
                print ("endd_________________________________________________")
            print ("COUNTER IS ", counter)
            print ("length of titles is ", len(titles))
            if titles:
                print([tokenizer.convert_ids_to_tokens(i) for i in titles[:3]])
                print([[id2tags[j] for j in i] for i in tags[:3]])
                print([tokenizer.convert_ids_to_tokens(i) for i in attributes[:3]])
                print([tokenizer.convert_ids_to_tokens(i) for i in values[:3]])
                df = pd.DataFrame({'titles':titles,'attributes':attributes,'values':values,'tags':tags}, index=range(len(titles)))
                print("DF SHAPE IS ", df.shape)
                df['x'] = df['titles'].apply(X_padding)
                print ("DF TITLES SHAPE IS ", df['titles'].shape)
                df['y'] = df['tags'].apply(X_padding)
                print ("DF TAGS SHAPE IS ", df['tags'].shape)
                df['att'] = df['attributes'].apply(tag_padding)
                print ("DF ATTRIBUTES SHAPE IS ", df['attributes'].shape)

                index = list(range(len(titles)))
                random.shuffle(index)
                train_index = index[:int(0.85*len(index))]
                valid_index = index[int(0.85*len(index)):int(0.95*len(index))]
                test_index = index[int(0.95*len(index)):]

                train = df.loc[train_index,:]
                valid = df.loc[valid_index,:]
                test = df.loc[test_index,:]

                train_x = np.asarray(list(train['x'].values))
                print ("size of train x is ", train_x.shape)
                train_att = np.asarray(list(train['att'].values))
                train_y = np.asarray(list(train['y'].values))

                valid_x = np.asarray(list(valid['x'].values))
                valid_att = np.asarray(list(valid['att'].values))
                valid_y = np.asarray(list(valid['y'].values))

                test_x = np.asarray(list(test['x'].values))
                test_att = np.asarray(list(test['att'].values))
                test_value = np.asarray(list(test['values'].values), dtype=object)
                test_y = np.asarray(list(test['y'].values))

                att_name = att_name.replace('/','_')
                with open('../data/sroire_loc_tl_beta.pkl', 'wb') as outp:
                # with open('../data/top105_att.pkl', 'wb') as outp:
                    pickle.dump(train_x, outp)
                    pickle.dump(train_att, outp)
                    pickle.dump(train_y, outp)
                    pickle.dump(valid_x, outp)
                    pickle.dump(valid_att, outp)
                    pickle.dump(valid_y, outp)
                    pickle.dump(test_x, outp)
                    pickle.dump(test_att, outp)
                    pickle.dump(test_value, outp)
                    pickle.dump(test_y, outp)

def get_attributes(path):
    atts = []
    with open(path, 'r', encoding='unicode_escape') as f:
        for line in f.readlines():
            line = line.strip('\n')
            if line:
                title, attribute, value = line.split('<$$$>')
                atts.append(attribute)
    return [item[0] for item in Counter(atts).most_common()]


if __name__=='__main__':
    TAGS = {'':0,'B':1,'I':2,'O':3}
    id2tags = {v:k for k,v in TAGS.items()}
    path = '../parsed_sroire_loc_tl_qwerty.txt'
    att_list = get_attributes(path)
    print ("attributes are", att_list)
    rawdata2pkl4bert(path, att_list)
    # rawdata2pkl4nobert(path)