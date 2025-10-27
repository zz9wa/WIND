import numpy as np
#from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI
import torch
import torch.nn.functional as F
#from utils import(cosine_similarity, find_most_similar_sample)
from sklearn.feature_extraction.text import TfidfVectorizer
import re
import json

def calculate_similarity(logits, train_embeddings):

    similarities = F.cosine_similarity(logits.unsqueeze(0), train_embeddings.unsqueeze(0), dim=-1)
    return similarities

def find_most_similar(batch_logits, train_embeddings):

    batch_size = batch_logits.shape[0]
    #train_size = train_embeddings.shape[0]


    #similarity_indices = []


    batch_vector = batch_logits.unsqueeze(0)  # shape: (1, 1024)


    cos_sim = F.cosine_similarity(batch_vector, train_embeddings)
    #print("cos_sim",cos_sim)


    #sorted_indices = torch.argsort(cos_sim, descending=True)
    topk_values, topk_indices = torch.topk(cos_sim, 1)


    #second_highest_index = sorted_indices[1].item()
    second_highest_similarity = topk_values[0].item()
    second_highest_idx = topk_indices[0].item()


    #similarity_indices.append(second_highest_index)

    return second_highest_idx, second_highest_similarity

def covert2list(logits):
    all_logits = []


    for sample_logits in logits:

        logits_list = sample_logits.tolist()
        all_logits.append(logits_list)


    return all_logits

def get_most_similar_sample(batch_sample, train_samples, train_embeddings,train_label,args):

    len_pos = len(train_samples)/2
    index,similarities = find_most_similar(batch_sample, train_embeddings)
    if index+1>len_pos:
        sim_pred=0
        #print("sim 0 similar with NEG!")
    elif index+1<=len_pos and similarities<0.8:
        sim_pred = 0
        #print("sim 0 sim_probos=",similarities)
        #print("sim_thres=",sim_thres)
    elif index+1<=len_pos and similarities>=0.8:
        sim_pred = 1

    most_similar_sample =train_samples[index]
    return most_similar_sample, sim_pred, similarities

def get_prompt_pos(sentence1, sentence2):
    prompt = f"""
        You are a text style protector. Sentence 2 is a style-transferred version of sentence 1. A style-transferred sentence means that although the content has changed, the language style remains consistent. Please follow these steps:

        **Analysis** :
            - Evaluate the similarities between **sentence 1** and **sentence 2** based on the following five aspects:
             - **Vocabulary and Word Choice**: Consider whether the two sentences use similar vocabulary or use a specific type of language related to the topic, write what they have in common, e.g., Old English, Internet slang, etc.
             - **Syntactic Structure and Grammatical Features**: Look for similarities in sentence structure specific to the topic, like technical terminology or specialized grammar.
             - **Rhetorical Devices and Stylistic Choices**: Identify the use of rhetorical devices specific to the topic, such as scientific metaphors, historical allusions, etc.
             - **Tone and Sentiment**: Compare the tone and sentiment in both sentences within the context of the topic being discussed, Such as narcissism, pessimism, cynicism, etc.
             - **Rhythm and Flow**: Evaluate the rhythm and flow of the sentences in relation to the topic, considering any stylistic choices related to the topic's nature.
            * Ensure each aspect is elaborated with a detailed sentence that captures the essence of the feature without introducing additional text, explanations, or line breaks. Output each description as part of the style feature list using the specified format: `style=[detailed_sentence1, detailed_sentence2, detailed_sentence3, detailed_sentence4, detailed_sentence5]`
             - **Do not include any explanations, or line breaks**. Ensure the output is a single line and follows the exact syntax.


        Sentence 1: "{sentence1}"
        Sentence 2: "{sentence2}"
        """
    #print("sentence1:",sentence1)
    #print("sentence2:",sentence2)

    return prompt


def get_prompt_neg(sentence):
    prompt = f"""
            You are asked to perform a text style migration task. You are asked to perform text style feature extraction according to the following for sentence according to five points. Please note that text style is independent of content.

            **Analysis**:
            - Analyze detailed the following five aspects of the second sentence:
                 - **Vocabulary and Word Choice**: Specify words or language choices.
                 - **Syntactic Structure and Grammatical Features**: Point out the sentence structure or grammar.
                 - **Rhetorical Devices and Stylistic Choices**: Highlight rhetorical devices or stylistic elements.
                 - **Tone and Sentiment**: Describe tone and emotional content that distinguish.
                 - **Rhythm and Flow**: Discuss rhythm, pacing, or flow.
           * Ensure each aspect is elaborated with a detailed sentence that captures the essence of the feature without introducing additional text, explanations, or line breaks. Output each description as part of the style feature list using the specified format: `style=[detailed_sentence1, detailed_sentence2, detailed_sentence3, detailed_sentence4, detailed_sentence5]`
           - **Exclude explanations, or line breaks**. Ensure the output is one line and follows the exact syntax.

            Sentence : "{sentence}"
            """

    return prompt



def query_gpt3_5(prompt,args):
    client = OpenAI(

        api_key="",
        base_url=""
    )
    completion = client.chat.completions.create(
        model="deepseek-ai/DeepSeek-V3",
        messages=[
            {"role": "system", "content": "You are a text style expert"},
            {"role": "user", "content": prompt}
        ],
    )
    answer = completion.choices[0].message.content
    return answer


def extract_pred_label_and_style(text):

    style_match = re.search(r"style=\[(.*?)\]", text)#'style=\[(".*?")\s*(?:,|$)'
    if style_match:
        #style = style_match.group(1).split(", ")
        style = [item.strip() for item in style_match.group(1).split(", ")]
        sentence = '+'.join(style)
        # style = [sentence]
        return sentence
    else:
        style = None
        #print("!!STYLE NOT FOUND!!")
        return None


def get_valid_answer(prompt, args):

    while True:

        gpt_answer = query_gpt3_5(prompt, args)
        #print(f"GPT-3.5 Response: {gpt_answer}")


        answer_split = gpt_answer.lower()
        #print("answer",answer_split )
        style = extract_pred_label_and_style(answer_split)



        if style:

            return style
        else:
            print("Incomplete answer. Retrying...")


def LLM2wm(pos_batch, neg_batch, pos_logits, neg_logits, train_text, train_emb,train_label, args):

    styles_pos=[]
    styles_neg=[]
    true_label=[]
    sim_preds=[]

    for sample, sample_embedding in zip(pos_batch, pos_logits):
        is_pos = True
        true_label.append(is_pos)

        most_similar_sample, sim_pred, sim_prob = get_most_similar_sample(sample_embedding, train_text, train_emb, train_label, args)
        if sim_pred==1:

            prompt=get_prompt_pos(sample,most_similar_sample)
            style = get_valid_answer(prompt, args)
            styles_pos.append(style)
            #print("sim 1")
        else:
            prompt = get_prompt_neg(sample)
            style = get_valid_answer(prompt, args)
            styles_neg.append(style)
        sim_preds.append(sim_pred)
        print("Style:", style)

    for sample, sample_embedding in zip(neg_batch, neg_logits):
        is_pos = False
        true_label.append(is_pos)

        most_similar_sample, sim_pred, sim_prob = get_most_similar_sample(sample_embedding, train_text, train_emb, train_label, args)
        if sim_pred==1:

            prompt=get_prompt_pos(sample,most_similar_sample)
            style = get_valid_answer(prompt, args)
            styles_pos.append(style)
            #print("sim 1")
        else:
            prompt = get_prompt_neg(sample)
            style = get_valid_answer(prompt, args)
            styles_neg.append(style)
        sim_preds.append(sim_pred)


    return styles_pos, styles_neg,true_label,sim_preds

