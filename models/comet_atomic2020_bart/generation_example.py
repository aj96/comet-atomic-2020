import math
import json
import torch
import argparse
from tqdm import tqdm
from pathlib import Path
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from utils import calculate_rouge, use_task_specific_params, calculate_bleu_score, trim_batch


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i : i + n]

def cosine_similarity(a, b):
    a = a.reshape(-1)
    b = b.reshape(-1)
    return torch.dot(a,b) / (torch.norm(a) * torch.norm(b))


class Comet:
    def __init__(self, model_path):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_path).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        task = "summarization"
        use_task_specific_params(self.model, task)
        self.batch_size = 1
        self.decoder_start_token_id = None

    def generate(
            self,
            queries,
            decode_method="beam",
            num_generate=5,
            ):
        """
        Given list of queries, this generates num_generate sentences for each query.
        Returns:
            decs: list of lists of generated sentences (B lists)
            decs_embeddings [B, L, H]: embeddings for the sentences
        """

        with torch.no_grad():
            examples = queries

            decs = []
            decs_embeddings = []
            for batch in list(chunks(examples, self.batch_size)):
                batch = self.tokenizer(batch, return_tensors="pt", truncation=True, padding="max_length").to(self.device)
                input_ids, attention_mask = trim_batch(**batch, pad_token_id=self.tokenizer.pad_token_id)

                summaries = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    decoder_start_token_id=self.decoder_start_token_id,
                    num_beams=num_generate,
                    num_return_sequences=num_generate,
                    )

                dec = self.tokenizer.batch_decode(summaries, skip_special_tokens=True, clean_up_tokenization_spaces=False)
                decs.append(dec)

                # converts list of strings to dictionary containing input_ids, masks, etc for input to transformer
                dec_batch = self.tokenizer(dec, return_tensors="pt", truncation=True, padding="max_length").to(self.device)
                dec_input_ids, _ = trim_batch(**dec_batch, pad_token_id=self.tokenizer.pad_token_id)
                dec_embeddings = self.ids2embeddings(dec_input_ids)
                decs_embeddings.append(dec_embeddings)
        
            # Return output as string as well as string converted to embeddings
            # [B, L, H]
            decs_embeddings = torch.cat(decs_embeddings, dim=1)
            return decs, decs_embeddings
    
    def ids2embeddings(self, input_ids):
        """
        Args:
            input_ids (pytorch tensor [B, L])
        Returns:
            result (pytorch tensor [B, L, H]) where H is output dimension of embedding layer
        """
        return self.model.get_input_embeddings()(input_ids)
    
    def str2embeddings(self, sentences):
        """
        Args:
            sentences: list of strings, each string is a sentence
        Returns:
            result (pytorch tensor [B, L, H]): B is number of sentences, L is max length of sentences, H is number of dimensions of embedding layer
        """
        batch = self.tokenizer(sentences, return_tensors="pt", truncation=True, padding="max_length").to(self.device)
        input_ids, _ = trim_batch(**batch, pad_token_id=self.tokenizer.pad_token_id)
        embeddings = self.ids2embeddings(input_ids)
        return embeddings

def bidirectional_bfs(start_sentence, end_sentence, top_k, thresh=0.95, model_path="comet-atomic_2020_BART"):
    """
    Args:
        top_k: how many
    """
    print("model loading ...")
    comet = Comet(model_path)
    comet.model.zero_grad()
    print("model loaded")


    embeddings = comet.str2embeddings([start_sentence, end_sentence])
    start_embedding = embeddings[0]
    end_embedding = embeddings[1]

    forward_q = [(start_sentence, start_embedding)]
    backward_q = [(end_sentence, end_embedding)]

    all_forward_results = []
    all_backward_results = []

    while len(forward_q) > 0 or len(backward_q) > 0:
        if len(forward_q) > 0:
            forward_sentence, forward_embedding = forward_q.pop(0)
            all_forward_results.append(forward_sentence)
            rel = "isBefore"
            queries = ["{} {} [GEN]".format(forward_sentence, rel)]
            forward_results, _ = comet.generate(queries, decode_method="beam", num_generate=top_k)
        
        if len(backward_q) > 0:
            backward_sentence, backward_embedding = backward_q.pop(0)
            all_backward_results.append(backward_sentence)
            rel = "isAfter"
            queries = ["{} {} [GEN]".format(backward_sentence, rel)]
            backward_results, _ = comet.generate(queries, decode_method="beam", num_generate=top_k)
        
        similarity_score = cosine_similarity(forward_embedding, backward_embedding)
        if similarity_score >= thresh:
            print("{} and {} had similarity score of {}".format(forward_sentence, backward_sentence, similarity_score))
            final_forward_sentences = backtrack(all_forward_results, top_k)
            final_backward_sentences = backtrack(all_backward_results, top_k)

            return final_forward_sentences + final_backward_sentences[::-1][1:]
        
        # Need to ensure forward and backward sentences have same dimensions for computing
        # cosine similarity
        for forward_result, backward_result in zip(forward_results[0], backward_results[0]):
            forward_embedding, backward_embedding = comet.str2embeddings([forward_result, backward_result])
            forward_q.append((forward_result, forward_embedding))
            backward_q.append((backward_result, backward_embedding))

def backtrack(all_sentences, b):
    """
    Do integer math to backtrack/recover path taken to reach the last sentence
    Starting from the last node reached in the graph using BFS, where index represents
    the ith node that was discovered in the graph during the bfs search (root node is index == 0)
    the next index would be given by this equation: index := ceil(index / b) - 1
    where b is branching factor or number of children each node has. We are taking advantage
    of the fact that every node has b children. In our case, b = top_k where top_k is number of sentences we chose to generate for each query.
    """

    # index starts at last node discovered which terminated our bfs search
    ind = len(all_sentences) - 1
    result = []
    # now let's backtrack to recover the path taken to reach this terminating node
    while ind != 0:
        result.append(all_sentences[ind]) 
        ind = math.ceil(float(ind) / float(b)) - 1
    # don't forget to add root-node
    result.append(all_sentences[0])
    return result[::-1]

            

all_relations = [
    "AtLocation",
    "CapableOf",
    "Causes",
    "CausesDesire",
    "CreatedBy",
    "DefinedAs",
    "DesireOf",
    "Desires",
    "HasA",
    "HasFirstSubevent",
    "HasLastSubevent",
    "HasPainCharacter",
    "HasPainIntensity",
    "HasPrerequisite",
    "HasProperty",
    "HasSubEvent",
    "HasSubevent",
    "HinderedBy",
    "InheritsFrom",
    "InstanceOf",
    "IsA",
    "LocatedNear",
    "LocationOfAction",
    "MadeOf",
    "MadeUpOf",
    "MotivatedByGoal",
    "NotCapableOf",
    "NotDesires",
    "NotHasA",
    "NotHasProperty",
    "NotIsA",
    "NotMadeOf",
    "ObjectUse",
    "PartOf",
    "ReceivesAction",
    "RelatedTo",
    "SymbolOf",
    "UsedFor",
    "isAfter",
    "isBefore",
    "isFilledBy",
    "oEffect",
    "oReact",
    "oWant",
    "xAttr",
    "xEffect",
    "xIntent",
    "xNeed",
    "xReact",
    "xReason",
    "xWant",
    ]

if __name__ == "__main__":

    start_sentence = "PersonX eats an apple"
    # NOTE: Need to add the "for PersonX" at the end to help the search converge faster
    end_sentence = "PersonX goes home"
    model_path = "./comet-atomic_2020_BART"
    result = bidirectional_bfs(start_sentence=start_sentence, end_sentence=end_sentence, top_k=5, thresh=0.95, model_path=model_path)
    print(result)

    # sample usage
    # print("model loading ...")
    # comet = Comet("./comet-atomic_2020_BART")
    # comet.model.zero_grad()
    # print("model loaded")

    # queries = []
    # head = "PersonX eats an apple"
    # rel = "isBefore"
    # query = "{} {} [GEN]".format(head, rel)
    # queries.append(query)
    # print(queries)
    # # sentences = ["I am a boy", "Where are you going"]
    # embeddings = comet.str2embeddings(sentences)
    # score = cosine_similarity(embeddings[0], embeddings[1])
    # print("score: ", score)
    # results, embeddings = comet.generate(queries, decode_method="beam", num_generate=5)
    # print("embeddings.shape: ", embeddings.shape)
    # print(results)

