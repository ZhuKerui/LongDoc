
from typing import cast
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer, GenerationConfig, BatchEncoding, pipeline
from openai import OpenAI
import concurrent.futures
from sklearn.metrics.pairwise import paired_cosine_distances

from .base import *
from .base_utils import get_synonym_pairs

def hidden_states_wo_instruction(input_ids:List[np.ndarray], hidden_states:np.ndarray, attention_masks:np.ndarray, instruction_masks:np.ndarray, normalized:bool=True):
    for iid, instruction_mask in enumerate(instruction_masks):
        attention_masks[iid, :sum(instruction_mask)-1] = 0
    input_ids = [input_id[mask>0] for input_id, mask in zip(input_ids, attention_masks)]
    s = np.sum(hidden_states * np.expand_dims(attention_masks, -1), axis=1)
    d = attention_masks.sum(axis=1, keepdims=True)
    embeddings = s / d
    temp_hidden_states:List[np.ndarray] = [hs[mask>0] for hs, mask in zip(hidden_states, attention_masks)]
    if normalized:
        temp_hidden_states = [lhs / np.linalg.norm(emb) for lhs, emb in zip(temp_hidden_states, embeddings)]
        embeddings = np.vstack([emb / np.linalg.norm(emb) for emb in embeddings])
    return input_ids, embeddings, temp_hidden_states
    
class GritLM(torch.nn.Module):
    def __init__(
        self,
        model_name_or_path: str = None,
        mode: str = 'unified', # One of ['unified', 'embedding', 'generative']        
        pooling_method: str = 'mean', # One of ['cls', 'lasttoken', 'mean', 'weightedmean']
        normalized: bool = True,
        projection: int = None,
        is_inference: bool = True,
        embed_eos: str = "",
        attn: str = 'bbcc',
        **kwargs, # Passed to the model, e.g. `attn_implementation`, `torch_dtype` etc.
    ) -> None:
        super().__init__()
        if mode == 'embedding':
            if any([x in model_name_or_path for x in ['gtr', 't5', 'instructor']]):
                # Somehow AutoModel does not pick the right one by default
                from transformers import T5EncoderModel
                self.model = T5EncoderModel.from_pretrained(model_name_or_path, **kwargs)
            else:
                self.model = AutoModel.from_pretrained(model_name_or_path, trust_remote_code=True, **kwargs)
            self.embedding_attr = None
        else:
            self.model = AutoModelForCausalLM.from_pretrained(model_name_or_path, trust_remote_code=True, **kwargs)
            self.generate = self.model.generate

            if hasattr(self.model, 'model'): # LLama2 & Mistral
                self.embedding_attr = 'model'
            elif hasattr(self.model, 'transformer'): # GPT-Neo & GPT-J
                self.embedding_attr = 'transformer'
            else: 
                raise ValueError("Could not find attribute to use for embedding: ", self.model)

        self.projection = torch.nn.Linear(
            in_features=self.model.config.hidden_size, 
            out_features=int(projection),
            dtype=self.model.dtype
        ) if projection is not None else None
        self.normalized = normalized
        self.pooling_method = pooling_method

        # self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.device = self.model.device
        self.num_gpus = 1
        self.embed_eos = embed_eos
        self.attn = attn
        if (self.attn is not None) and self.attn not in ['bbcc', 'cccc', 'bb', 'cc']:
            raise ValueError(f"Mixed attention no longer supported: {self.attn}. Only bbcc, cccc, bb, cc are supported")

        print(f"Created GritLM: {self.model.dtype} dtype, {pooling_method} pool, {mode} mode, {attn} attn")

        if is_inference:
            # Padding side right is necessary for `embed_instruction` to index correctly
            self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, padding_side='right')
            if not(self.tokenizer.pad_token) and self.tokenizer.eos_token:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                print('Set pad token to eos token: ' + self.tokenizer.pad_token)        
            if self.embed_eos:
                assert self.embed_eos in self.tokenizer.vocab, f"EOS token {self.embed_eos} not in vocab"
            self.model.eval()
            if not("device_map" in kwargs):
                self.model.to(self.device)
                # Parallelize embedding model
                if mode == 'embedding':
                    self.num_gpus = torch.cuda.device_count()
                    if self.num_gpus > 1:
                        print(f"----------Using {self.num_gpus} data-parallel GPUs----------")
                        self.model = torch.nn.DataParallel(self.model)

    def encode_queries(self, queries: List[str] | str, **kwargs) -> np.ndarray:
        """Used for encoding the queries of retrieval or reranking tasks"""
        return self.encode(queries, **kwargs)

    def encode_corpus(self, corpus: List[str] | str | List[Dict[str, str]], **kwargs) -> np.ndarray:
        """Used for encoding the corpus of retrieval tasks"""
        if isinstance(corpus, dict):
            corpus = [corpus]
        if isinstance(corpus, list) and isinstance(corpus[0], dict):
            corpus = [
                doc["title"] + " " + doc["text"] if "title" in doc 
                else doc["text"] for doc in corpus
            ]
        return self.encode(corpus, **kwargs)

    def gritlm_instruction(self, instruction):
        return "<|user|>\n" + instruction + "\n<|embed|>\n" if instruction else "<|embed|>\n"
    
    @torch.no_grad()
    def encode(
        self,
        sentences: List[str],
        batch_size: int = 256,
        max_length: int = 512,
        instructions: List[str] = [],
        embed_instruction: bool = False,
        get_cache: bool = False,
        convert_to_tensor: bool = False,
        recast: bool = False,
        add_special_tokens: bool = True,
        **kwargs,
    ) -> np.ndarray:
        assert len(sentences) == len(instructions) or not instructions
        if not instructions:
            instructions = [self.gritlm_instruction('')] * len(sentences)
        if self.num_gpus > 1:
            batch_size *= self.num_gpus

        all_embeddings, all_kv_caches, input_ids, last_hidden_states = [], [], [], []
        for start_index in tqdm(range(0, len(sentences), batch_size), desc="Batches", disable=len(sentences)<256):
            instructions_batch = instructions[start_index:start_index + batch_size]
            sentences_batch = [
                instruction + s + self.embed_eos for s, instruction in zip(sentences[start_index:start_index + batch_size], instructions_batch)
            ]
            # This will prepend the bos token if the tokenizer has `add_bos_token=True`
            inputs = self.tokenizer(
                sentences_batch,
                padding=True,
                truncation=True,
                return_tensors='pt',
                max_length=max_length,
                add_special_tokens=add_special_tokens,
            ).to(self.device)

            if (self.attn is not None) and (self.attn[:2] == 'bb'):
                inputs["is_causal"] = False
            if get_cache:
                inputs['use_cache'] = True
            outputs = (
                getattr(self.model, self.embedding_attr) if self.embedding_attr else self.model
            )(**inputs)
            last_hidden_state = outputs[0]
            if get_cache:
                # Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape `(batch_size, num_heads, sequence_length, embed_size_per_head)`
                assert len(all_kv_caches) == 0, "Can only get cache for one batch at a time"
                all_kv_caches = outputs[1]

            if self.projection:
                last_hidden_state = self.projection(last_hidden_state)
            if (instructions) and (embed_instruction is False) and ("mean" in self.pooling_method):
                # Remove instruction tokens from the embeddings by masking them
                instruction_masks = self.tokenizer(
                    instructions_batch,
                    padding=True,
                    truncation=True,
                    return_tensors='pt',
                    max_length=max_length,
                    add_special_tokens=add_special_tokens,
                )["attention_mask"]
                for iid, instruction_mask in enumerate(instruction_masks):
                    inputs['attention_mask'][iid, :sum(instruction_mask)] = 0
            input_ids.extend([input_id[mask>0].cpu().numpy() for input_id, mask in zip(inputs['input_ids'], inputs['attention_mask'])])
            last_hidden_state = last_hidden_state.to(inputs['attention_mask'].device)
            temp_last_hidden_states = [lhs[mask>0].cpu().to(torch.float32).numpy() for lhs, mask in zip(last_hidden_state, inputs['attention_mask'])]
            embeddings = self.pooling(last_hidden_state, inputs['attention_mask'], recast=recast)
            # Normalize can change the dtype (https://discuss.pytorch.org/t/tensor-in-float16-is-transformed-into-float32-after-torch-norm/110891)
            if self.normalized: 
                in_dtype = embeddings.dtype
                temp_last_hidden_states = [lhs / torch.norm(emb).item() for lhs, emb in zip(temp_last_hidden_states, embeddings)]
                embeddings = torch.nn.functional.normalize(embeddings, dim=-1).to(in_dtype)
            embeddings = cast(torch.Tensor, embeddings)
            if convert_to_tensor:
                all_embeddings.append(embeddings)
                last_hidden_states.extend(temp_last_hidden_states)
            else:
                # NumPy does not support bfloat16
                all_embeddings.append(embeddings.cpu().to(torch.float32).numpy())
                last_hidden_states.extend(temp_last_hidden_states)

        all_embeddings = (
            torch.cat(all_embeddings, dim=0) if convert_to_tensor else np.concatenate(all_embeddings, axis=0)
        )
        if get_cache:
            # all_kv_caches = (
            #     torch.stack(all_kv_caches, dim=0) if convert_to_tensor else np.concatenate(all_kv_caches, axis=0)
            # )
            return all_embeddings, all_kv_caches
        return all_embeddings, input_ids, last_hidden_states

    def pooling(
        self, hidden_state: torch.Tensor, attention_mask: torch.Tensor = None, recast: bool = False
    ) -> torch.Tensor:
        """
        Args:
            hidden_state: [b, n, d]
            attention_mask: [b, n]
        """
        # In case the model is distributed across multiple devices; hidden_state may end up on diff device
        hidden_state = hidden_state.to(attention_mask.device)
        if self.pooling_method == 'cls':
            embedding = hidden_state[:, 0]
        elif self.pooling_method == 'lasttoken':
            b, n, d = hidden_state.size()
            # Get the last `1` in the attention mask of each item
            # Often it is just `gather_indices = torch.argmin(attention_mask, 1, keepdim=False) - 1`
            # except when 1) There's all 1's 2) There's 0's before the 1's
            reversed_mask = torch.flip(attention_mask, dims=(1,))
            argmax_reverse = torch.argmax(reversed_mask, dim=1, keepdim=False)
            gather_indices = attention_mask.size(1) - argmax_reverse - 1
            # If there are empty sequences, where the index would become -1 it will crash so set them to 0
            gather_indices = torch.clamp(gather_indices, min=0)
            # Turn indices from shape [b] -> [b, 1, d]
            gather_indices = gather_indices.unsqueeze(-1).repeat(1, d)
            gather_indices = gather_indices.unsqueeze(1)
            assert gather_indices.shape == (b, 1, d)
            # Gather along the seq len: [b, n, d] -> [b, d]
            # Actually no need for the attention mask as we gather the last token where attn_mask=1 but
            # as some indices (which shouldn't be attended to) may be 0 due to clamp, use mask to ignore them again
            input_mask_expanded = attention_mask.unsqueeze(-1).expand((b, n, d)).float()
            embedding = torch.gather(hidden_state * input_mask_expanded, 1, gather_indices).squeeze(dim=1)
        elif self.pooling_method in ['mean', 'weightedmean']:
            if self.pooling_method == 'weightedmean':
                attention_mask *= attention_mask.cumsum(dim=1) # [0,1,1,1,0,0] -> [0,1,2,3,0,0]
            s = torch.sum(hidden_state * attention_mask.unsqueeze(-1).float(), dim=1)
            d = attention_mask.sum(dim=1, keepdim=True).float()
            embedding = s / d
        else: raise NotImplementedError(f"Unknown pooling method: {self.pooling_method}")
        # Recasting performs slightly worse but saves 50% space
        if recast: return embedding.to(hidden_state.dtype)
        return embedding


class LLM:
    def __init__(self, llm_name:str="mistralai/Mistral-7B-Instruct-v0.2", device_map:str='auto', batch_size:int=3) -> None:
        self.generator = pipeline("text-generation", llm_name, device_map=device_map, batch_size=batch_size, torch_dtype=torch.float16)
        self.generator.tokenizer.pad_token_id = self.generator.tokenizer.eos_token_id
        
    def __call__(self, prompt:str | List[str], n:int=1, temperature:float=0.0) -> List[List[str]]:
        config = GenerationConfig(
            num_return_sequences=n, 
            max_new_tokens=1000, 
            do_sample=True, 
            temperature=0.001 if temperature == 0. else temperature, 
            pad_token_id=self.generator.tokenizer.eos_token_id, 
            return_full_text=False)
        
        if isinstance(prompt, str):
            prompt = [prompt]
        return [
            [
                gen['generated_text'][-1]['content'] 
                for gen in p_gen
            ] 
            for p_gen in self.generator([[{"role": "user", "content": p}] for p in prompt], generation_config=config)]
        

class LLMServer:
    def __init__(self, llm:str | LLM ="mistralai/Mistral-7B-Instruct-v0.2") -> None:
        self.llm:LLM = None
        if isinstance(llm, LLM):
            self.llm = llm
        else:
            self.llm_name = llm
            if 'gpt' in self.llm_name:
                # self.doc_split = DocSplit("mistralai/Mistral-7B-Instruct-v0.2")
                self.llm_server = OpenAI()
            else:
                # self.doc_split = DocSplit(self.llm_name)
                
                # Set OpenAI's API key and API base to use vLLM's API server.
                openai_api_key = "EMPTY"
                # openai_api_base = "http://localhost:8000/v1"
                openai_api_base = 'http://128.174.136.28:8000/v1'
                self.llm_server = OpenAI(
                    api_key=openai_api_key,
                    base_url=openai_api_base,
                )
    
    def single_api_call(self, prompt:str, n:int=1, temperature:float=0.0, return_str:bool=True):
        chat_response = self.llm_server.chat.completions.create(
            model=self.llm_name,
            messages=[
                {"role": "user", "content": prompt},
            ],
            n=n,
            temperature=temperature,
        )
        return [str(chat_response.choices[i].message.content) for i in range(n)] if return_str else chat_response


    def __call__(self, prompt:str | List[str], n:int=1, temperature:float=0.0, return_str:bool=True):
        if self.llm is None:
            if isinstance(prompt, str):
                return [self.single_api_call(prompt, n, temperature, return_str)]
            else:
                with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
                    futures = [executor.submit(self.single_api_call, p, n, temperature, return_str) for p in prompt]
                    return [f.result() for f in futures]
        else:
            return self.llm(prompt, n, temperature)


class RetrieverOutput:
    def __init__(self, embeddings:np.ndarray, last_hidden_states:np.ndarray=None, retriever_input:BatchEncoding=None) -> None:
        self.embeddings = embeddings
        self.input_ids:np.ndarray
        self.attention_mask:np.ndarray
        if retriever_input:
            self.input_ids = retriever_input['input_ids'].cpu().numpy()
            self.attention_mask = retriever_input['attention_mask'].cpu().numpy()
        self.last_hidden_states = last_hidden_states
        
        
class Retriever:
    def __init__(self, retriever_name:str='intfloat/multilingual-e5-large', device='cpu', syn_dist:float=None) -> None:
        self.retriever_tokenizer = AutoTokenizer.from_pretrained(retriever_name)#, force_download=True)
        self.retriever_model = AutoModel.from_pretrained(retriever_name)#, force_download=True)
        if device == 'cpu':
            self.device = None
        else:
            self.device = torch.device(device)
            self.retriever_model.cuda(device=self.device)
        # Get synonyms boundary
        
        if syn_dist:
            self.syn_dist = syn_dist
        else:
            synonym_pairs = get_synonym_pairs()
            word_list1, word_list2 = zip(*synonym_pairs)
            emb1 = self.embed_paragraphs(word_list1)
            emb2 = self.embed_paragraphs(word_list2)
            self.syn_dist = paired_cosine_distances(emb1, emb2).mean()
        self.syn_similarity = 1 - self.syn_dist
        
    @staticmethod
    def _mean_pooling(token_embeddings:torch.Tensor, mask:torch.Tensor) -> torch.Tensor:
        token_embeddings = token_embeddings.masked_fill(~mask[..., None].bool(), 0.)
        sentence_embeddings:torch.Tensor  = token_embeddings.sum(dim=1) / mask.sum(dim=1)[..., None]
        return sentence_embeddings
    
    def embed_paragraphs(self, paragraphs:List[str], normalize:bool=False, complete_return:bool=False):
        retriever_input = self.retriever_tokenizer.batch_encode_plus(paragraphs, padding=True, truncation=True, return_tensors='pt')
        if self.device:
            retriever_input = retriever_input.to(self.device)
        with torch.no_grad():
            retriever_output = self.retriever_model(**retriever_input)
            paragraph_embeddings:np.ndarray = self._mean_pooling(retriever_output[0], retriever_input['attention_mask']).cpu().numpy()
            last_hidden_states = retriever_output[0].masked_fill(~retriever_input['attention_mask'][..., None].bool(), 0.).cpu().numpy()
        if normalize:
            norms = np.linalg.norm(paragraph_embeddings, axis=1)
            paragraph_embeddings = paragraph_embeddings / np.expand_dims(norms, axis=1)
            last_hidden_states = last_hidden_states / np.expand_dims(norms, axis=(1, 2))
        if complete_return:
            return RetrieverOutput(paragraph_embeddings, last_hidden_states, retriever_input)
        return paragraph_embeddings
    
    def dense_retrieval(self, question:str | np.ndarray, paragraphs:List[str] | np.ndarray, k:int=5, normalize:bool=False, return_score:bool=False):
        doc_embeds = paragraphs if isinstance(paragraphs, np.ndarray) else self.embed_paragraphs(paragraphs, normalize)
        query_embed = question if isinstance(question, np.ndarray) else self.embed_paragraphs([question], normalize)
        scores = doc_embeds.dot(query_embed.squeeze())
        max_indices:List[int] = np.argsort(scores)[::-1][:k].tolist()
        if return_score:
            return max_indices, [scores[i] for i in max_indices]
        return max_indices


class ReadingAgent:
    def __init__(self, dataset:Dataset, llm:str | LLM="mistralai/Mistral-7B-Instruct-v0.2", model_type:str = 'gpt') -> None:
        self.llm_server = LLMServer(llm)
        self.model_type = model_type
        self.dataset = dataset
    

    def gisting(self, example, pages, verbose=True):
        article = self.dataset.get_article(example)
        word_count = Dataset.count_words(article)

        shortened_pages = []
        for i, page in enumerate(pages):
            prompt = GeneralPrompt.shorten('\n'.join(page))
            response = self.llm_server(prompt)[0]
            shortened_text = response.strip()
            shortened_pages.append(shortened_text)
            if verbose:
                print("[gist] page {}:".format(i), shortened_text, flush=True)
        shortened_article = '\n'.join(shortened_pages)
        gist_word_count = Dataset.count_words(shortened_article)
        if verbose:
            print("Shortened article:\n", shortened_article, flush=True)
        output:dict = copy.deepcopy(example)
        output.update({'word_count': word_count, 'gist_word_count': gist_word_count, 'shortened_pages': shortened_pages, 'pages': pages})
        if verbose:
            print(f"compression rate {round(100.0 - gist_word_count/word_count*100, 2)}% ({gist_word_count}/{word_count})")
        return output
    
    
    def get_page_ids(self, response:str, max_page_num:int):
        start = response.lower().index('look up Page'.lower()) + len('look up Page')
        page_ids_str = response[start:].split('to', 1)[0].split()
        page_ids = []
        for p in page_ids_str:
            if p.strip(',.[]').isnumeric():
                page_id = int(p.strip(',.[]'))
                if page_id >= 0 and page_id <= max_page_num:
                    page_ids.append(page_id)
        return page_ids

    def main(self, query:str, page_file:str, gist_file:str, log_file:str=None, lookup_method:str='s'):
        pages = read_json(page_file)
        shortened_pages = read_json(gist_file)['shortened_pages']
        expanded_shortened_pages = shortened_pages[:]
        
        log_info(log_file, 'query', query)
        
        if lookup_method == 'p':
            menu = '\n'.join([f"<Page {i}>\n{shortened_text}" for i, shortened_text in enumerate(shortened_pages)])
            log_info(log_file, 'menu', menu)

            prompt_lookup = ReadingAgentPrompt.parallel_lookup(menu, query)
            # log_info(log_file, 'lookup_prompt', prompt_lookup)

            page_ids = []
            response = self.llm_server(prompt=prompt_lookup)[0].strip()
            log_info(log_file, 'retrieval_command', response)
            
            if 'look up Page'.lower() in response.lower():
                page_ids = self.get_page_ids(response, len(pages) - 1)

            # Memory expansion after look-up, replacing the target shortened page with the original page
            if len(page_ids) > 0:
                for page_id in page_ids:
                    expanded_shortened_pages[page_id] = '\n'.join(pages[page_id])
        else:
            retrieved_passage_nums = []
            for _ in range(7):
                menu = '\n'.join([f"<Page {i}>\n{shortened_text}" for i, shortened_text in enumerate(expanded_shortened_pages)])
                log_info(log_file, 'menu', menu)
                
                # prompt_lookup = prompt_sequential_lookup_template.format(menu, query, ', '.join(map(str, retrieved_passage_nums)))
                prompt_lookup = ReadingAgentPrompt.sequential_lookup(menu, query.split('\n')[0], ', '.join(map(str, retrieved_passage_nums)))
                # log_info(log_file, 'lookup_prompt', prompt_lookup)
                
                response = self.llm_server(prompt=prompt_lookup)[0].strip()
                log_info(log_file, 'retrieval_command', response)
                
                # page_id:List[str] = re.findall(r'page \d+', response.lower())
                
                if 'STOP' in response or 'look up Page'.lower() not in response.lower():
                    log_info(log_file, 're-read complete', 'yes')
                    break
                else:
                    # Memory expansion after look-up, replacing the target shortened page with the original page
                    if len(retrieved_passage_nums) == 6:
                        log_info(log_file, 're-read complete', 'no')
                    else:
                        page_ids = self.get_page_ids(response, len(pages) - 1)
                        new_page_id_added = False
                        for page_id in page_ids:
                            if page_id not in retrieved_passage_nums:
                                expanded_shortened_pages[page_id] = '\n'.join(pages[page_id])
                                retrieved_passage_nums.append(page_id)
                                log_info(log_file, 'retrieved_pids', retrieved_passage_nums)
                                new_page_id_added = True
                                break
                        if not new_page_id_added:
                            log_info(log_file, 're-read complete', 'yes')
                            break

        expanded_shortened_article = '\n'.join(expanded_shortened_pages)
        log_info(log_file, 'retrieval_result', expanded_shortened_article)

        response = self.llm_server(prompt=GeneralPrompt.answer(self.dataset.question_type, expanded_shortened_article, query, self.dataset.answer_format))[0]
        response = response.strip()
        log_info(log_file, 'generation', response)
    
    
class LongDoc:
    
    def __init__(self, embedding:Embeddings) -> None:
        # self.dataset = dataset
        self.embedding = embedding
        self.nlp = spacy.load('en_core_web_lg')
        
    # def retrieve_descriptions(self, notes:List[ChunkInfo], relation_graph:nx.Graph, target_ents:List[str], match_num:int=5, r_num:int=1, retrieval_guaranteed:bool=False):
    #     ent2pids = defaultdict(list)
    #     pid:int
    #     for pid, note in enumerate(notes):
    #         for ent in note.ent_descriptions.keys():
    #             ent2pids[ent].append(pid)
    #     all_ents = list(ent2pids.keys())
    #     target_ents_emb:np.ndarray = self.retriever.embed_paragraphs(target_ents, True)
    #     refer_ents_emb:np.ndarray = self.retriever.embed_paragraphs(all_ents, True)
    #     list_ent = LongDocPrompt.ListEnt()
    #     ent_map = list_ent.match_entities(target_ents, all_ents, target_ents_emb, refer_ents_emb, match_num, retrieval_guaranteed, self.retriever.syn_similarity)
    #     prev_ent_descriptions:Dict[int, Dict[str, str]] = defaultdict(dict)
    #     for _, mentions in ent_map.items():
    #         for mention in mentions:
    #             for pid in ent2pids[mention][-r_num:]:
    #                 prev_ent_descriptions[pid][mention] = notes[pid].ent_descriptions[mention]
    #     prev_relation_descriptions:Dict[int, List[Tuple[List[str], str]]] = defaultdict(list)
    #     all_nodes = list(relation_graph.nodes)
    #     refer_ents_emb:np.ndarray = self.retriever.embed_paragraphs(all_nodes, True)
    #     node_map = list_ent.match_entities(target_ents, all_nodes, target_ents_emb, refer_ents_emb, match_num, retrieval_guaranteed, self.retriever.syn_similarity)
    #     sub_nodes = set()
    #     for ent, mentions in node_map.items():
    #         sub_nodes.update(mentions)
    #     sub_graph:nx.Graph = relation_graph.subgraph(sub_nodes)
    #     visited_relation_descriptions = set()
    #     for e1, e2, pids in sub_graph.edges.data('pids'):
    #         for pid in pids[-r_num:]:
    #             for rid, (related_ents, relation_description) in enumerate(notes[pid].relation_descriptions):
    #                 if e1 in related_ents and e2 in related_ents and (pid, rid) not in visited_relation_descriptions:
    #                     prev_relation_descriptions[pid].append((related_ents, relation_description))
    #                     visited_relation_descriptions.add((pid, rid))
    #     return prev_ent_descriptions, prev_relation_descriptions

    def collect_entities_from_text(self, text:str):
        doc = self.nlp(text)
        ncs = [nc if nc[0].pos_ != 'DET' else nc[1:] for nc in doc.noun_chunks if nc.root.pos_ not in ['NUM', 'PRON']]
        ents = [ent if ent[0].pos_ != 'DET' else ent[1:] for ent in doc.ents if ent.root.pos_ not in ['NUM', 'PRON']]
        ncs_spans = [(nc.start, nc.end) for nc in ncs]
        ents_spans = [(ent.start, ent.end) for ent in ents]
        nc_id, eid = 0, 0
        spans = []
        while nc_id < len(ncs_spans) and eid < len(ents_spans):
            nc_span, ent_span = ncs_spans[nc_id], ents_spans[eid]
            if set(range(*nc_span)).intersection(range(*ent_span)):
                merged_span = (min(nc_span[0], ent_span[0]), max(nc_span[1], ent_span[1]))
                spans.append(merged_span)
                nc_id += 1
                eid += 1
            else:
                if nc_span[0] < ent_span[0]:
                    spans.append(nc_span)
                    nc_id += 1
                else:
                    spans.append(ent_span)
                    eid += 1
        spans.extend(ncs_spans[nc_id:])
        spans.extend(ents_spans[eid:])
        updated_spans:List[Tuple[int, int]] = []
        for span in spans:
            doc_span = doc[span[0]:span[1]]
            if ',' in doc_span.text:
                start = doc_span.start
                for t in doc_span:
                    if t.text == ',':
                        if t.i != start:
                            updated_spans.append((start, t.i))
                        start = t.i + 1
                if start < span[1]:
                    updated_spans.append((start, span[1]))
            else:
                updated_spans.append(span)
        updated_spans = [span for span in updated_spans if any([t.pos_ in ['NOUN', 'PROPN'] for t in doc[span[0]:span[1]]])]
        updated_spans = [span if doc[span[0]].pos_ != 'PRON' else (span[0]+1, span[1]) for span in updated_spans]
        ent_candidates = [doc[span[0]:span[1]].text if (span[1] - span[0]) > 1 else doc[span[0]].lemma_ for span in updated_spans]
        ent_candidates = [ent.strip('"') for ent in ent_candidates]
        ent_candidates = [ent for ent in ent_candidates if len(ent) >= 2]
        return ent_candidates
    
    def collect_global_entities(self, paragraphs:List[str], return_ent_class:bool=False):
        ent_candidates_all = list()
        ent_candidates_pid = list()
        for pid, passage in enumerate(paragraphs):
            ent_candidates = self.collect_entities_from_text(passage)
            ent_candidates_all.extend(ent_candidates)
            ent_candidates_pid.extend([pid] * len(ent_candidates))
        ent_candidates_all_emb = self.retriever.embed_paragraphs(ent_candidates_all)
        ent_candidates_pid = np.array(ent_candidates_pid)
        db = DBSCAN(eps=self.retriever.syn_dist, min_samples=2, metric="cosine").fit(ent_candidates_all_emb)
        ent_class = defaultdict(set)
        for class_id in range(db.labels_.max() + 1):
            if len(set(ent_candidates_pid[db.labels_ == class_id])) >= 2:
                ent_class[class_id]
        ent_candidates_per_passage = defaultdict(set)
        for label, ent, pid in zip(db.labels_, ent_candidates_all, ent_candidates_pid):
            if label in ent_class:
                ent_class[label].add(ent)
                ent_candidates_per_passage[pid].add(ent)
        return ent_candidates_per_passage if not return_ent_class else (ent_candidates_per_passage, ent_class)
    
    # def index_text(self, paragraphs:List[str], w_note:bool=True, match_num:int=5, r_num:int=1):
    #     # Collect useful entities
    #     ent_candidates_per_passage = self.collect_global_entities(paragraphs)
    #     results:List[ChunkInfo] = []
    #     relation_graph = nx.Graph()
    #     for cur_pid, paragraph in enumerate(tqdm(paragraphs)):
    #         chunk_info = ChunkInfo(cur_pid, paragraph)
    #         if results and w_note:
    #             summary_recap = {pid: results[pid].summary for pid in range(max(len(results)-r_num, 0), len(results))}
    #             chunk_info.prev_summaries = summary_recap
                
    #         if cur_pid not in ent_candidates_per_passage:
    #             if results and w_note:
    #                 list_entity_prompt = LongDocPrompt.list_entity_w_note(chunk_info.recap_str, paragraph)
    #             else:
    #                 list_entity_prompt = LongDocPrompt.list_entity(paragraph)
    #             # Extract important entities
    #             chat_response = self.llm_server(list_entity_prompt, 5, 0.7)[0]
    #             important_ents = LongDocPrompt.parse_entities(chat_response, lambda x: self.retriever.embed_paragraphs(x, True), self.retriever.syn_similarity)
    #         else:
    #             important_ents = ent_candidates_per_passage[cur_pid]
    #         chunk_info.important_ents = list(important_ents)
            
    #         # Generate entity description, summary, relation description
    #         if results and w_note:
    #             chunk_info.prev_ent_descriptions, chunk_info.prev_relation_descriptions = self.retrieve_descriptions(results, relation_graph, chunk_info.important_ents, match_num, r_num)
    #             recap_str = chunk_info.recap_str
    #             ent_description_prompt = LongDocPrompt.ent_description_w_note(recap_str, paragraph, chunk_info.important_ents)
    #             summary_prompt = LongDocPrompt.shorten_w_note(recap_str, paragraph)
    #             relation_description_prompt = LongDocPrompt.relation_description_w_note(recap_str, paragraph, chunk_info.important_ents)
    #         else:
    #             ent_description_prompt = LongDocPrompt.ent_description(paragraph, chunk_info.important_ents)
    #             summary_prompt = LongDocPrompt.shorten(paragraph)
    #             relation_description_prompt = LongDocPrompt.relation_description(paragraph, chunk_info.important_ents)
            
    #         ent_description, relation_description, summary = self.llm_server([ent_description_prompt, relation_description_prompt, summary_prompt])
    #         ent_description, relation_description, summary = ent_description[0], relation_description[0], summary[0]
    #         chunk_info.summary = summary
    #         chunk_info.ent_descriptions = LongDocPrompt.parse_ent_description(ent_description, chunk_info.important_ents)
    #         chunk_info.relation_descriptions = LongDocPrompt.parse_relation_description(relation_description, chunk_info.important_ents)
    #         results.append(chunk_info)
    #         for related_ents, _ in chunk_info.relation_descriptions:
    #             for ent1, ent2 in itertools.combinations(related_ents, 2):
    #                 if not relation_graph.has_edge(ent1, ent2):
    #                     relation_graph.add_edge(ent1, ent2, pids=[])
    #                 temp_pids:List[int] = relation_graph.get_edge_data(ent1, ent2)['pids']
    #                 if cur_pid not in temp_pids:
    #                     temp_pids.append(cur_pid)
    #     return results
    
    # def index_text_into_map(self, paragraphs:List[str], nbr_max_dist:int=1):
    #     # Extract important entities
    #     print('Collect important entities')
    #     collect_start_time = time()
    #     important_ents_list = [
    #         LongDocPrompt.parse_entities(
    #             chat_response, 
    #             lambda x: self.retriever.embed_paragraphs(x, True), 
    #             self.retriever.syn_similarity)
    #         for chat_response in self.llm_server([LongDocPrompt.list_entity(p) for p in paragraphs], 5, 0.7)
    #     ]
    #     print('Collect important entities done', time() - collect_start_time)
        
    #     results = [ChunkInfo(cur_pid, paragraph, important_ents=important_ents) for cur_pid, (paragraph, important_ents) in enumerate(zip(paragraphs, important_ents_list))]
    #     raw = {}
    #     for nbr_dist in range(nbr_max_dist + 1):
    #         relation_description_prompts = [
    #             LongDocPrompt.pairwise_relation_description(
    #                 ' '.join(paragraphs[cur_pid - nbr_dist : cur_pid + 1]), 
    #                 results[cur_pid - nbr_dist].important_ents, 
    #                 results[cur_pid].important_ents
    #             )
    #             for cur_pid in range(nbr_dist, len(paragraphs))
    #         ]
    #         temp_raw = []
    #         for cur_pid, relation_description in zip(range(nbr_dist, len(paragraphs)), self.llm_server(relation_description_prompts)):
    #             temp_raw.append((cur_pid, relation_description[0]))
    #             relation_descriptions = LongDocPrompt.parse_pairwise_relation_description(relation_description[0], results[cur_pid - nbr_dist].important_ents, results[cur_pid].important_ents)
    #             if nbr_dist == 0:
    #                 results[cur_pid].relation_descriptions = relation_descriptions
    #             else:
    #                 results[cur_pid].prev_relation_descriptions[-nbr_dist] = relation_descriptions
    #         raw[nbr_dist] = temp_raw
    #     return results, raw
    
    # def lossless_index(self, pages:List[str], rewrite_chunk_num:int=5, prev_chunk_num:int=5, post_chunk_num:int=5, target:str='relation'):
    #     missed_chunk_ids:List[List[int]] = []
    #     batch_start_ends = []
    #     summaries:List[dict] = []
    #     for batch_start in range(0, len(pages), rewrite_chunk_num):
    #         prev_start = max(batch_start - prev_chunk_num, 0)
    #         batch_end = min(batch_start + rewrite_chunk_num, len(pages))
    #         post_end = min(batch_end + post_chunk_num, len(pages))
    #         chunk_start = batch_start - prev_start
    #         chunk_end = batch_end - prev_start
    #         chunk_ids = list(range(chunk_start, chunk_end))
    #         missed_chunk_ids.append(chunk_ids)
    #         batch_start_ends.append((prev_start, post_end))
    #         summaries.append({})

    #     while any(missed_chunk_ids):
    #         temp_prompts = []
    #         temp_summaries:List[dict] = []
    #         temp_chunk_ids:List[list] = []
    #         for chunk_ids, (batch_start, batch_end), summary in zip(missed_chunk_ids, batch_start_ends, summaries):
    #             if chunk_ids:
    #                 chunk_wise_func = LongDocPrompt.chunk_wise_rewrite if target == 'relation' else LongDocPrompt.chunk_wise_entity_extraction
    #                 temp_prompts.append(chunk_wise_func(pages[batch_start:batch_end], chunk_ids))
    #                 temp_summaries.append(summary)
    #                 temp_chunk_ids.append(chunk_ids)

    #         print(len(temp_prompts))
    #         responses = self.llm_server(temp_prompts)

    #         for response, chunk_ids, summary in zip(responses, temp_chunk_ids, temp_summaries):
    #             chunk_wise_parse_func = LongDocPrompt.parse_chunk_wise_rewrite if target == 'relation' else LongDocPrompt.parse_chunk_wise_entity_extraction
    #             new_summaries = chunk_wise_parse_func(response[0])
    #             summarized_chunk_ids = [chunk_id for chunk_id in chunk_ids if chunk_id in new_summaries]
    #             for chunk_id in summarized_chunk_ids:
    #                 summary[chunk_id] = new_summaries[chunk_id]
    #                 chunk_ids.remove(chunk_id)
        
    #     all_summary:List[str] = []
    #     for summary in summaries:
    #         cid_sum_pairs = list(summary.items())
    #         cid_sum_pairs.sort(key=lambda x: x[0])
    #         all_summary.extend([s for _, s in cid_sum_pairs])
        
    #     return all_summary
    
    def build_relation_graph(self, notes:List[ChunkInfo]):
        relation_graph = nx.Graph()
        for cur_pid, chunk_info in enumerate(notes):
            for related_ents, _ in chunk_info.relation_descriptions:
                for ent1, ent2 in itertools.combinations(related_ents, 2):
                    if not relation_graph.has_edge(ent1, ent2):
                        relation_graph.add_edge(ent1, ent2, pids=[])
                    temp_pids:List[int] = relation_graph.get_edge_data(ent1, ent2)['pids']
                    if cur_pid not in temp_pids:
                        temp_pids.append(cur_pid)
        return relation_graph
    
