
from typing import cast
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer, GenerationConfig, BatchEncoding, pipeline
from openai import OpenAI
import concurrent.futures

from .base import *


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
            instructions = [''] * len(sentences)
        instructions = [self.gritlm_instruction(instruction) for instruction in instructions]
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
            last_hidden_states.extend([lhs[mask>0].cpu().to(torch.float32).numpy() for lhs, mask in zip(last_hidden_state, inputs['attention_mask'])])
            embeddings = self.pooling(last_hidden_state, inputs['attention_mask'], recast=recast)
            # Normalize can change the dtype (https://discuss.pytorch.org/t/tensor-in-float16-is-transformed-into-float32-after-torch-norm/110891)
            if self.normalized: 
                in_dtype = embeddings.dtype
                last_hidden_states = [lhs / torch.norm(emb).item() for lhs, emb in zip(last_hidden_states, embeddings)]
                embeddings = torch.nn.functional.normalize(embeddings, dim=-1).to(in_dtype)
            embeddings = cast(torch.Tensor, embeddings)
            if convert_to_tensor:
                all_embeddings.append(embeddings)
            else:
                # NumPy does not support bfloat16
                all_embeddings.append(embeddings.cpu().to(torch.float32).numpy())

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
        self.generator = pipeline("text-generation", llm_name, device_map=device_map, batch_size=batch_size)
        
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
                openai_api_base = "http://localhost:8000/v1"
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
    def __init__(self, embeddings:np.ndarray, last_hidden_states:List[np.ndarray]=None, retriever_input:BatchEncoding=None) -> None:
        self.embeddings = embeddings
        self.input_ids:np.ndarray
        self.attention_mask:np.ndarray
        if retriever_input:
            self.input_ids = retriever_input['input_ids'].cpu().numpy()
            self.attention_mask = retriever_input['attention_mask'].cpu().numpy()
        self.last_hidden_states = last_hidden_states
        
        
class Retriever:
    def __init__(self, retriever_name:str='facebook/contriever', device='cpu') -> None:
        self.retriever_tokenizer = AutoTokenizer.from_pretrained(retriever_name)
        self.retriever_model = AutoModel.from_pretrained(retriever_name)
        if device == 'cpu':
            self.device = None
        else:
            self.device = torch.device(device)
            self.retriever_model.cuda(device=self.device)
        
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
            last_hidden_states = [lhs[mask>0].cpu().numpy() for lhs, mask in zip(retriever_output[0], retriever_input['attention_mask'])]
        if normalize:
            norms = np.linalg.norm(paragraph_embeddings, axis=1)
            paragraph_embeddings = paragraph_embeddings / np.expand_dims(norms, axis=1)
            last_hidden_states = [lhs / n for lhs, n in zip(last_hidden_states, norms)]
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
