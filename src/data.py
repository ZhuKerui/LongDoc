
from nltk import sent_tokenize
from datasets import load_dataset

from .models import *
from .base_utils import *
from .prompt import GeneralPrompt

class Dataset:
    def __init__(self, dataset_name:str, llm:str | LLM="mistralai/Mistral-7B-Instruct-v0.2", split:str='train') -> None:
        self.dataset_name = dataset_name
        self.answer_format = None
        self.question_type = None
        self.data = []
        if llm:
            self.llm_server = LLMServer(llm)
        self.split = split
        self.load_split()
        self.data_dir = Path(os.path.join(self.dataset_name, self.split))
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
    def load_split(self):
        raise NotImplementedError
    
    @staticmethod
    def parse_pause_point(text:str):
        text = text.strip("Break point: ")
        if text[0] != '<':
            return None
        for i, c in enumerate(text):
            if c == '>':
                if text[1:i].isnumeric():
                    return int(text[1:i])
                else:
                    return None
        return None
    
    def paragraph_parser(self, article: str) -> List[str]:
        """Parse Gutenberg articles."""
        lines = []
        previous_line = None
        for i, line in enumerate(article.split('\n')):
            line = line.strip()
            original_line = line
            if line == '':
                if previous_line == '':
                    line = '\n'
                else:
                    previous_line = original_line
                    continue
            previous_line = original_line
            lines.append(line)
        return (' '.join(lines)).split('\n')
    
    @staticmethod
    def count_words(text:str):
        """Simple word counting."""
        return len(text.split())
    
    
    def get_questions_and_answers(self, sample:dict) -> Tuple[List[str], List[str]]:
        raise NotImplementedError
    
    def get_article(self, sample:dict) -> str:
        raise NotImplementedError


    def pagination(
        self,
        article:str,
        word_limit=600,
        start_threshold=280,
        verbose=True,
        allow_fallback_to_last=True
    ):
        paragraphs = self.paragraph_parser(article)

        i = 0
        pages = []
        while i < len(paragraphs):
            preceding = "" if i == 0 else "...\n" + '\n'.join(pages[-1])
            passage = [paragraphs[i]]
            wcount = self.count_words(paragraphs[i])
            j = i + 1
            while wcount < word_limit and j < len(paragraphs):
                wcount += self.count_words(paragraphs[j])
                if wcount >= start_threshold:
                    passage.append(f"<{j}>")
                passage.append(paragraphs[j])
                j += 1
            passage.append(f"<{j}>")
            end_tag = "" if j == len(paragraphs) else paragraphs[j] + "\n..."

            pause_point = None
            if wcount < 350:
                pause_point = len(paragraphs)
            else:
                prompt = GeneralPrompt.pagination(preceding, '\n'.join(passage), end_tag)
                response = self.llm_server(prompt=prompt)[0].strip()
                pause_point = self.parse_pause_point(response)
                if pause_point and (pause_point <= i or pause_point > j):
                    print(f"prompt:\n{prompt},\nresponse:\n{response}\n")
                    print(f"i:{i} j:{j} pause_point:{pause_point}")
                    pause_point = None
                if pause_point is None:
                    if allow_fallback_to_last:
                        pause_point = j
                    else:
                        raise ValueError(f"prompt:\n{prompt},\nresponse:\n{response}\n")

            page = paragraphs[i:pause_point]
            pages.append(page)
            if verbose:
                print(f"Paragraph {i}-{pause_point-1}", page)
            i = pause_point
        if verbose:
            print(f"[Pagination] Done with {len(pages)} pages")
        return pages
    
    def eval(self, gen:str, answer:str):
        raise NotImplementedError
    
    def load_and_eval_result(self, task_start:int, task_end:int, prefix:dict={}):
        results = defaultdict(list)
        for r_tool in ['index', 'dpr', 'gist']:
            for task_i in range(task_start, task_end):
                generation_file = os.path.join(self.data_dir, f'generation_{prefix.get(r_tool, "")}{r_tool}_s_{task_i}.jsonl')
                if os.path.exists(generation_file):
                    _, answers = self.get_questions_and_answers(self.data[task_i])
                    temp_results = []
                    temp_dict = {}
                    steps = []
                    for line in read_jsonline(generation_file):
                        if line[0] == 'query':
                            temp_dict[line[0]] = line[1]
                        elif line[0] == 'generation':
                            temp_dict['steps'] = steps
                            temp_dict[line[0]] = line[1]
                            temp_dict['q_i'] = len(temp_results)
                            temp_dict['gold'] = answers[temp_dict['q_i']]
                            temp_dict['task_i'] = task_i
                            pred, score = self.eval(line[1], temp_dict['gold'])
                            temp_dict['predict'] = pred
                            temp_dict['acc'] = score
                            temp_results.append(temp_dict)
                            temp_dict = {}
                            steps = []
                        else:
                            steps.append(line)
                    results[r_tool].extend(temp_results)
            if r_tool in results:
                print(r_tool, sum([r['acc'] for r in results[r_tool]]) * 1. / len(results[r_tool]))
        return results
    

class QualityDataset(Dataset):
    
    # Fields that are straight text copies from raw example to processed example.
    _ONE2ONE_FIELDS = (
        'article',
        'article_id',
        'set_unique_id',
        'writer_id',
        'source',
        'title',
        'topic',
        'url',
        'writer_id',
        'author',
    )
    
    bracketed_lowercase_letters_set = set(
        [f"({l})" for l in string.ascii_lowercase]
    )  # {"(a)", ...}
    bracketed_uppercase_letters_set = set(
        [f"({l.upper()})" for l in string.ascii_lowercase]
    )  # {"(A)", ...}

    choices = ['(A)', '(B)', '(C)', '(D)']
    
    def __init__(self, llm: str | LLM = "mistralai/Mistral-7B-Instruct-v0.2", split: str = 'train') -> None:
        super().__init__('quality', llm, split)
        self.answer_format = '''If (A) is correct, answer with \"Answer: (A) ...\"\nIf (B) is correct, answer with \"Answer: (B) ...\"\nIf (C) is correct, answer with \"Answer: (C) ...\"\nIf (D) is correct, answer with \"Answer: (D) ...\"'''
        self.question_type = 'multiple choice question'
        
    def load_split(self):
        self.data = []
        with open(f'../../data/QuALITY/QuALITY.v1.0.1.htmlstripped.{self.split}', 'r') as f:
            for line in f.readlines():
                j = json.loads(line)
                fields = {k: j[k] for k in self._ONE2ONE_FIELDS}
                fields.update({
                    'questions': [q['question'] for q in j['questions']],
                    'question_ids': [q['question_unique_id'] for q in j['questions']],
                    'difficults': [q['difficult'] for q in j['questions']],
                    'options': [q['options'] for q in j['questions']],
                })

                fields.update({
                    'gold_labels': [q['gold_label'] for q in j['questions']],
                    'writer_labels': [q['writer_label'] for q in j['questions']],
                })

                self.data.append(fields)
                
    def get_index_from_symbol(self, answer):
        """Get the index from the letter symbols A, B, C, D, to extract answer texts.

        Args:
            answer (str): the string of answer like "(B)".

        Returns:
            index (int): how far the given choice is from "a", like 1 for answer "(B)".
        """
        answer = str(answer).lower()
        # extract the choice letter from within bracket
        if answer in self.bracketed_lowercase_letters_set:
            answer = re.findall(r"\(.*?\)", answer)[0][1]
        index = ord(answer) - ord("a")
        return index

    def get_questions_and_answers(self, sample: dict):
        return ['\n'.join([question] + [f"{ol} {o}" for ol, o in zip(self.choices, option)]) for option, question in zip(sample['options'], sample['questions'])], sample['gold_labels']
        
    def get_article(self, sample: dict) -> str:
        return sample['article']
    
    def eval(self, gen: str, answer: str):
        gen = gen.strip()
        if gen.endswith('</s>'):
            gen = gen[:-4]
        if 'answer: ' in gen.lower():
            gen = gen.lower().split('answer: ', 1)[-1].split()[0].strip('().,:').upper()
        elif 'answer is' in gen.lower():
            gen = gen.lower().split('answer is', 1)[-1].split()[0].strip('().,:').upper()
        
        if gen in 'ABCD':
            predict = 'ABCD'.index(gen) + 1
        else:
            predict = 0
            # print(gen)
        return predict, predict == answer
    

class NarrativeQADataset(Dataset):
    
    def __init__(self, llm: str | LLM = "mistralai/Mistral-7B-Instruct-v0.2", split: str = 'train') -> None:
        super().__init__('narrativeqa', llm, split)
        self.answer_format = '''Generate your answer and explanation using the following format:\n"Answer: your answer to the question ...\nExplanation: your explanation or reasoning ...". Your answer should be brief and specific.'''
        self.question_type = 'question'
        
    def load_split(self):
        self.data = list(load_dataset('THUDM/LongBench', 'narrativeqa', split='test'))
        
    def paragraph_parser(self, article: str) -> List[str]:
        return article.split('\n\n')
        
    def get_questions_and_answers(self, sample: dict) -> Tuple[List[str], List[str]]:
        return [sample['input']], [sample['answers']]
    
    def get_article(self, sample: dict) -> str:
        return sample['context']
    
    def eval(self, gen: str, answers: List[str]):
        gen = gen.strip().lower()
        f1 = 0.
        pred = ''
        if gen.startswith('answer: '):
            gen = gen.split('answer: ', 1)[-1]
            gen = gen.split('explanation:')[0].strip()
            pred = gen
            gen = normalize_answer(gen).split()
            
            for answer in answers:
                answer = normalize_answer(answer).split()
                common = Counter(gen) & Counter(answer)
                num_same = sum(common.values())
                if num_same == 0:
                    temp_f1 = 0
                else:
                    precision = 1.0 * num_same / len(gen)
                    recall = 1.0 * num_same / len(answer)
                    temp_f1 = (2 * precision * recall) / (precision + recall)
                if temp_f1 > f1:
                    f1 = temp_f1
        return pred, f1


class MuSiQueDataset(Dataset):
    
    def __init__(self, llm: str | LLM = "mistralai/Mistral-7B-Instruct-v0.2", split: str = 'train') -> None:
        super().__init__('musique', llm, split)
        self.answer_format = '''Generate your answer and explanation using the following format:\n"Answer: your answer to the question ...\nExplanation: your explanation or reasoning ...". Your answer should be brief and specific.'''
        self.question_type = 'question'
        
    def load_split(self):
        self.data = list(load_dataset('THUDM/LongBench', 'musique', split='test'))
        
    def paragraph_parser(self, article: str) -> List[str]:
        return article.split('\n\n')
        
    def get_questions_and_answers(self, sample: dict) -> Tuple[List[str], List[str]]:
        return [sample['input']], [sample['answers']]
    
    def get_article(self, sample: dict) -> str:
        return sample['context']
    
    def eval(self, gen: str, answers: List[str]):
        gen = gen.strip().lower()
        f1 = 0.
        pred = ''
        if gen.startswith('answer: '):
            gen = gen.split('answer: ', 1)[-1]
            gen = gen.split('explanation:')[0].strip()
            pred = gen
            gen = normalize_answer(gen).split()
            
            for answer in answers:
                answer = normalize_answer(answer).split()
                common = Counter(gen) & Counter(answer)
                num_same = sum(common.values())
                if num_same == 0:
                    temp_f1 = 0
                else:
                    precision = 1.0 * num_same / len(gen)
                    recall = 1.0 * num_same / len(answer)
                    temp_f1 = (2 * precision * recall) / (precision + recall)
                if temp_f1 > f1:
                    f1 = temp_f1
        return pred, f1
    
        
class LooGlEDataset(Dataset):
    def __init__(self, llm_name: str = "mistralai/Mistral-7B-Instruct-v0.2", split: str = 'test') -> None:
        super().__init__('loogle', llm_name, split)
        self.answer_format = '''Generate your answer and explanation using the following format:\n"Answer: your answer to the question ...\nExplanation: your explanation or reasoning ...". Your answer should be brief and specific.'''
        self.question_type = 'question'
        
    def load_split(self):
        self.data = [self.parse_qa_pairs(data) for data in load_dataset('bigainlco/LooGLE', 'longdep_qa', split='test')]

    def parse_qa_pairs(self, data:dict):
        qa_pairs:str = data['qa_pairs']
        pairs = []
        for qa_pair in qa_pairs.strip('[{}]').split('}, {'):
            A = qa_pair.index("'A': ")
            Q = qa_pair.index("'Q': ")
            S = qa_pair.index("'S': ")
            T = qa_pair.index("'type': ")
            qs = qa_pair[Q + 5: A].strip(' ,"\'').replace("\\n", '\n')
            a_s = qa_pair[A + 5:T].strip(' ,"\'').replace("\\n", '\n')
            ts = qa_pair[T + 8:S].strip(' ,"\'').replace("\\n", '\n')
            ss = [s_.replace("\\n", '\n').replace('\\', '') for s_ in re.split('", "|", \'|\', "|\', \'', qa_pair[S + 5:].strip('[]\'"'))]
            pairs.append({'Q': qs, 'A': a_s, 'type': ts, 'S': ss})
        return {'input': data['input'], 'title': data['title'], 'qa_pairs': pairs}
    
    def paragraph_parser(self, article: str) -> List[str]:
        return sent_tokenize(article) if '\n' not in article else article.split('\n')
    
    def get_questions_and_answers(self, sample: dict) -> Tuple[List[str]]:
        return [qa_pair['Q'] for qa_pair in sample['qa_pairs']], [qa_pair['A'] for qa_pair in sample['qa_pairs']]
    
    def get_article(self, sample: dict) -> str:
        return sample['input']

