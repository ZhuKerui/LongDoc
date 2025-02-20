from .base import *

import pymupdf
import re
import networkx as nx
import itertools
import string
import copy
from pydantic import BaseModel
from time import sleep

import spacy
from spacy.tokens import Span, Doc
from spacy.lang.en import stop_words
import pytextrank
from pytextrank import Phrase
from fastcoref import spacy_component
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from openai import OpenAI

import dash
import dash_cytoscape as cyto
from dash import html

MARGIN_RATIO = 10
MIN_WIDTH_RATIO = 3
ABS_HEADER = 'abstract'
INTRO_HEADER = 'introduction'
REF_HEADER = 'references'
ACK_HEADER = 'acknowledgement'

PUNCTUATION_WITHOUT_HYPHEN = string.punctuation.replace('-', '')

COREF = 'coref'
SUBJ_OBJ = 'subj_obj'
ADJACENT = 'adjacent'
SHARED_TEXT = 'shared_text'

ADVCL, CCOMP, ACL, XCOMP, RELCL, PCOMP = 'advcl', 'ccomp', 'acl', 'xcomp', 'relcl', 'pcomp'
NOUN, PROPN, PRON = 'NOUN', 'PROPN', 'PRON'
NOUN_POS = {NOUN, PROPN, PRON}
ENT_POS = {NOUN, PROPN}
PUNCT, DET = 'PUNCT', 'DET'

class DocBlock(BaseModel):
    text: str
    i: int = None
    is_section_header: bool = False
    bbox: tuple[float, float, float, float] = None
    phrases_with_hyphen: set[str] = None
    potential_broken_phrases: set[str] = None
    lines: list[str] = None
    line_start_fonts: list[tuple[float, str]] = None
    
    
class OutlineSection:
    def __init__(self, header:str, level:int, section_id:int, blocks:list[DocBlock]=[]):
        self.header = header
        self.level = level
        self.section_id = section_id
        self.blocks = blocks
        self.merged_blocks:list[Document] = None
        self.next_sibling:OutlineSection = None
        self.prev_sibling:OutlineSection = None
        self.children:list[OutlineSection] = None
        self.parent:OutlineSection = None
        
        self.section_nlp_global:Span = None
        self.section_nlp_local:Doc = None
        
        self.prons:list[Span] = None
        self.pron_root2coref:dict[int, Span] = None
        
        
    @property
    def text(self):
        return PARAGRAPH_SEP.join([block.text for block in self.blocks])
    
    @property
    def text_wo_header(self):
        return PARAGRAPH_SEP.join([block.text for block in self.blocks[1:]])

    @property
    def complete_header(self):
        headers = [self.header]
        parent = self.parent
        while parent:
            headers.insert(0, parent.header)
            parent = parent.parent
        return '\n'.join(headers)
    

class DocManager:
    def __init__(self, tool_llm:str = GPT_MODEL_CHEAP, emb_model:str = DEFAULT_EMB_MODEL, word_vocab:set[str] = None):
        self.tool_llm = tool_llm
        self.nlp = spacy.load(SPACY_MODEL)
        self.nlp.add_pipe("positionrank")
        self.nlp.add_pipe("fastcoref")
        self.client = OpenAI(api_key=os.environ[OPENAI_API_KEY_VARIABLE])
        
        # Load Embeddings
        self.emb_model_name = emb_model
        self.model_kwargs = {'device': 'cpu'}
        self.encode_kwargs = {'normalize_embeddings': False}
        self.embedding = HuggingFaceEmbeddings(
            model_name=self.emb_model_name,
            model_kwargs=self.model_kwargs,
            encode_kwargs=self.encode_kwargs
        )
        
        self.word_vocab = word_vocab or set()
        
    def load_doc(self, doc_file:str = None, doc_strs:list[str] = None, outline:str = None, simple_load:bool = False):
        assert (doc_file is not None) ^ (doc_strs is not None), "Only doc_file or doc_strs should be not None"
        
        self.doc_vocab = copy.deepcopy(self.word_vocab)
        self.simple_load = simple_load
        
        if doc_file:
            self.is_from_pdf = True
            self.pdf_doc = pymupdf.open(doc_file)
            self.remove_tables()
            self.extract_blocks()
            outline = outline or '\n'.join(f"{'    ' * (level - 1)}{section_name}" for level, section_name, page in self.pdf_doc.get_toc())
        else:
            assert all('\n' not in doc_str for doc_str in doc_strs), "doc_strs should have no \n"
            self.is_from_pdf = False
            self.pdf_doc = None
            self.main_text_style = None
            self.blocks = [DocBlock(text=doc_str, bbox=(0., 0., 0., 0.), phrases_with_hyphen=set(), potential_broken_phrases=set(), lines=[], line_start_fonts=[]) for doc_str in doc_strs]
            outline = outline or ''
            
        if not outline:
            outline = self.generate_outline()
        
        re_gen_cnt = 0
        outline_pass, err_msg = self.parse_outline(outline)
        err_msg_cnt = Counter()
        while not outline_pass:
            re_gen_cnt += 1
            if re_gen_cnt > 5:
                raise Exception("Failed to regenerate the outline")
            err_msg_cnt[err_msg] += 1
            if err_msg_cnt.most_common()[0][1] > 2:
                print('Regeneration from beginning')
                sleep(10)
                outline = self.generate_outline()
                err_msg_cnt.clear()
            else:
                print(f'Regeneration attempt: {re_gen_cnt}\n\n{outline}\n\n{err_msg}')
                outline = self.correct_outline(outline, err_msg)
            outline_pass, err_msg = self.parse_outline(outline)
            
        self.full_outline = self.outline
        self.reduce_noise()
        self.index_blocks()
        if not self.simple_load:
            self.collect_keyphrases()
            self.build_dkg()

    @property
    def doc_str(self):
        return PARAGRAPH_SEP.join(block.text for block in self.blocks)
    
    @property
    def doc_str_wo_headers(self):
        return PARAGRAPH_SEP.join(block.text for block in self.blocks if not block.is_section_header)
    
    @property
    def outline_generation_prompt(self):
        return f"Paper:\n\n{self.doc_str}\n\nExtract the complete table of contents (bookmarks) from the above paper. Write one section a line and use the original section number if exists. You should include the {ABS_HEADER.capitalize()}, the {INTRO_HEADER.capitalize()} and the {REF_HEADER.capitalize()} section. Use indentation to show the hierarchy of the sections."
    @property
    def outline(self):
        return '\n'.join('    ' * section.level + section.header for section in self.sections)
    
    @property
    def sents(self):
        return [section.section_nlp_global[sent.start:sent.end] for section in self.sections if section.section_nlp_global for sent in section.section_nlp_local.sents]

    def generate_outline(self):
        print('Generating outline')
        chat_completion = self.client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": self.outline_generation_prompt,
                }
            ],
            model=self.tool_llm,
        )
        return chat_completion.choices[0].message.content
        
    def correct_outline(self, outline:str, err_msg:str):
        print('Correcting outline')
        chat_completion = self.client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": self.outline_generation_prompt,
                },{
                    "role": "assistant",
                    "content": outline
                },{
                    "role": "user",
                    "content": f'{err_msg}\n\nRegenerate the table of contents and fix the issue above.'
                }],
            model=self.tool_llm,
        )
        return chat_completion.choices[0].message.content
            
    # =========================== Doc Initialization Functions ===========================
    @staticmethod
    def strip_fixed_text(source_text:str, fixed_text:str, from_start:bool=True):
        fixed_text = ''.join(fixed_text.split()).lower()
        if not fixed_text or not source_text or len(fixed_text) > len(''.join(source_text.split())):
            return source_text
        
        source_text = source_text if from_start else source_text[::-1]
        fixed_text = fixed_text if from_start else fixed_text[::-1]
        
        fixed_char_idx = 0
        for source_char_idx, source_char in enumerate(source_text):
            if source_char.isspace():
                continue
            if source_char.lower() == fixed_text[fixed_char_idx]:
                fixed_char_idx += 1
                if fixed_char_idx == len(fixed_text):
                    source_text = source_text[source_char_idx+1:]
                    break
            else:
                break
        return source_text if from_start else source_text[::-1]
    
    @staticmethod
    def extract_fixed_text(source_text:str, fixed_text:str, from_start:bool=True):
        fixed_text = ''.join(fixed_text.split()).lower()
        if not fixed_text or not source_text or len(fixed_text) > len(''.join(source_text.split())):
            return ''
        
        source_text = source_text if from_start else source_text[::-1]
        fixed_text = fixed_text if from_start else fixed_text[::-1]
        extracted_text = ''
        
        fixed_char_idx = 0
        for source_char_idx, source_char in enumerate(source_text):
            if source_char.isspace():
                continue
            if source_char.lower() == fixed_text[fixed_char_idx]:
                fixed_char_idx += 1
                if fixed_char_idx == len(fixed_text):
                    extracted_text = source_text[:source_char_idx+1]
                    break
            else:
                break
        return extracted_text if from_start else extracted_text[::-1]
        
    def parse_outline(self, outline:str):
        # Remove generated prompt if any
        outline = outline.strip()
        if PARAGRAPH_SEP in outline:
            outline = [paragraph for paragraph in outline.split(PARAGRAPH_SEP) if '\n' in paragraph.strip()][0].strip()
            
        # Strip code block boundaries if any
        outline = outline.strip('`').strip()
        
        # Get indentation
        indentations = [line.index(line.strip()) for line in outline.splitlines() if line.strip()]
        min_indentation = min(indentations)
        indentations = [ind-min_indentation for ind in indentations]
        indentations_gt0 = [ind for ind in indentations if ind > 0]
        tab_width = min(indentations_gt0) if indentations_gt0 else 1
        
        # Parse outline
        sections = list[OutlineSection]()
        for line in outline.splitlines():
            section_header = line.strip().strip('-').strip()
            if not section_header:
                # Skip empty lines
                continue
            current_section = OutlineSection(section_header, (line.index(section_header)-min_indentation) // tab_width, len(sections))
            sections.append(current_section)

        meta_data = list[DocBlock]()
        blocks = copy.deepcopy(self.blocks)
        temp_content = list[DocBlock]()
        curr_sid, bid = -1, 0
        next_section: OutlineSection
        lower_next_header_wo_space: str
        lower_next_header_wo_space_num: str
                
        while bid < len(blocks):
            block = blocks[bid]
            
            if curr_sid + 1 < len(sections):
                lower_block_wo_space = ''.join(block.text.split()).lower()
                lower_block_wo_space_rstrip_period = lower_block_wo_space.rstrip('.')
                next_section = sections[curr_sid+1]
                if len(next_section.header.split()) > 1 and all(c.isdigit() or c == '.' for c in next_section.header.split()[0]):
                    next_header_wo_num = next_section.header.split(maxsplit=1)[1]
                else:
                    next_header_wo_num = next_section.header
                lower_next_header_wo_space = ''.join(next_section.header.split()).lower()
                lower_next_header_wo_space_num = ''.join(next_header_wo_num.split()).lower()
                
                startswith_section_header = False
                if not self.is_from_pdf:
                    startswith_section_header = next_section.header == block.text
                elif lower_block_wo_space.startswith(lower_next_header_wo_space):
                    startswith_section_header = True
                elif lower_block_wo_space.startswith(lower_next_header_wo_space_num):
                    next_section.header = next_section.header.split(maxsplit=1)[1]
                    lower_next_header_wo_space = lower_next_header_wo_space_num
                    startswith_section_header = True
                elif lower_block_wo_space.endswith(lower_next_header_wo_space):
                    new_block = copy.deepcopy(block)
                    block.text = DocManager.strip_fixed_text(block.text, lower_next_header_wo_space, from_start=False)
                    new_block.text = next_section.header
                    blocks.insert(bid+1, new_block)
                elif lower_next_header_wo_space in lower_block_wo_space or lower_next_header_wo_space_num in lower_block_wo_space:
                    for lid in range(len(block.lines)):
                        sub_block = ' '.join(block.lines[lid:])
                        sub_block_start_font = block.line_start_fonts[lid]
                        sub_block_wo_space = ''.join(sub_block.split())
                        lower_sub_block_wo_space = sub_block_wo_space.lower()
                        sub_block_startswith_number = len(sub_block.split()) > 1 and all(c.isdigit() or c in '().' for c in sub_block.split(maxsplit=1)[0])
                        if sub_block_startswith_number:
                            sub_block_wo_space_num = ''.join(sub_block.split()[1:])
                        else:
                            sub_block_wo_space_num = sub_block_wo_space
                        lower_sub_block_wo_space_num = sub_block_wo_space_num.lower()
                        
                        if (lower_sub_block_wo_space.startswith(lower_next_header_wo_space) or lower_sub_block_wo_space.startswith(lower_next_header_wo_space_num) or lower_sub_block_wo_space_num.startswith(lower_next_header_wo_space_num)) and (sub_block_start_font != self.main_text_style or sub_block_startswith_number) and (lid == 0 or block.lines[lid-1][-1] in PUNCTUATION_WITHOUT_HYPHEN) and sub_block_wo_space_num[0].isupper():
                            if lid > 0:
                                new_block = copy.deepcopy(block)
                                block.text = ' '.join(block.lines[:lid])
                                block.lines = block.lines[:lid]
                                block.line_start_fonts = block.line_start_fonts[:lid]
                                new_block.text = sub_block
                                new_block.lines = new_block.lines[lid:]
                                new_block.line_start_fonts = new_block.line_start_fonts[lid:]
                                blocks.insert(bid+1, new_block)
                            else:
                                prefix_wo_space = lower_sub_block_wo_space[:lower_sub_block_wo_space.index(lower_next_header_wo_space_num)]
                                prefix = DocManager.extract_fixed_text(sub_block, prefix_wo_space).strip()
                                next_section.header = prefix + ' ' + next_section.header.split(maxsplit=1)[1]
                                lower_next_header_wo_space = ''.join(next_section.header.split()).lower()
                                startswith_section_header = True
                            break
                    
                if startswith_section_header:
                    # The next section is matched with current block
                    if curr_sid >= 0:
                        # Fill the blocks for the previous section
                        sections[curr_sid].blocks = temp_content
                        temp_content = []
                        # print(sections[curr_sid].header)
                    curr_sid += 1
                    
                    if not (lower_block_wo_space == lower_next_header_wo_space or lower_block_wo_space_rstrip_period == lower_next_header_wo_space):
                        new_block = copy.deepcopy(block)
                        accumulated_len = 0
                        for lid, line in enumerate(block.lines):
                            accumulated_len += len(''.join(line.split()))
                            if accumulated_len > len(lower_next_header_wo_space):
                                new_line = DocManager.strip_fixed_text(' '.join(block.lines[:lid+1]), lower_next_header_wo_space)
                                block.lines = block.lines[:lid] + [DocManager.strip_fixed_text(line, new_line, from_start=False).strip()]
                                block.line_start_fonts = block.line_start_fonts[:lid+1]
                                block.text = sections[curr_sid].header
                                if new_line[0] in string.punctuation:
                                    new_line = new_line[1:]
                                new_line = new_line.strip()
                                new_block.lines = [new_line] + new_block.lines[lid+1:]
                                new_block.line_start_fonts = [self.main_text_style] + new_block.line_start_fonts[lid+1:]
                                new_block.text = ' '.join(new_block.lines)
                                blocks.insert(bid+1, new_block)
                                break
                        
                    block.is_section_header = True
                
            if curr_sid < 0:
                meta_data.append(block)
            else:
                temp_content.append(block)
            
            bid += 1
            
        sections[curr_sid].blocks = temp_content
        
        if curr_sid + 1 < len(sections):
            if any(DocManager.block_endswith(block.text, lower_next_header_wo_space_num) or DocManager.block_startswith(block.text, lower_next_header_wo_space_num) or DocManager.block_startswith(block.text, lower_next_header_wo_space) for section in sections[:curr_sid+1] for block in section.blocks):
                return False, f"The following section name is placed in the wrong position in the table of content. Please check the section names before this section name and ensure that all sections exist and are listed in the correct order as they appear in the paper.\n\n{next_section.header}"
            
            # return False, f"The following section name does not appear in the paper. Please review the paper and ensure that all listed section names are accurately included in the text.\n\n{next_section.header}"
            print(f'Automatically dropping {next_section.header}')
            outline = '\n'.join('    ' * section.level + section.header for section in sections)
            outline = outline.replace(next_section.header, '')
            outline = '\n'.join([line for line in outline.splitlines() if line.strip()])
            return self.parse_outline(outline)
        
        for header in [ABS_HEADER, REF_HEADER, INTRO_HEADER]:
            if not any(section.header.lower().endswith(header) for section in sections) and any(DocManager.block_endswith(block.text, header) or DocManager.block_startswith(block.text, header) for block in blocks):
                return False, f"The {header.capitalize()} section is missing. You should find the {header.capitalize()} section and insert the {header.capitalize()} section header in its corresponding position in the table of content."
        
        sid = 0
        while sid < len(sections):
            section = sections[sid]
            section.section_id = sid
            if self.is_from_pdf:
                new_sub_section_ids = [bid for bid, block in enumerate(section.blocks[:-1]) if block.bbox == section.blocks[bid+1].bbox and not block.is_section_header]
                if new_sub_section_ids:
                    for sub_section_id in new_sub_section_ids:
                        section.blocks[sub_section_id].is_section_header = True
                    section.blocks, sub_section_blocks = section.blocks[:new_sub_section_ids[0]], section.blocks[new_sub_section_ids[0]:]
                    last_new_sub_section:OutlineSection = None
                    for sub_section_block in sub_section_blocks:
                        if sub_section_block.is_section_header:
                            sid += 1
                            new_sub_section = OutlineSection(header=sub_section_block.text, level=section.level+1, section_id=sid, blocks=[sub_section_block])
                            sections.insert(sid, new_sub_section)
                            last_new_sub_section = new_sub_section
                        else:
                            last_new_sub_section.blocks.append(sub_section_block)
            sid += 1
    
        self.blocks = blocks
        self.sections = sections
        self.meta_data = meta_data
        
        return True, ''
    
    def reduce_noise(self):
        if not self.is_from_pdf:
            return
        xmin, _, xmax, _ = self.blocks[0].bbox
        xs = []
        section:OutlineSection
        for section in self.sections:
            if section.header.lower().endswith(REF_HEADER) or section.header.lower().endswith(ACK_HEADER) or section.header.lower().endswith(ACK_HEADER+'s'):
                break
            for block in section.blocks:
                if not block.is_section_header:
                    x0, y0, x1, y1 = block.bbox
                    xmin = min(x0, xmin)
                    xmax = max(x1, xmax)
                    xs.append(x1 - x0)
        max_width = xmax - xmin
        margin = int(max_width / MARGIN_RATIO)
        min_req_width = max_width / MIN_WIDTH_RATIO
        normal_width = Counter([int(x // margin * margin) for x in xs if x > min_req_width]).most_common(1)[0][0]
        
        new_blocks = list[DocBlock]()
        new_sections = list[OutlineSection]()
        last_block:DocBlock = None
        for section in self.sections:
            if section.header.lower().endswith(REF_HEADER) or section.header.lower().endswith(ACK_HEADER) or section.header.lower().endswith(ACK_HEADER+'s'):
                break
            new_section_blocks = list[DocBlock]()
            for block in section.blocks:
                x0, y0, x1, y1 = block.bbox
                width = x1 - x0
                if block.is_section_header or (width > min_req_width and width < normal_width + margin and re.search(r'https?://[^\s/$.?#].[^\s]*', block.text) is None and re.match(r'^Table \d+: ', block.text) is None and re.match(r'^Figure \d+: ', block.text) is None):
                    potential_broken_phrases:set[str] = block.potential_broken_phrases
                    if last_block is not None and not last_block.is_section_header and not block.is_section_header and (last_block.text[-1].isalpha() or last_block.text[-1] not in {'.', '!', '?', ':'}):
                        if last_block.text[-1] == '-':
                            last_block.text = ''.join([last_block.text, block.text])
                            potential_broken_phrases.add(last_block.text.split()[-1] + block.text.split()[0])
                        else:
                            last_block.text = ' '.join([last_block.text, block.text])
                    else:
                        new_section_blocks.append(block)
                        new_blocks.append(block)
                        last_block = block
                        
                    for potential_broken_phrase in potential_broken_phrases:
                        if all(word.lower() in self.doc_vocab for word in potential_broken_phrase.split('-')):
                            continue
                        tokens = potential_broken_phrase.split('-')
                        token_validity = [token in self.doc_vocab for token in tokens]
                        invalid_token_idx = token_validity.index(False)
                        if invalid_token_idx < len(tokens) - 1:
                            if f'{tokens[invalid_token_idx]}{tokens[invalid_token_idx+1]}'.lower() in self.doc_vocab:
                                fixed_phrase = potential_broken_phrase.replace(f'{tokens[invalid_token_idx]}-{tokens[invalid_token_idx+1]}', f'{tokens[invalid_token_idx]}{tokens[invalid_token_idx+1]}')
                                last_block.text = last_block.text.replace(potential_broken_phrase, fixed_phrase)
                                continue
                        if invalid_token_idx > 0:
                            if f'{tokens[invalid_token_idx-1]}{tokens[invalid_token_idx]}'.lower() in self.doc_vocab:
                                fixed_phrase = potential_broken_phrase.replace(f'{tokens[invalid_token_idx-1]}-{tokens[invalid_token_idx]}', f'{tokens[invalid_token_idx-1]}{tokens[invalid_token_idx]}')
                                last_block.text = last_block.text.replace(potential_broken_phrase, fixed_phrase)
                                continue
                        
            # Remove citations from the block text
            for block in new_section_blocks:
                block.text = DocManager.remove_citations(block.text)
                
            section.blocks = new_section_blocks
            new_sections.append(section)
            
        for bid, block in enumerate(new_blocks):
            block.i = bid
        
        for sid, section in enumerate(new_sections):
            section.children = []
            if sid:
                last_section = new_sections[sid-1]
                while section.level < last_section.level:
                    last_section = last_section.parent
                
                if section.level == last_section.level:
                    last_section.next_sibling = section
                    section.prev_sibling = last_section
                    if last_section.parent is not None:
                        last_section.parent.children.append(section)
                        section.parent = last_section.parent
                else:
                    last_section.children.append(section)
                    section.parent = last_section
            
        self.blocks = new_blocks
        self.sections = new_sections
        
    def index_blocks(self):
        # Finalize the sections and blocks
        doc_spacy = self.nlp(self.doc_str_wo_headers, disable=['fastcoref'] if not self.simple_load else ['fastcoref', "lemmatizer", "ner", "positionrank"])
        tid2section_id = np.ones(len(doc_spacy), dtype=int) * -1
        tid2sent_id = np.ones(len(doc_spacy), dtype=int) * -1
        
        section_contents = list[str]()
        section_with_content = list[OutlineSection]()
        for section in self.sections:
            section_content = PARAGRAPH_SEP.join(block.text for block in section.blocks if not block.is_section_header)
            if section_content:
                section_contents.append(section_content)
                section_with_content.append(section)
        
        section_nlp_global:Span
        for section, section_nlp_global in zip(section_with_content, DocManager.iterate_find(doc_spacy, section_contents)):
            section.section_nlp_global = section_nlp_global
            section.section_nlp_local = self.nlp(section_nlp_global.text, disable=["lemmatizer", "ner", "positionrank"] if not self.simple_load else ['fastcoref', "lemmatizer", "ner", "positionrank"])
            tid2section_id[section.section_nlp_global.start:section.section_nlp_global.end] = section.section_id
            
        for sent_id, sent in enumerate(doc_spacy.sents):
            tid2sent_id[sent.start:sent.end] = sent_id

        self.doc_spacy = doc_spacy
        self.tid2section_id = tid2section_id
        self.tid2sent_id = tid2sent_id
        
    def collect_keyphrases(self):
        def split_noun_phrases(noun_phrase:Span):
            
            phrases = list[Span]()
            # Split the phrase into sub-phrases if PARAGRAPH_SEP is found
            for sub_chunk in DocManager.iterate_find(noun_phrase, [sub_noun_phrase.strip() for sub_noun_phrase in noun_phrase.text.split(PARAGRAPH_SEP)]):
                if all(len(token.text) < 2 for token in sub_chunk):
                    continue
                
                if sub_chunk.root.pos_ in ENT_POS or sub_chunk.root.text.lower() == 'we':
                    phrases.append(sub_chunk)
            return phrases
        
        def get_tid2phrase_id(span_ranges:list[tuple[int, int]]):
            tid2phrase_id = np.ones(len(self.doc_spacy), dtype=int) * -1
            for pid, span_range in enumerate(span_ranges):
                tid2phrase_id[span_range[0]:span_range[1]] = pid
            return tid2phrase_id
        
        def merge_overlapping_spans(spans:list[tuple[int, int]]):
            span_ranges = list[tuple[int, int]]()
            span_range = spans[0]
            for span in spans:
                if span[0] > span_range[1] - 1:
                    span_ranges.append(span_range)
                    span_range = span
                else:
                    span_range = span_range[0], max(span[1], span_range[1])
            span_ranges.append(span_range)
            return span_ranges
                    
        ph:Phrase
        ph_strs = set[str]()
        for ph in self.doc_spacy._.phrases:
            ph_strs.update(sub_chunk.text for sub_chunk in split_noun_phrases(ph.chunks[0]))
                    
        spans = list[Span]()
        for ph_str in ph_strs:
            for temp_ph in DocManager.iterate_find(self.doc_spacy, [ph_str]*self.doc_spacy.text.count(ph_str)):
                if temp_ph:
                    spans.append(temp_ph)
                    
        for noun_phrase in self.doc_spacy.noun_chunks:
            spans.extend(split_noun_phrases(noun_phrase))
        
        # Extend the spans to full noun phrases
        extended_spans = set[tuple[int, int]]()
        for span in spans:
            section = self.get_section_for_span(span)
            span_local = section.section_nlp_local[span.start-section.section_nlp_global.start:span.end-section.section_nlp_global.start]
            span_local = DocManager.get_full_noun_phrase(span_local)
            extended_spans.add((span_local.start+section.section_nlp_global.start, span_local.end+section.section_nlp_global.start))
        
        # Merge the overlapping spans
        extended_spans = list(extended_spans)
        extended_spans.sort()
        new_span_ranges = merge_overlapping_spans(extended_spans)
        temp_tid2phrase_id = get_tid2phrase_id(new_span_ranges)
        
        # Merge phrases based on prep_pobj relation or subj subtree
        for phrase_id, span_range in enumerate(new_span_ranges):
            curr_section = self.get_section_for_span(self.doc_spacy[span_range[0]:span_range[1]])
            phrase_local = curr_section.section_nlp_local[span_range[0]-curr_section.section_nlp_global.start: span_range[1]-curr_section.section_nlp_global.start]
            if phrase_local.root.dep_ == 'pobj' and phrase_local.root.head.dep_ == 'prep' and temp_tid2phrase_id[phrase_local.root.head.head.i+curr_section.section_nlp_global.start] >= 0 and phrase_local.root.head.head.i < phrase_local.root.i:
                new_span_ranges[phrase_id] = new_span_ranges[temp_tid2phrase_id[phrase_local.root.head.head.i+curr_section.section_nlp_global.start]][0], span_range[1]
            elif 'subj' in phrase_local.root.dep_:
                subj_tokens = list(phrase_local.root.subtree)
                new_span_ranges[phrase_id] = subj_tokens[0].i+curr_section.section_nlp_global.start, subj_tokens[-1].i+curr_section.section_nlp_global.start+1
            
        new_span_ranges.sort()
        new_span_ranges = merge_overlapping_spans(new_span_ranges)
        self.phrases = [self.doc_spacy[span_range[0]:span_range[1]] for span_range in new_span_ranges]
        self.tid2phrase_id = get_tid2phrase_id(new_span_ranges)
                    
    def build_dkg(self):
        dkg = nx.MultiDiGraph()
        
        for section in self.sections:
            if section.section_nlp_local:
                section.prons = []
                section.pron_root2coref = {}
                for coref_cluster in section.section_nlp_local._.coref_clusters:
                    last_chunk_id = -1
                    last_coref:Span = None
                    for coref_mention in coref_cluster:
                        mention = section.section_nlp_local.char_span(coref_mention[0], coref_mention[1])
                        curr_phrase_id = self.tid2phrase_id[mention.root.i+section.section_nlp_global.start]
                        if curr_phrase_id >= 0:
                            # If the mention is a noun phrase
                            if last_chunk_id >= 0 and curr_phrase_id != last_chunk_id:
                                # dkg.add_edges_from([(last_chunk_id, curr_phrase_id, COREF), (curr_phrase_id, last_chunk_id, COREF)], weight=0)
                                dkg.add_edge(last_chunk_id, curr_phrase_id, COREF, weight=0)
                            last_chunk_id = curr_phrase_id
                            last_coref = mention
                            
                        elif mention.root.pos_ in {'VERB', 'AUX'}:
                            last_coref = mention
                        else:
                            # If the mention is a pronoun
                            section.prons.append(mention)
                            if last_coref is not None:
                                section.pron_root2coref[mention.root.i] = last_coref
        
        CLAUSE_DEPS = {ADVCL, CCOMP, ACL, XCOMP, RELCL, PCOMP}
        
        # xcomp: open clausal complement, subject is not in the clause by definition
        # relcl: relative clause modifier, subject could be the subject in the clause or the noun phrase that the clause modifies
        CLAUSE_DEPS_WO_CLEAR_SUBJ = {XCOMP, RELCL}
        
        # advcl: adverbial clause modifier, modify the predicate controlled by the subject, can take the subject of the head clause as the subject
        # ccomp: has an overt subject or no obligatory control, should not transist the subject of the head clause as the subject
        # acl: clausal modifier of noun (adjectival clause), should take the head of the clause as the subject, not the subject of the head clause
        # xcomp: open clausal complement, should take the subject of the head clause as the subject
        # relcl: relative clause modifier, the subject is not the subject of the head clause
        # pcomp: complement of preposition, usually modify the predicate controlled by the subject, can take the subject of the head clause as the subject
        # CLAUSE_DEPS_TRANSIST_HEAD_SUBJ = {ADVCL, XCOMP, PCOMP}
        CLAUSE_DEPS_TRANSIST_HEAD_SUBJ = {ADVCL, XCOMP, PCOMP, CCOMP}
        CLAUSE_DEPS_USE_HEAD_SUBJ = CLAUSE_DEPS_TRANSIST_HEAD_SUBJ | {CCOMP}
        
        def add_subj_obj_edges(dkg:nx.MultiDiGraph, subjs:list[tuple[int, Span]], objs:list[tuple[int, Span]], sent_dep_tree:nx.Graph, section:OutlineSection):
            for (subj, subj_root), (obj, obj_root) in itertools.product(subjs, objs):
                try:
                    if subj_root.sent.start != obj_root.sent.start:
                        continue
                    path = tuple([path_id + section.section_nlp_global.start for path_id in nx.shortest_path(sent_dep_tree, source=subj_root.root.i, target=obj_root.root.i)[1:-1]])
                except:
                    # print('')
                    # raise ValueError('No path found')
                    continue
                if not dkg.has_edge(subj, obj, key=SUBJ_OBJ):
                    dkg.add_edge(subj, obj, key=SUBJ_OBJ, paths=[], weight=1)
                dkg[subj][obj][SUBJ_OBJ]['paths'].append(path)
                
        def get_clause_head_noun_phrase_id(section:OutlineSection, clause_root:int, doc:DocManager, tid2tree_id:dict[int, int], clause_id2subjs:dict[int, list[tuple[int, Span]]]):
            clause_head_tid = section.section_nlp_local[clause_root].head.i
            clause_head_root = section.section_nlp_local[clause_head_tid:clause_head_tid+1]
            clause_head_noun_phrase_id:int = self.tid2phrase_id[clause_head_tid + section.section_nlp_global.start]
            if clause_head_noun_phrase_id < 0:
                coref = section.pron_root2coref.get(clause_head_tid, None)
                if coref is not None:
                    coref_phrase_id = self.tid2phrase_id[section.section_nlp_global.start + coref.root.i]
                    if coref_phrase_id >= 0:
                        clause_head_noun_phrase_id = coref_phrase_id
                    else:
                        clause_id = tid2tree_id[coref.root.i]
                        return [(subj, clause_head_root) for subj, subj_root in clause_id2subjs[clause_id]]
            return [(clause_head_noun_phrase_id, clause_head_root)]
        
        for section in self.sections:
            if not section.section_nlp_local:
                continue
        
            last_subjs, last_sent_start = set[int](), -1
            dep_trees = list[nx.DiGraph]()
            tid2tree_id = dict[int, int]()
            clause_id2subjs = defaultdict(list[tuple[int, Span]])
            clause_id2objs = defaultdict(list[tuple[int, Span]])
            root2clause_roots = defaultdict(list[int])
            
            for sent in section.section_nlp_local.sents:
                prev_clause_num = len(dep_trees)
                sent_dep_tree = nx.Graph()
                roots = [sent.root]

                while roots:
                    dep_tree = nx.DiGraph()
                    root = roots.pop()
                    dep_tree.add_node(root.i, dep=root.dep_ if root.dep_ not in CLAUSE_DEPS else f'{root.dep_}_ROOT')
                    dep_tree.graph['root'] = root.i
                    tokens = [root]
                    tid2tree_id[root.i] = len(dep_trees)
                    while tokens:
                        token = tokens.pop()
                        for child in token.children:
                            sent_dep_tree.add_edge(token.i, child.i)
                            if child.dep_ in CLAUSE_DEPS and self.tid2phrase_id[child.i + section.section_nlp_global.start] < 0:
                                roots.append(child)
                                # root2head_id[child.i] = token.i
                                root2clause_roots[root.i].append(child.i)
                            else:
                                tid2tree_id[child.i] = tid2tree_id[root.i]
                                dep_tree.add_node(child.i, dep=child.dep_)
                                dep_tree.add_edge(token.i, child.i)
                                tokens.append(child)
                    dep_trees.append(dep_tree)
                    
                new_last_subjs = set[int]()

                # Collect subjs and objs in each clause and add in-clause subj-obj edges (coreference outside the clause is considered)
                for clause_id in range(prev_clause_num, len(dep_trees)):
                    dep_tree = dep_trees[clause_id]
                    # Find the noun phrases
                    noun_phrases = list[Span]()
                    phrase_ids = {self.tid2phrase_id[section.section_nlp_global.start + node] for node in dep_tree.nodes}
                    for phrase_id in phrase_ids:
                        if phrase_id >= 0:
                            noun_phrase = self.phrases[phrase_id]
                            noun_phrases.append(section.section_nlp_local[noun_phrase.start-section.section_nlp_global.start: noun_phrase.end-section.section_nlp_global.start])
                    
                    for np_label, noun_phrase in [('np', np) for np in noun_phrases] + [('p', pron) for pron in section.prons if dep_tree.has_node(pron.root.i)]:
                        root_phrase = noun_phrase

                        while root_phrase.root.dep_ in {'conj', 'appos'}:
                            root_phrase_id:int = self.tid2phrase_id[section.section_nlp_global.start + root_phrase.root.head.i]
                            if root_phrase_id < 0:
                                if root_phrase.root.head.pos_ in NOUN_POS:
                                    root_phrase = section.section_nlp_local[root_phrase.root.head.i: root_phrase.root.head.i+1]
                                else:
                                    break
                            else:
                                root_phrase_global = self.phrases[root_phrase_id]
                                root_phrase = section.section_nlp_local[root_phrase_global.start-section.section_nlp_global.start: root_phrase_global.end-section.section_nlp_global.start]
                        
                        global_phrase_ids = []
                        if np_label == 'np':
                            global_phrase_ids.append(self.tid2phrase_id[section.section_nlp_global.start + noun_phrase.root.i])
                        else:
                            if noun_phrase.root.i in section.pron_root2coref:
                                coref = section.pron_root2coref[noun_phrase.root.i]
                                coref_phrase_id = self.tid2phrase_id[section.section_nlp_global.start + coref.root.i]
                                if coref_phrase_id >= 0:
                                    global_phrase_ids.append(coref_phrase_id)
                                else:
                                    coref_clause_id = tid2tree_id[coref.root.i]
                                    global_phrase_ids.extend(subj for subj, _ in clause_id2subjs[coref_clause_id])
                        if not global_phrase_ids:
                            continue
                        
                        if 'subj' in root_phrase.root.dep_:
                            clause_id2subjs[clause_id].extend((global_phrase_id, root_phrase) for global_phrase_id in global_phrase_ids)
                            if dep_tree.nodes[dep_tree.graph['root']]['dep'] == 'ROOT':
                                new_last_subjs.update(global_phrase_ids)
                        else:
                            clause_id2objs[clause_id].extend((global_phrase_id, root_phrase) for global_phrase_id in global_phrase_ids)

                    add_subj_obj_edges(dkg, clause_id2subjs[clause_id], clause_id2objs[clause_id], sent_dep_tree, section)
                        # for last_subj in last_subjs:
                        #     if dkg.has_edge(last_subj, obj, key=ADJACENT):
                        #         dkg[last_subj][obj][ADJACENT]['sent_range'].append((last_sent_start + section.section_nlp_global.start, sent.end + section.section_nlp_global.start))
                        #     else:
                        #         dkg.add_edge(last_subj, obj, key=ADJACENT, sent_range=[(last_sent_start + section.section_nlp_global.start, sent.end + section.section_nlp_global.start)], weight=2)
                        
                # Collect cross-clause edges
                for clause_id in range(prev_clause_num, len(dep_trees)):
                    dep_tree = dep_trees[clause_id]
                    for sub_clause_root in root2clause_roots[dep_tree.graph['root']]:
                        sub_clause_id = tid2tree_id[sub_clause_root]
                        
                        # Add subj-obj edges between clauses
                        if section.section_nlp_local[sub_clause_root].dep_ in CLAUSE_DEPS_USE_HEAD_SUBJ:
                            # Get subjects
                            temp_subjs = clause_id2subjs[clause_id]
                            # Get objects
                            temp_objs = clause_id2subjs[sub_clause_id] + clause_id2objs[sub_clause_id]
                            # Add edges
                            add_subj_obj_edges(dkg, temp_subjs, temp_objs, sent_dep_tree, section)
                            # Update subjects if necessary
                            if not clause_id2subjs[sub_clause_id] and section.section_nlp_local[sub_clause_root].dep_ in CLAUSE_DEPS_TRANSIST_HEAD_SUBJ:
                                clause_id2subjs[sub_clause_id] = clause_id2subjs[clause_id]
                            
                        elif section.section_nlp_local[sub_clause_root].dep_ == ACL:
                            head_noun_phrases = [
                                (clause_head_noun_phrase_id, head_noun_phrase) 
                                for clause_head_noun_phrase_id, head_noun_phrase in get_clause_head_noun_phrase_id(section, sub_clause_root, self, tid2tree_id, clause_id2subjs)
                                if clause_head_noun_phrase_id >= 0
                            ]
                            if not head_noun_phrases:
                                continue
                            
                            # Get subjects
                            temp_subjs = head_noun_phrases
                            # Get objects
                            temp_objs = clause_id2subjs[sub_clause_id] + clause_id2objs[sub_clause_id]
                            # Add edges
                            add_subj_obj_edges(dkg, temp_subjs, temp_objs, sent_dep_tree, section)
                            # Update subjects if necessary
                            if not clause_id2subjs[sub_clause_id]:
                                clause_id2subjs[sub_clause_id] = head_noun_phrases
                        
                        # elif section.section_nlp_local[sub_clause_root].dep_ == XCOMP:
                        #     for (subj, subj_root), (obj, obj_root) in itertools.product(clause_id2subjs[clause_id], clause_id2subjs[sub_clause_id] + clause_id2objs[sub_clause_id]):
                        #         path = tuple([path_id + section.section_nlp_global.start for path_id in nx.shortest_path(sent_dep_tree, source=subj_root.root.i, target=obj_root.root.i)[1:-1]])
                        #         if not dkg.has_edge(subj, obj, key=SUBJ_OBJ):
                        #             dkg.add_edge(subj, obj, key=SUBJ_OBJ, paths=[], weight=1)
                        #         dkg[subj][obj][SUBJ_OBJ]['paths'].append(path)
                        #     for (subj, subj_root), (obj, obj_root) in itertools.product(clause_id2objs[clause_id], clause_id2subjs[sub_clause_id] + clause_id2objs[sub_clause_id]):
                        #         path = tuple([path_id + section.section_nlp_global.start for path_id in range(subj_root.root.i+1, obj_root.root.i)])
                        #         if not dkg.has_edge(subj, obj, key=SUBJ_OBJ):
                        #             dkg.add_edge(subj, obj, key=SUBJ_OBJ, paths=[], weight=1)
                        #         dkg[subj][obj][SUBJ_OBJ]['paths'].append(path)
                        
                        elif section.section_nlp_local[sub_clause_root].dep_ == RELCL:
                            head_noun_phrases = [
                                (clause_head_noun_phrase_id, head_noun_phrase) 
                                for clause_head_noun_phrase_id, head_noun_phrase in get_clause_head_noun_phrase_id(section, sub_clause_root, self, tid2tree_id, clause_id2subjs)
                                if clause_head_noun_phrase_id >= 0
                            ]
                            if not head_noun_phrases:
                                continue
                            
                            if not clause_id2subjs[sub_clause_id]:
                                # Get subjects
                                temp_subjs = head_noun_phrases
                                # Get objects
                                temp_objs = clause_id2objs[sub_clause_id]
                                # Add edges
                                add_subj_obj_edges(dkg, temp_subjs, temp_objs, sent_dep_tree, section)
                                # Update subjects if necessary
                                clause_id2subjs[sub_clause_id] = head_noun_phrases
                                
                            else:
                                # Get subjects
                                temp_subjs = clause_id2subjs[sub_clause_id]
                                # Get objects
                                temp_objs = head_noun_phrases
                                # Add edges
                                add_subj_obj_edges(dkg, temp_subjs, temp_objs, sent_dep_tree, section)
                                
                        
                        # elif section.section_nlp_local[sub_clause_root].dep_ == PCOMP:
                        #     clause_head = section.section_nlp_local[sub_clause_root].head
                        #     while clause_head.pos_ not in NOUN_POS:
                        #         new_clause_head = clause_head.head
                        #         if new_clause_head == clause_head:
                        #             break
                        #         clause_head = new_clause_head
                        #     if clause_head.pos_ not in NOUN_POS:
                        #         continue
                        #     clause_head_tid = clause_head.i
                        #     clause_head_noun_phrase_id = self.tid2phrase_id[clause_head_tid + section.section_nlp_global.start]
                        #     if clause_head_noun_phrase_id < 0:
                        #         if clause_head_tid in section.pron_root2coref:
                        #                 coref = section.pron_root2coref[clause_head_tid]
                        #                 coref_phrase_id = self.tid2phrase_id[section.section_nlp_global.start + coref.root.i]
                        #                 if coref_phrase_id >= 0:
                        #                     clause_head_noun_phrase_id = coref_phrase_id
                                            
                        #     if clause_head_noun_phrase_id < 0:
                        #         continue
                        #     for (obj, obj_root) in clause_id2subjs[sub_clause_id] + clause_id2objs[sub_clause_id]:
                        #         path = tuple([path_id + section.section_nlp_global.start for path_id in nx.shortest_path(sent_dep_tree, source=clause_head_tid, target=obj_root.root.i)[1:-1]])
                        #         if not dkg.has_edge(clause_head_noun_phrase_id, obj, key=SUBJ_OBJ):
                        #             dkg.add_edge(clause_head_noun_phrase_id, obj, key=SUBJ_OBJ, paths=[], weight=1)
                        #         dkg[clause_head_noun_phrase_id][obj][SUBJ_OBJ]['paths'].append(path)
                            
                last_subjs = new_last_subjs
                last_sent_start = sent.start
                
        # Build edges between phrases based on shared n-grams
        
        shared_text2phrases:dict[tuple[str], set[tuple[int, tuple[int]]]] = defaultdict(set[tuple[int, tuple[int]]])
        for phrase_id, phrase in enumerate(self.phrases):
            for tid, token in enumerate(phrase):
                if token.lemma_ not in stop_words.STOP_WORDS and token.pos_ != PUNCT:
                    shared_text2phrases[(token.lemma_, ) if token.pos_ in ENT_POS else (token.text.lower(), )].add((phrase_id, (tid, )))
        shared_text2phrases = {shared_text: phrase_ids for shared_text, phrase_ids in shared_text2phrases.items() if len({phrase_id for phrase_id, _ in phrase_ids}) > 1}
        new_shared_text2phrases = shared_text2phrases
        
        while new_shared_text2phrases:
            temp_new_shared_text2phrases:dict[tuple[str], set[tuple[int, tuple[int]]]] = defaultdict(set[tuple[int, tuple[int]]])
            for shared_text, phrases in new_shared_text2phrases.items():
                for phrase_id, tids in phrases:
                    phrase = self.phrases[phrase_id]
                    if tids[-1] < len(phrase) - 1:
                        right_extended_tids = tids + (tids[-1]+1, )
                        new_token = phrase[right_extended_tids[-1]]
                        right_extended_text = (shared_text + (new_token.lemma_, )) if new_token.pos_ in ENT_POS else (shared_text + (new_token.text.lower(), ))
                        temp_new_shared_text2phrases[right_extended_text].add((phrase_id, right_extended_tids))
                    if tids[0] > 0:
                        left_extended_tids = (tids[0]-1, ) + tids
                        new_token = phrase[left_extended_tids[0]]
                        left_extended_text = ((new_token.lemma_, ) + shared_text) if new_token.pos_ in ENT_POS else ((new_token.text.lower(), ) + shared_text)
                        temp_new_shared_text2phrases[left_extended_text].add((phrase_id, left_extended_tids))
            
            temp_new_shared_text2phrases = {shared_text: phrase_ids for shared_text, phrase_ids in temp_new_shared_text2phrases.items() if len({phrase_id for phrase_id, _ in phrase_ids}) > 1}
            shared_text2phrases.update(temp_new_shared_text2phrases)
            new_shared_text2phrases = temp_new_shared_text2phrases
            
        # complete_shared_text2phrases = copy.deepcopy(shared_text2phrases)
        # striped_shared_text2full_shared_text:dict[tuple[str], dict[int, tuple[str]]] = defaultdict(dict[int, tuple[str]])
        for shared_text, phrases in shared_text2phrases.items():
            if not phrases:
                continue
            sample_phrase_id, sample_tids = list(phrases)[0]
            phrase = self.phrases[sample_phrase_id]
            shared_span = phrase[sample_tids[0]:sample_tids[-1]+1]
            striped_shared_span = DocManager.strip_ent(shared_span)
            if shared_span.text == striped_shared_span.text:
                continue
            if not striped_shared_span.text.strip():
                phrases.clear()
                continue
            global_tids = [tid + phrase.start for tid in sample_tids]
            new_sample_tid_idx = tuple(global_tids.index(token.i) for token in striped_shared_span)
            new_shared_text = tuple(token.lemma_ if token.pos_ in ENT_POS else token.text.lower() for token in striped_shared_span)
            for phrase_id, tids in phrases:
                shared_text2phrases[new_shared_text].add((phrase_id, tuple(tids[idx] for idx in new_sample_tid_idx)))
            phrases.clear()
            
        shared_text2phrases = {shared_text: phrases for shared_text, phrases in shared_text2phrases.items() if phrases}
        phrase_id2shared_text2tids:dict[int, dict[tuple[str], set[tuple[int]]]] = defaultdict(lambda: defaultdict(set[tuple[int]]))
        shared_text2formal_text = dict[tuple[str], str]()
        
        # test_pairs = list[tuple[int, int, str, tuple[Span, Span]]]()
        phrase_id2shared_text2phrases:dict[int, dict[tuple[str], set[int]]] = defaultdict(lambda: defaultdict(set[tuple[int, tuple[int]]]))
        for shared_text, phrases in shared_text2phrases.items():
            sample_phrase_id, tids = list(phrases)[0]
            shared_span = self.phrases[sample_phrase_id][tids[0]:tids[-1]+1]
            formal_shared_text = shared_span.text
            for token, formal_token in zip(shared_span, shared_text):
                if token.text != formal_token:
                    formal_shared_text = formal_shared_text.replace(token.text, formal_token, 1)
            shared_text2formal_text[shared_text] = formal_shared_text
            for phrase_id, tids in phrases:
                phrase_id2shared_text2tids[phrase_id][shared_text].add(tids)
            for (phrase_id_0, tids_0), (phrase_id_1, tids_1) in itertools.combinations(sorted(phrases, key=lambda x: (x[0], x[1][0])), 2):
                phrase_id2shared_text2phrases[phrase_id_0][shared_text].add(phrase_id_1)
            
        shared_ngram_edges2tids: dict[tuple[int, int, tuple[str]], list[tuple[int]]] = defaultdict(list[tuple[int]])
        for phrase_id_0, temp_shared_text2phrases in phrase_id2shared_text2phrases.items():
            phrase_id2seen_tids: dict[int, set[int]] = defaultdict(set)
            for shared_text, phrases in sorted(temp_shared_text2phrases.items(), key=lambda k: len(k[0]), reverse=True):
                for phrase_id_1 in sorted(phrases):
                    if self.phrases[phrase_id_0].sent.start != self.phrases[phrase_id_1].sent.start:
                        for tids_1 in phrase_id2shared_text2tids[phrase_id_1][shared_text]:
                            if not phrase_id2seen_tids[phrase_id_1].intersection(tids_1):
                                phrase_id2seen_tids[phrase_id_1].update(tids_1)
                                shared_ngram_edges2tids[(phrase_id_0, phrase_id_1, shared_text)].append(tids_1)
                        break
        
        tid2score = np.zeros(len(self.doc_spacy))
        for phrase in self.phrases:
            curr_tokens = [phrase.root]
            layer = 0
            while curr_tokens:
                next_tokens = []
                for token in curr_tokens:
                    tid2score[token.i] = 1. / (layer * 0.5 + 1)
                    next_tokens.extend([child for child in token.children if child.i >= phrase.start and child.i < phrase.end])
                curr_tokens = next_tokens
                layer += 1
            for token in phrase:
                if tid2score[token.i] == 0:
                    tid2score[token.i] = 1. / (abs(token.i - phrase.root.i) * 0.5 + 1)
        
        # Build similarity matrix using TF-IDF vectors
        corpus = [' '.join(token.lemma_ if token.pos_ in ENT_POS else token.text.lower() for token in phrase if token.pos_ not in {PUNCT, DET}) for phrase in self.phrases]
        vectorizer = TfidfVectorizer(ngram_range=(1, 4))
        tfidf_matrix = vectorizer.fit_transform(corpus)
        cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
        
        for (phrase_id_0, phrase_id_1, shared_text), tids_1s in shared_ngram_edges2tids.items():
            if not dkg.has_edge(phrase_id_0, phrase_id_1, SHARED_TEXT):
                similarity = min(cosine_sim[phrase_id_0][phrase_id_1], 1.0)
                weight = (abs(self.tid2sent_id[self.phrases[phrase_id_1].root.i] - self.tid2sent_id[self.phrases[phrase_id_0].root.i]) + 1) * (1 - similarity)
                if (weight <= 2 and similarity > 0.2) or similarity > 0.5:
                    dkg.add_edge(phrase_id_0, phrase_id_1, SHARED_TEXT, weight=weight, similarity=similarity)

        self.dkg = dkg
    
    def build_chunks(self, sent_chunk:bool = False, max_seq_length:int = None):
        cid = 0
        self.tid2chunk_id = np.ones(len(self.doc_spacy), dtype=int) * -1
        self.chunks = list[Document]()
        
        if not sent_chunk:
            # Load text splitter
            
            text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
                tokenizer=self.embedding._client.tokenizer,
                chunk_size=min(max_seq_length or self.embedding._client.get_max_seq_length(), self.embedding._client.get_max_seq_length()), 
                chunk_overlap=0,
                separators=["\n\n", ". ", ", ", " ", ""],
                keep_separator='end'
            )
        
        for section in self.sections:
            section.merged_blocks = []
            if section.section_nlp_global:
                chunks = [section.section_nlp_global[chunk.start:chunk.end] for chunk in section.section_nlp_local.sents] if sent_chunk else DocManager.iterate_find(section.section_nlp_global, text_splitter.split_text(section.section_nlp_global.text))
                for chunk in chunks:
                    new_chunk = Document(chunk.text, metadata={'start_idx': chunk[0].idx, 'chunk_id': cid})
                    section.merged_blocks.append(new_chunk)
                    self.chunks.append(new_chunk)
                    self.tid2chunk_id[chunk.start:chunk.end] = cid
                    cid += 1
        
        if not self.simple_load:
            self.pid2chunk_ids = {pid: set(self.tid2chunk_id[phrase.start:phrase.end]) for pid, phrase in enumerate(self.phrases)}
        
        if hasattr(self, 'vectorstore'):
            self.vectorstore.delete_collection()
            del self.vectorstore
        self.vectorstore = Chroma.from_documents(documents=self.chunks, embedding=self.embedding)
                    
    
    # =========================== Doc Helper Functions ===========================
    
    # --------------------------- PyMuPDF Related Functions ---------------------------
    @staticmethod
    def split_block_by_font(block:dict, normal_font:str):
        if not block['lines']:
            return []
        start_font = (round(block['lines'][0]['spans'][0]['size'], 0), block['lines'][0]['spans'][0]['font'])
        if start_font == normal_font:
            return [block]
        
        if len(block['lines'][0]['spans'][0]['chars']) < 2:
            return [block]
        
        if normal_font not in [(round(span['size'], 0), span['font']) for line in block['lines'] for span in line['spans']]:
            return [block]
        
        blocks = list[dict]()
        new_block = copy.deepcopy(block)
        new_block['lines'] = []
        startswith_diff_font = True
        last_span_chars:str = ''
        for line in block['lines']:
            new_line = copy.deepcopy(line)
            new_line['spans'] = []
            for span in line['spans']:
                span_chars = ''.join(c['c'] for c in span['chars']).strip()
                if startswith_diff_font and (round(span['size'], 0), span['font']) != start_font:
                    startswith_diff_font = False
                    if span_chars and (span_chars[0].isnumeric() or (span_chars[0].isalpha() and span_chars[0].isupper())) and last_span_chars[-1] not in {';', ':', ','}:
                        if new_line['spans']:
                            new_block['lines'].append(new_line)
                            new_line = copy.deepcopy(line)
                            new_line['spans'] = []
                        blocks.append(new_block)
                        new_block = copy.deepcopy(block)
                        new_block['lines'] = []
                new_line['spans'].append(span)
                last_span_chars = span_chars
            new_block['lines'].append(new_line)
        
        blocks.append(new_block)
        return blocks
    
    @staticmethod
    def remove_space_before_punct(text:str):
        return text.replace(' .', '.').replace(' ,', ',').replace(' ?', '?').replace(' !', '!').replace(' :', ':').replace(' ;', ';').replace(' )', ')').replace('( ', '(')
    
    @staticmethod
    def remove_citations(text:str):
        for parentheses in re.findall(r'\([^()]*\)', text):
            if ' et al.' in parentheses:
                text = text.replace(parentheses, '')
        return DocManager.remove_space_before_punct(' '.join(text.split()))
    
    @staticmethod
    def get_block_from_dict(block:dict):
        lines, temp_line, line_start_fonts, temp_start_font, last_bbox_bottom_right = list[str](), '', list[tuple[float, str]](), None, 0.
        phrases_with_hyphen = set[str]()
        for lid, line in enumerate(block['lines']):
            line_text = ' '.join(''.join([char['c'] for char in span['chars']]) for span in line['spans'])
            if not temp_line and line_text:
                temp_start_font = (round(line['spans'][0]['size'], 0), line['spans'][0]['font'])
            last_line = temp_line
            temp_line += line_text
            if re.match(r'.*(?<!-)-$', temp_line):
                # Do not clear the temp_line if the line ends with a hyphen so that the next line can be concatenated
                # if lid != len(block['lines']) - 1:
                #     temp_line = temp_line[:-1]
                pass
            else:
                phrases_with_hyphen.update([phrase for phrase in line_text.split() if '-' in phrase and (not line_text.startswith(phrase) or re.match(r'.*(?<!-)-$', last_line) is None)])
                if not lines or line['bbox'][1] > last_bbox_bottom_right:
                    lines.append(DocManager.remove_space_before_punct(' '.join(temp_line.split())))
                    line_start_fonts.append(temp_start_font)
                    last_bbox_bottom_right = line['bbox'][3]
                else:
                    lines[-1] += ' ' + DocManager.remove_space_before_punct(' '.join(temp_line.split()))
                temp_line = ''
                # temp_start_font = None
        if temp_line:
            if not lines or line['bbox'][1] > last_bbox_bottom_right:
                lines.append(DocManager.remove_space_before_punct(' '.join(temp_line.split())))
                line_start_fonts.append(temp_start_font)
            else:
                lines[-1] += ' ' + DocManager.remove_space_before_punct(' '.join(temp_line.split()))
            
        return DocBlock(text=DocManager.remove_space_before_punct(' '.join(lines)), bbox=block['bbox'], phrases_with_hyphen=phrases_with_hyphen, lines=lines, line_start_fonts=line_start_fonts)

    @staticmethod
    def get_block_from_lines(line_texts:list[str]):
        lines, temp_line = list[str](), ''
        phrases_with_hyphen = set[str]()
        for lid, line_text in enumerate(line_texts):
            last_line = temp_line
            temp_line += line_text
            if re.match(r'.*(?<!-)-$', temp_line):
                # Do not clear the temp_line if the line ends with a hyphen so that the next line can be concatenated
                # if lid != len(block['lines']) - 1:
                #     temp_line = temp_line[:-1]
                pass
            else:
                phrases_with_hyphen.update([phrase for phrase in line_text.split() if '-' in phrase and (not line_text.startswith(phrase) or re.match(r'.*(?<!-)-$', last_line) is None)])
                if not lines:
                    lines.append(DocManager.remove_space_before_punct(' '.join(temp_line.split())))
                else:
                    lines[-1] += ' ' + DocManager.remove_space_before_punct(' '.join(temp_line.split()))
                temp_line = ''
                # temp_start_font = None
        if temp_line:
            if not lines:
                lines.append(DocManager.remove_space_before_punct(' '.join(temp_line.split())))
            else:
                lines[-1] += ' ' + DocManager.remove_space_before_punct(' '.join(temp_line.split()))
            
        return DocBlock(text=DocManager.remove_space_before_punct(' '.join(lines)), bbox=None, phrases_with_hyphen=phrases_with_hyphen, lines=lines, line_start_fonts=None)

    @staticmethod
    def block_startswith(block:str, prefix:str):
        return ''.join(block.split()).lower().startswith(''.join(prefix.split()).lower())
    
    @staticmethod
    def block_endswith(block:str, suffix:str):
        return ''.join(block.split()).lower().endswith(''.join(suffix.split()).lower())
    
    def remove_tables(self):
        for page in self.pdf_doc:
            for table in page.find_tables():
                page.add_redact_annot(table.bbox)
            page.apply_redactions()
    
    def extract_blocks(self):
        block_dicts = [block_dict for page in self.pdf_doc for block_dict in page.get_textpage().extractRAWDICT()['blocks']]
        blocks = [DocManager.get_block_from_dict(block_dict) for block_dict in block_dicts]
        
        # Get the main text style
        size_cnt:dict[str, int] = Counter()
        for block, block_dict in zip(blocks, block_dicts):
            if DocManager.block_endswith(block.text, REF_HEADER) or DocManager.block_startswith(block.text, REF_HEADER):
                break
            for line in block_dict['lines']:
                for span in line['spans']:
                    size_cnt[(round(span['size'], 0), span['font'])] += len(span['chars'])

        blocks = [block for block in blocks if block.text]
        self.main_text_style = size_cnt.most_common()[0][0]
        self.blocks = self.handle_broken_phrases(blocks)
        
    def handle_broken_phrases(self, blocks:list[DocBlock]):
        finalized_blocks = list[DocBlock]()
        for block in blocks:
            phrases = [phrase.strip(PUNCTUATION_WITHOUT_HYPHEN) for phrase in block.text.split()]
            self.doc_vocab.update(phrase.lower() for phrase in phrases if '-' not in phrase)
            self.doc_vocab.update(word.lower() for phrase in block.phrases_with_hyphen for word in phrase.split('-'))
            potential_broken_phrases = {phrase for phrase in phrases if '-' in phrase and phrase not in block.phrases_with_hyphen}
            for potential_broken_phrase in list(potential_broken_phrases):
                if potential_broken_phrase.endswith('-'):
                    potential_broken_phrases.remove(potential_broken_phrase)
                
                elif all(word in self.doc_vocab for word in potential_broken_phrase.split('-')):
                    if potential_broken_phrase.replace('-', '').lower() in self.doc_vocab:
                        fixed_phrase = potential_broken_phrase.replace('-', '')
                        block.text = block.text.replace(potential_broken_phrase, fixed_phrase)
                        block.lines = [line.replace(potential_broken_phrase, fixed_phrase) for line in block.lines]
                    else:
                        potential_broken_phrases.remove(potential_broken_phrase)
                    
                else:
                    tokens = potential_broken_phrase.split('-')
                    token_validity = [token in self.doc_vocab for token in tokens]
                    invalid_token_idx = token_validity.index(False)
                    if invalid_token_idx < len(tokens) - 1:
                        if f'{tokens[invalid_token_idx]}{tokens[invalid_token_idx+1]}'.lower() in self.doc_vocab:
                            potential_broken_phrases.remove(potential_broken_phrase)
                            fixed_phrase = potential_broken_phrase.replace(f'{tokens[invalid_token_idx]}-{tokens[invalid_token_idx+1]}', f'{tokens[invalid_token_idx]}{tokens[invalid_token_idx+1]}')
                            block.text = block.text.replace(potential_broken_phrase, fixed_phrase)
                            block.lines = [line.replace(potential_broken_phrase, fixed_phrase) for line in block.lines]
                            continue
                    if invalid_token_idx > 0:
                        if f'{tokens[invalid_token_idx-1]}{tokens[invalid_token_idx]}'.lower() in self.doc_vocab:
                            potential_broken_phrases.remove(potential_broken_phrase)
                            fixed_phrase = potential_broken_phrase.replace(f'{tokens[invalid_token_idx-1]}-{tokens[invalid_token_idx]}', f'{tokens[invalid_token_idx-1]}{tokens[invalid_token_idx]}')
                            block.text = block.text.replace(potential_broken_phrase, fixed_phrase)
                            block.lines = [line.replace(potential_broken_phrase, fixed_phrase) for line in block.lines]
                            continue
                    
            block.potential_broken_phrases = potential_broken_phrases
            finalized_blocks.append(block)
        return finalized_blocks

    # --------------------------- Spacy Related Functions ---------------------------
    @staticmethod
    def strip_ent(noun_chunk:Span):
        start, end = 0, 0
        while start < len(noun_chunk):
            if ((noun_chunk[start].pos_ not in [PUNCT, DET] and noun_chunk[start].text not in string.punctuation) or (noun_chunk[start].text.strip() == '(' and ')' in noun_chunk[start:].text)) and noun_chunk[start].lemma_ not in stop_words.STOP_WORDS:
                break
            start += 1
        for end in range(len(noun_chunk)-1, start-1, -1):
            if ((noun_chunk[end].pos_ not in [PUNCT, DET] and noun_chunk[end].text not in string.punctuation) or (noun_chunk[end].text.strip() == ')' and '(' in noun_chunk[start:end+1].text)) and noun_chunk[end].lemma_ not in stop_words.STOP_WORDS:
                break
        return noun_chunk[start:end+1]
    
    @staticmethod
    def get_full_noun_phrase(noun_phrase:Span):
        # Get the root of the noun phrase
        temp_front_token = noun_phrase[0]
        temp_back_token = noun_phrase[-1]
        noun_phrase_deps = {'compound', 'nmod', 'amod', 'det', 'poss'}
        
        root = noun_phrase.root
        temp_front_token = root if temp_front_token.i > root.i else temp_front_token
        temp_back_token = root if temp_back_token.i < root.i else temp_back_token
        while root.dep_ in noun_phrase_deps:
            root = root.head
            temp_front_token = root if temp_front_token.i > root.i else temp_front_token
            temp_back_token = root if temp_back_token.i < root.i else temp_back_token
        
        children = [child for child in root.children if child.dep_ in noun_phrase_deps]
        while children:
            new_children = []
            for c in children:
                temp_front_token = c if temp_front_token.i > c.i else temp_front_token
                temp_back_token = c if temp_back_token.i < c.i else temp_back_token
                new_children.extend(child for child in c.children if child.dep_ in noun_phrase_deps)
            children = new_children
        return root.doc[temp_front_token.i:temp_back_token.i+1]
    
    @staticmethod
    def iterate_find(doc:Span, texts:list[str]):
        start_idx = 0
        for text in texts:
            start_idx = doc.text.index(text, start_idx)
            end_idx = start_idx + len(text)
            yield doc.char_span(start_idx, end_idx)
            start_idx = end_idx
    # --------------------------- Self-Defined Structure Related Functions ---------------------------
    
    @staticmethod
    def get_chunks_from_section(section:OutlineSection):
        passages = list[Document]()
        passages.extend(section.merged_blocks)
        for child in section.children:
            passages.extend(DocManager.get_chunks_from_section(child))
        return passages
    
    def get_section_by_header(self, header:str):
        search_sections = self.sections
        target_section = None
        for sub_section_header in header.split('\n'):
            sub_section_header = sub_section_header.strip()
            new_search_sections = list[OutlineSection]()
            for section in search_sections:
                if section.header == sub_section_header:
                    target_section = section
                    if section.children:
                        new_search_sections = section.children
                    break
            search_sections = new_search_sections
        return target_section
                
    def collect_appos(self):
        pass
    
    def advanced_ctrl_f(self, search_words:list[str]):
        sw_map = []
        section_id2idx = {section.section_id : vid for vid, section in enumerate(self.sections)}
        for sw in search_words:
            temp_array = np.zeros((len(section_id2idx)))
            for section in self.ctrl_f(sw):
                temp_array[section_id2idx[section.section_id]] = 1
            sw_map.append(temp_array)
        sw_map = np.vstack(sw_map)
        for width in range(sw_map.shape[1]):
            for idx in range(sw_map.shape[1]):
                if idx + width < sw_map.shape[1]:
                    if sw_map[:, idx].any() and sw_map[:, idx + width].any() and sw_map[:, idx:idx+width+1].sum(1).all():
                        yield self.sections[idx:idx+width+1]
            
    def get_section_for_span(self, span:Span) -> OutlineSection | None:
        if self.tid2section_id[span.start] >= 0 and self.tid2section_id[span.end-1] >= 0 and self.tid2section_id[span.start] == self.tid2section_id[span.end-1]:
            return self.sections[self.tid2section_id[span.start]]
        
    def plot_dkg(self):
        y_start = 0
        width = 80
        nodes = []
        for section in self.sections:
            if section.section_nlp_local:
                # start_idx = 0
                for sent in section.section_nlp_local.sents:
                    start_idx = sent[0].idx
                    last_chunk_tid = sent.start + section.section_nlp_global.start
                    last_chunk_id = self.tid2phrase_id[last_chunk_tid]
                    curr_chunk_tid = last_chunk_tid
                    curr_chunk_id = last_chunk_id
                    for token in sent[1:]:
                        curr_chunk_tid = token.i + section.section_nlp_global.start
                        curr_chunk_id = self.tid2phrase_id[curr_chunk_tid]
                        if curr_chunk_id != last_chunk_id:
                            dist2start = self.doc_spacy[last_chunk_tid].idx - section.section_nlp_global[0].idx - start_idx
                            if dist2start > width:
                                y_start += 20
                                start_idx = self.doc_spacy[last_chunk_tid].idx - section.section_nlp_global[0].idx
                                dist2start = 0
                            nodes.append({'data': {'id': last_chunk_tid, 'label': self.doc_spacy[last_chunk_tid:curr_chunk_tid].text}, 'position': {'x': dist2start * 6.1, 'y': y_start}, 'classes': 'noun_phrase' if last_chunk_id >= 0 else 'text', 'locked': True})
                            last_chunk_tid = curr_chunk_tid
                            last_chunk_id = curr_chunk_id
                    dist2start = self.doc_spacy[last_chunk_tid].idx - section.section_nlp_global[0].idx - start_idx
                    if dist2start > width:
                        y_start += 20
                        start_idx = self.doc_spacy[last_chunk_tid].idx - section.section_nlp_global[0].idx
                        dist2start = 0
                    nodes.append({'data': {'id': last_chunk_tid, 'label': self.doc_spacy[last_chunk_tid:sent.end+section.section_nlp_global.start].text}, 'position': {'x': dist2start * 6.1, 'y': y_start}, 'classes': 'noun_phrase' if last_chunk_id >= 0 else 'text', 'locked': True})
                    y_start += 50
                y_start += 100
                
        edges = [{'data': {'source': self.phrases[u].start, 'target': self.phrases[v].start, 'type': edge_type, 'similarity': similarity * 0.5}} for u, v, edge_type, similarity in self.dkg.edges(data='similarity', keys=True)]

        app = dash.Dash(__name__)
        app.layout = html.Div([
            cyto.Cytoscape(
                id='cytoscape',
                # elements=[
                #     {'data': {'id': 'one', 'label': 'Node 1'}, 'position': {'x': 50, 'y': 50}},
                #     {'data': {'id': 'two', 'label': 'Node 2'}, 'position': {'x': 200, 'y': 200}},
                #     {'data': {'source': 'one', 'target': 'two','label': 'Node 1 to 2'}}
                # ],
                # elements=[{'data': {'id': token.i, 'label': token.text}, 'position': {'x': (token.idx - sent[0].idx) * 6, 'y': 0}} for token in sent],
                elements=nodes+edges,
                layout={'name': 'preset',
                        'zoomingEnabled': False,
                        },
                style={
                    # 'width': '1000px', 
                    'height': '2000px', 
                    # 'height': '100%',
                    'backgroundColor': "#1E1E1E",
                },
                stylesheet=[
                    {
                        'selector': 'node',
                        'style': {
                            'label': 'data(label)',
                            'font-family': 'Courier',
                            'font-size': '10',
                            'text-halign': 'right',
                            'width': '1',
                            'height': '1',
                            'color': "#ce9178",
                            # 'autoungrabify': True,
                        }
                    },
                    {
                        'selector': '.noun_phrase',
                        'style': {
                            # 'label': 'data(label)',
                            # 'font-family': 'Courier',
                            # 'font-weight': 'bold',
                            # 'text-halign': 'right',
                            # 'width': '1',
                            # 'height': '1',
                            'color': "#9CDCFE",
                            'font-weight': 'bold',
                        }
                    },
                    {
                        'selector': 'edge',
                        'style': {
                            # 'label': 'data(label)',
                            # 'font-family': 'Courier New',
                            # 'font-size': '12px'
                            'curve-style': 'unbundled-bezier',
                            'target-arrow-shape': 'vee',
                            'width': 1,
                            'opacity': 0.3
                        }
                    },
                    {
                        'selector': f'edge[type = "{SUBJ_OBJ}"]', 
                        'style': {
                            'line-color': "#C586C0",
                            'target-arrow-color': "#C586C0",
                        }
                    },
                    {
                        'selector': f'edge[type = "{COREF}"]', 
                        'style': {
                            'line-color': "#4EC9B0", 
                            'line-style': 'dashed',
                            'target-arrow-color': "#4EC9B0",
                        }
                    },
                    {
                        'selector': f'edge[type = "{SHARED_TEXT}"]', 
                        'style': {
                            'line-color': "#4FC1FF", 
                            'line-style': 'dashed',
                            'target-arrow-color': "#4FC1FF",
                            'opacity': 'data(similarity)'
                        }
                    }
                ]
            )
        ])

        app.run_server(jupyter_mode="external", port=8051, host='128.174.136.27')