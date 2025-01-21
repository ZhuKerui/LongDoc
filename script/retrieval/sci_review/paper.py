import langchain_core.documents
from .base import *
from .text import *

import pymupdf
import re
import spacy
import spacy.tokens
from nltk import ngrams
from spacy.lang.en import stop_words
import pytextrank
from fastcoref import spacy_component
from pytextrank import Phrase
import networkx as nx
import itertools
import copy
import networkx as nx
from networkx.algorithms.traversal.depth_first_search import dfs_tree

from langchain_community.vectorstores import Chroma
from langchain_community.retrievers import BM25Retriever


PARAGRAPH_SEP = '\n\n'
MARGIN_RATIO = 10
MIN_WIDTH_RATIO = 3
REF_HEADER = 'references'
ACK_HEADER = 'acknowledgement'

COREF = 'coref'
SUBJ_OBJ = 'subj_obj'
ADJACENT = 'adjacent'
SHARED_TEXT = 'shared_text'

class DocBlock(BaseModel):
    text: str
    i: int
    is_section_header: bool = False
    startswith_section_header: bool = False
    
    
class OutlineSection:
    def __init__(self, header:str, level:int, section_id:int, blocks:list[DocBlock]=[]):
        self.header = header
        self.level = level
        self.section_id = section_id
        self.blocks = blocks
        self.merged_blocks = list[langchain_core.documents.Document]()
        self.next_sibling:OutlineSection = None
        self.prev_sibling:OutlineSection = None
        self.children = list[OutlineSection]()
        self.parent:OutlineSection = None
        
        self.section_nlp_global:Span = None
        self.section_nlp_local:Doc = None
        self.prons = list[Span]()
        self.pron_root2corefs = dict[int, list[Span]]()
        
        
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
    def __init__(self, tool_llm:str = "gpt-4o-mini", emb_model:str = DEFAULT_EMB_MODEL):
        self.tool_llm = tool_llm
        self.nlp = spacy.load('en_core_web_lg')
        self.nlp.add_pipe("positionrank")
        self.nlp.add_pipe("fastcoref")
        self.client = OpenAI(api_key=os.environ['OPENAI_AUTO_SURVEY'])
        
        # Load Embeddings
        self.emb_model_name = emb_model
        self.model_kwargs = {'device': 'cpu'}
        self.encode_kwargs = {'normalize_embeddings': False}
        self.embedding = HuggingFaceEmbeddings(
            model_name=self.emb_model_name,
            model_kwargs=self.model_kwargs,
            encode_kwargs=self.encode_kwargs
        )
        
        # Load text splitter
        self.text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
            tokenizer=self.embedding._client.tokenizer,
            chunk_size=self.embedding._client.get_max_seq_length(), 
            chunk_overlap=0,
            separators=["\n\n", ".", ",", " ", ""],
            keep_separator='end'
        )
        
    def load_doc(self, doc_file:str = None, doc_strs:list[str] = None):
        assert (doc_file is not None) ^ (doc_strs is not None), "Only doc_file or doc_strs should be not None"
        
        if doc_file:
            self.pdf_doc:pymupdf.Document = pymupdf.open(doc_file)
            self.remove_tables()
            self.get_main_text_style()
            self.extract_blocks()
            outline = '\n'.join(f"{'    ' * (level - 1)}{section_name}" for level, section_name, page in self.pdf_doc.get_toc())
        else:
            assert all('\n' not in doc_str for doc_str in doc_strs), "doc_strs should have no \n"
            self.pdf_doc = None
            self.main_text_style = None
            self.bid2block = None
            self.blocks = [DocBlock(text=doc_str, i=bid) for bid, doc_str in enumerate(doc_strs)]
            outline = ''
            
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
                outline = self.generate_outline()
                err_msg_cnt.clear()
            else:
                print(f'Regeneration attempt: {re_gen_cnt}\n\n{outline}\n\n{err_msg}')
                outline = self.correct_outline(outline, err_msg)
            outline_pass, err_msg = self.parse_outline(outline)
            
        self.reduce_noise()
        self.build_chunks()
        self.collect_keyphrases()
        # self.coref_resolution()
        self.build_dkg()

    @property
    def doc_str(self):
        return PARAGRAPH_SEP.join(block.text for block in self.blocks)
    
    @property
    def doc_str_wo_headers(self):
        return PARAGRAPH_SEP.join(block.text for block in self.blocks if not block.is_section_header)
    
    @property
    def outline(self):
        return '\n'.join('    ' * section.level + section.header for section in self.sections)
    
    def generate_outline(self):
        chat_completion = self.client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": f"Paper:\n\n{self.doc_str}\n\nExtract the complete table of contents (bookmarks) from the above paper. Write one section a line and use the original section number if exists. Do not include the Abstract section. Use indentation to show the hierarchy of the sections.",
                    # "content": f"Paper:\n\n{self.doc_str}\n\nExtract the complete table of contents (bookmarks) from the above paper. Write one section a line. Do not include the Abstract section. Use indentation to show the hierarchy of the sections.",
                }
            ],
            model=self.tool_llm,
        )
        return chat_completion.choices[0].message.content
        
    def correct_outline(self, outline:str, err_msg:str):
        chat_completion = self.client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    # "content": f"Paper:\n\n{self.doc_str}\n\nExtract the complete table of contents (bookmarks) from the above paper. Write one section a line and use the original section number if exists. Do not include the Abstract section. Use indentation to show the hierarchy of the sections.",
                    "content": f"Paper:\n\n{self.doc_str}\n\nExtract the complete table of contents (bookmarks) from the above paper. Write one section a line. Do not include the Abstract section. Use indentation to show the hierarchy of the sections.",
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
    def parse_outline(self, outline:str):
        # Remove generated prompt if any
        outline = outline.strip()
        if PARAGRAPH_SEP in outline:
            outline = [paragraph for paragraph in outline.split(PARAGRAPH_SEP) if '\n' in paragraph.strip()][0]
        
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
            
        doc_str_wo_space = ''.join(self.doc_str.split()).lower()
        ref_detected = False
        # mis_match_sections = list[str]()
        unexist_sections = list[str]()
        not_start_with_sections = list[str]()
        
        for section in sections:
            header_wo_space = ''.join(section.header.split()).lower()
            ref_detected = ref_detected or section.header.lower().endswith(REF_HEADER)
            if header_wo_space not in doc_str_wo_space:
                
                header_wo_space_num = ''.join(section.header.split()[1:]).lower()
                if header_wo_space_num and header_wo_space_num in doc_str_wo_space:
                    
                    if any(''.join(block.text.split()).lower().startswith(header_wo_space_num) for block in self.blocks):
                        new_section_header = section.header.split(maxsplit=1)[1]
                        section.header = new_section_header
                        continue
                    else:
                        section_block_found = False
                        for block in self.blocks:
                            block_wo_space = ''.join(block.text.split()).lower()
                            if block_wo_space.endswith(header_wo_space_num) and len(block_wo_space) - len(header_wo_space_num) < 20 and re.match(r'^[a-zA-Z0-9.]+$', block_wo_space[:block_wo_space.index(header_wo_space_num)]) is not None:
                                new_section_header = block_wo_space[:block_wo_space.index(header_wo_space_num)] + ' ' + section.header.split(maxsplit=1)[1]
                                section.header = new_section_header
                                section_block_found = True
                                break
                        if section_block_found:
                            continue
                
                unexist_sections.append(section.header)
            
            if not any(''.join(block.text.split()).lower().startswith(header_wo_space) for block in self.blocks):
                not_start_with_sections.append(section.header)
                
        # if mis_match_sections:
        #     sections_with_issue = '\n'.join(mis_match_sections)
        #     return False, f"The following section name has a mismatch in its numbering. Please ensure that all section names in your response match the numbering exactly as they appear in the paper.\n\n{sections_with_issue}"
        
        if unexist_sections:
            sections_with_issue = '\n'.join(unexist_sections)
            return False, f"The following section name does not appear in the paper. Please review the paper and ensure that all listed section names are accurately included in the text.\n\n{sections_with_issue}"
        
        if not_start_with_sections:
            sections_with_issue = '\n'.join(not_start_with_sections)
            return False, f"The specified section name does not appear at the beginning of any paragraph in the paper. Section names must either appear at the start of a paragraph or stand alone as an independent paragraph. Please ensure all section names meet these requirements. If not, the section name should be removed.\n\n{sections_with_issue}"
        
        if not ref_detected and any(''.join(block.text.split()).lower().startswith(REF_HEADER) for block in self.blocks):
            return False, "The References section is missing. You should find the References section and insert the Reference section header in its corresponding position in the table of content."
        
        self.sections = sections
        
        fill_success = self.fill_outline()
        if not fill_success:
            return False, "The order of the table of contents is incorrect. Please ensure that all sections are listed in the correct order as they appear in the paper."
        
        return True, ''
    
    def fill_outline(self):
        self.meta_data = list[DocBlock]()
        temp_content = list[DocBlock]()
        hid = -1
        for block in self.blocks:
            if hid+1 < len(self.sections):
                block_str_wo_space = ''.join(block.text.split()).lower()
                next_header_wo_space = ''.join(self.sections[hid+1].header.split()).lower()
                block.startswith_section_header = block_str_wo_space.startswith(next_header_wo_space)
                    
                if block.startswith_section_header:
                    # The next section is matched with current block
                    if hid >= 0:
                        # Fill the blocks for the previous section
                        self.sections[hid].blocks = temp_content
                        temp_content = []
                    hid += 1
                    
                    block.is_section_header = block_str_wo_space == next_header_wo_space
            
            if hid < 0:
                self.meta_data.append(block)
            else:
                temp_content.append(block)
            
        self.sections[hid].blocks = temp_content
        
        if hid + 1 < len(self.sections):
            self.sections.clear()
            self.meta_data.clear()
            return False
        
        sid = 0
        while sid < len(self.sections):
            section = self.sections[sid]
            section.section_id = sid
            new_sub_section_ids = [bid for bid, block in enumerate(section.blocks[:-1]) if self.bid2block[block.i]['bbox'] == self.bid2block[section.blocks[bid+1].i]['bbox'] and not block.is_section_header and not block.startswith_section_header]
            if new_sub_section_ids:
                for sub_section_id in new_sub_section_ids:
                    section.blocks[sub_section_id].is_section_header = True
                section.blocks, sub_section_blocks = section.blocks[:new_sub_section_ids[0]], section.blocks[new_sub_section_ids[0]:]
                last_new_sub_section:OutlineSection = None
                for sub_section_block in sub_section_blocks:
                    if sub_section_block.is_section_header:
                        sid += 1
                        new_sub_section = OutlineSection(header=sub_section_block.text, level=section.level+1, section_id=sid, blocks=[sub_section_block])
                        self.sections.insert(sid, new_sub_section)
                        last_new_sub_section = new_sub_section
                    else:
                        last_new_sub_section.blocks.append(sub_section_block)
            sid += 1
        return True
            
    def reduce_noise(self):
        new_blocks = list[DocBlock]()
        xmin, _, xmax, _ = self.bid2block[0]['bbox']
        xs = []
        section:OutlineSection
        for section in self.sections:
            if section.header.lower().endswith(REF_HEADER) or section.header.lower().endswith(ACK_HEADER) or section.header.lower().endswith(ACK_HEADER+'s'):
                break
            for block in section.blocks:
                if not block.is_section_header:
                    x0, y0, x1, y1 = self.bid2block[block.i]['bbox']
                    xmin = min(x0, xmin)
                    xmax = max(x1, xmax)
                    xs.append(x1 - x0)
        max_width = xmax - xmin
        margin = int(max_width / MARGIN_RATIO)
        min_req_width = max_width / MIN_WIDTH_RATIO
        normal_width = Counter([int(x // margin * margin) for x in xs if x > min_req_width]).most_common(1)[0][0]
        
        last_block:DocBlock = None
        new_sections = list[OutlineSection]()
        for section in self.sections:
            if section.header.lower().endswith(REF_HEADER) or section.header.lower().endswith(ACK_HEADER) or section.header.lower().endswith(ACK_HEADER+'s'):
                break
            new_section_blocks = list[DocBlock]()
            for block in section.blocks:
                x0, y0, x1, y1 = self.bid2block[block.i]['bbox']
                width = x1 - x0
                if block.is_section_header or block.startswith_section_header or (width > min_req_width and width < normal_width + margin and re.search(r'https?://[^\s/$.?#].[^\s]*', block.text) is None and re.match(r'^Table \d+: ', block.text) is None and re.match(r'^Figure \d+: ', block.text) is None):
                    if last_block is not None and not last_block.is_section_header and not block.is_section_header and not block.startswith_section_header and (last_block.text[-1].isalpha() or last_block.text[-1] not in {'.', '!', '?', ':'}):
                        if last_block.text[-1] == '-':
                            last_block.text = ''.join([last_block.text[:-1], block.text])
                        else:
                            last_block.text = ' '.join([last_block.text, block.text])
                    else:
                        new_section_blocks.append(block)
                        new_blocks.append(block)
                        last_block = block
                    
            section.blocks = new_section_blocks
            new_sections.append(section)
            
        for sid, section in enumerate(new_sections):
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
        for bid, block in enumerate(self.blocks):
            block.i = bid
        self.sections = new_sections
        
        # Finalize the sections and blocks
        self.doc_spacy = self.nlp(self.doc_str_wo_headers, disable=['fastcoref'])
        self.tid2section_id = np.ones(len(self.doc_spacy), dtype=int) * -1
        
        section_contents = list[str]()
        section_with_content = list[OutlineSection]()
        for section in self.sections:
            section_content = PARAGRAPH_SEP.join(block.text for block in section.blocks if not block.is_section_header)
            if section_content:
                section_contents.append(section_content)
                section_with_content.append(section)
        
        section_nlp_global:Span
        for section, section_nlp_global, section_nlp_local in zip(section_with_content, DocManager.iterate_find(self.doc_spacy, section_contents), self.nlp.pipe(section_contents, disable=["lemmatizer", "ner", "positionrank"])):
            section.section_nlp_global = section_nlp_global
            section.section_nlp_local = section_nlp_local
            self.tid2section_id[section.section_nlp_global.start:section.section_nlp_global.end] = section.section_id
        
    def build_chunks(self):
        cid = 0
        self.tid2chunk_id = np.ones(len(self.doc_spacy), dtype=int) * -1
        self.chunks = list[langchain_core.documents.Document]()
        
        for section in self.sections:
            if section.section_nlp_global:
                self.tid2section_id[section.section_nlp_global.start:section.section_nlp_global.end] = section.section_id
                
                for chunk in DocManager.iterate_find(section.section_nlp_global, self.text_splitter.split_text(section.section_nlp_global.text)):
                    new_chunk = langchain_core.documents.Document(chunk.text, metadata={'start_idx': chunk[0].idx, 'chunk_id': cid})
                    section.merged_blocks.append(new_chunk)
                    self.chunks.append(new_chunk)
                    self.tid2chunk_id[chunk.start:chunk.end] = cid
                    cid += 1
        
        if hasattr(self, 'vectorstore'):
            self.vectorstore.delete_collection()
            del self.vectorstore
        self.vectorstore = Chroma.from_documents(documents=self.chunks, embedding=self.embedding)
        self.dense_retriever = self.vectorstore.as_retriever()
        self.bm25_retriever = BM25Retriever.from_documents(self.chunks)
                    
    def collect_keyphrases(self):
        def split_noun_phrases(noun_phrase:spacy.tokens.Span):
            
            phrases = list[spacy.tokens.Span]()
            # Split the phrase into sub-phrases if PARAGRAPH_SEP is found
            for sub_chunk in DocManager.iterate_find(noun_phrase, [sub_noun_phrase.strip() for sub_noun_phrase in noun_phrase.text.split(PARAGRAPH_SEP)]):
                if all(len(token.text) < 2 for token in sub_chunk):
                    continue
                
                if sub_chunk.root.pos_ in ['NOUN', 'PROPN'] or sub_chunk.root.text.lower() == 'we':
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
                    
        spans = list[spacy.tokens.Span]()
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
            # span_local = DocManager.strip_ent(SentenceAnalyzerBase.get_full_noun_phrase(span_local))
            span_local = SentenceAnalyzerBase.get_full_noun_phrase(span_local)
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
            
        self.pid2chunk_ids = {pid: set(self.tid2chunk_id[phrase.start:phrase.end]) for pid, phrase in enumerate(self.phrases)}
                    
    def build_dkg(self):
        dkg = nx.MultiDiGraph()
        
        for section in self.sections:
            if section.section_nlp_local:
                for coref_cluster in section.section_nlp_local._.coref_clusters:
                    corefs = list[Span]()
                    last_phrase_id = -1
                    for coref_mention in coref_cluster:
                        mention = section.section_nlp_local.char_span(coref_mention[0], coref_mention[1])
                        curr_phrase_id = self.tid2phrase_id[mention.root.i+section.section_nlp_global.start]
                        if curr_phrase_id >= 0:
                            # If the mention is a noun phrase
                            if last_phrase_id >= 0 and curr_phrase_id != last_phrase_id:
                                dkg.add_edges_from([(last_phrase_id, curr_phrase_id, COREF), (curr_phrase_id, last_phrase_id, COREF)], weight=0)
                            last_phrase_id = curr_phrase_id
                        elif mention.root.pos_ == 'PRON':
                            # If the mention is a pronoun
                            section.prons.append(mention)
                            section.pron_root2corefs[mention.root.i] = [coref for coref in corefs]
                        elif mention.root.pos_ == 'VERB' or mention.root.pos_ == 'AUX':
                            # If the mention is a verb or an auxiliary verb
                            print('Initial Verb:', mention)
                            for child in mention.root.children:
                                if 'subj' in child.dep_:
                                    subj_id:int = self.tid2phrase_id[child.i+section.section_nlp_global.start]
                                    if subj_id >= 0:
                                        phrase = self.phrases[subj_id]
                                        mention = section.section_nlp_local[phrase.start-section.section_nlp_global.start: phrase.end-section.section_nlp_global.start]
                                    elif child.i in section.pron_root2corefs:
                                        for coref in section.pron_root2corefs[child.i][::-1]:
                                            coref_phrase_id:int = self.tid2phrase_id[section.section_nlp_global.start + coref.root.i]
                                            if coref_phrase_id >= 0:
                                                phrase = self.phrases[coref_phrase_id]
                                                mention = section.section_nlp_local[phrase.start-section.section_nlp_global.start: phrase.end-section.section_nlp_global.start]
                                                break
                                    break
                            print('Final Noun:', mention)
                        corefs.append(mention)
        
        clause_deps = {'relcl', 'advcl'}
        
        for section in self.sections:
            if not section.section_nlp_local:
                continue
        
            last_subjs, last_sent_start = set[int](), -1
            
            for sent in section.section_nlp_local.sents:
                dep_trees = list[nx.DiGraph]()
                dep_tree = nx.DiGraph()
                for token in sent:
                    dep_tree.add_node(token.i, dep=token.dep_ if token.dep_ not in clause_deps else f'{token.dep_}_ROOT')
                    if token.head.i != token.i and token.dep_ not in clause_deps:
                        dep_tree.add_edge(token.head.i, token.i)
                for node, dep in dep_tree.nodes.data('dep'):
                    if 'ROOT' in dep:
                        dep_trees.append(dep_tree.subgraph(dfs_tree(dep_tree, node).nodes))
                        
                new_last_subjs = set[int]()
                for dep_tree in dep_trees:
                    # Find the noun phrases
                    noun_phrases = list[Span]()
                    phrase_ids = {self.tid2phrase_id[section.section_nlp_global.start + node] for node in dep_tree.nodes}
                    for phrase_id in phrase_ids:
                        if phrase_id >= 0:
                            noun_phrase = self.phrases[phrase_id]
                            noun_phrases.append(section.section_nlp_local[noun_phrase.start-section.section_nlp_global.start: noun_phrase.end-section.section_nlp_global.start])
                    
                    subjs, objs = list[tuple[Span, Span]](), list[tuple[Span, Span]]()
                    for noun_phrase in noun_phrases + [pron for pron in section.prons if dep_tree.has_node(pron.root.i)]:
                        root_phrase = noun_phrase

                        while root_phrase.root.dep_ in {'conj', 'appos'}:
                            root_phrase_id:int = self.tid2phrase_id[section.section_nlp_global.start + root_phrase.root.head.i]
                            if root_phrase_id < 0:
                                break
                            root_phrase_global = self.phrases[root_phrase_id]
                            root_phrase = section.section_nlp_local[root_phrase_global.start-section.section_nlp_global.start: root_phrase_global.end-section.section_nlp_global.start]
                            
                        if 'subj' in root_phrase.root.dep_:
                            subjs.append((noun_phrase, root_phrase))
                        else:
                            objs.append((noun_phrase, root_phrase))
                    
                    undirected_dep_tree = dep_tree.to_undirected()
                
                    for (subj, subj_root), (obj, obj_root) in itertools.product(subjs, objs):
                        global_subj, global_obj = -1, -1
                        
                        subj_id = self.tid2phrase_id[section.section_nlp_global.start + subj.root.i]
                        if subj_id >= 0:
                            global_subj = subj_id
                        else:
                            for coref in section.pron_root2corefs[subj.root.i][::-1]:
                                coref_phrase_id = self.tid2phrase_id[section.section_nlp_global.start + coref.root.i]
                                if coref_phrase_id >= 0:
                                    global_subj = coref_phrase_id
                                    break
                        if global_subj >= 0:
                            new_last_subjs.add(global_subj)
                        else:
                            continue
                        
                        obj_id = self.tid2phrase_id[section.section_nlp_global.start + obj.root.i]
                        if obj_id >= 0:
                            global_obj = obj_id
                        else:
                            for coref in section.pron_root2corefs[obj.root.i][::-1]:
                                coref_phrase_id = self.tid2phrase_id[section.section_nlp_global.start + coref.root.i]
                                if coref_phrase_id >= 0:
                                    global_obj = coref_phrase_id
                                    break
                        if global_obj >= 0:
                            path = [section.section_nlp_local[path_id] for path_id in nx.shortest_path(undirected_dep_tree, source=subj_root.root.i, target=obj_root.root.i)[1:-1]]
                            if dkg.has_edge(global_subj, global_obj, key=SUBJ_OBJ):
                                dkg[global_subj][global_obj][SUBJ_OBJ]['paths'].append(path)
                            else:
                                dkg.add_edge(global_subj, global_obj, key=SUBJ_OBJ, paths=[path], weight=1)
                            for last_subj in last_subjs:
                                if dkg.has_edge(last_subj, global_obj, key=ADJACENT):
                                    dkg[last_subj][global_obj][ADJACENT]['sent_range'].append((last_sent_start + section.section_nlp_global.start, sent.end + section.section_nlp_global.start))
                                else:
                                    dkg.add_edge(last_subj, global_obj, key=ADJACENT, sent_range=[(last_sent_start + section.section_nlp_global.start, sent.end + section.section_nlp_global.start)], weight=2)
        
                last_subjs = new_last_subjs
                last_sent_start = sent.start
                
        # Build edges between phrases based on shared n-grams
        max_n = 5
        n_gram2phrase_ids = defaultdict(lambda: dict[tuple, list]())
        for phrase_id, phrase in enumerate(self.phrases):
            for n in range(1, min(max_n+1, len(phrase)+1)):
                for ngram in ngrams([(tid, token.lemma_) for tid, token in enumerate(phrase) if n > 1 or (token.lemma_ not in stop_words.STOP_WORDS and token.pos_ != 'PUNCT')], n):
                    ngram_tokens = tuple(token for tid, token in ngram)
                    ngram_tids = [tid for tid, token in ngram]
                    if ngram_tokens not in n_gram2phrase_ids[n]:
                        n_gram2phrase_ids[n][ngram_tokens] = []
                    n_gram2phrase_ids[n][ngram_tokens].append((phrase_id, ngram_tids))
                    
        test_pairs = []
        phrase_id_0: int
        phrase_id_1: int
        shared_spans: tuple[Span, Span]
        for n in range(max_n, 0, -1):
            pair_shared_substrings:dict[tuple[int, int], list[tuple]] = defaultdict(list[tuple])
            for ngram, phrases in n_gram2phrase_ids[n].items():
                if len(phrases) > 1:
                    for (phrase_id_0, tids_0), (phrase_id_1, tids_1) in itertools.combinations(phrases, 2):
                        pair_shared_substrings[(phrase_id_0, phrase_id_1)].append((ngram, tids_0, tids_1))
            for (phrase_id_0, phrase_id_1), shared_substrings in pair_shared_substrings.items():
                if n == max_n:
                    if len(shared_substrings) > 1:
                        min_0, max_0, min_1, max_1 = len(self.phrases[phrase_id_0]), 0, len(self.phrases[phrase_id_1]), 0
                        for ngram, tids_0, tids_1 in shared_substrings:
                            min_0 = min(min_0, tids_0[0])
                            max_0 = max(max_0, tids_0[-1])
                            min_1 = min(min_1, tids_1[0])
                            max_1 = max(max_1, tids_1[-1])
                        shared_spans = (self.phrases[phrase_id_0][min_0:max_0+1], self.phrases[phrase_id_1][min_1:max_1+1])
                    else:
                        shared_spans = (self.phrases[phrase_id_0][shared_substrings[0][1][0]:shared_substrings[0][1][-1]+1], self.phrases[phrase_id_1][shared_substrings[0][2][0]:shared_substrings[0][2][-1]+1])
                else:
                    if len(shared_substrings) > 1:
                        continue
                    shared_spans = (self.phrases[phrase_id_0][shared_substrings[0][1][0]:shared_substrings[0][1][-1]+1], self.phrases[phrase_id_1][shared_substrings[0][2][0]:shared_substrings[0][2][-1]+1])
                shared_span = DocManager.strip_ent(shared_spans[0])
                shared_span_str = shared_span.text
                for token in shared_span:
                    if token.text != token.lemma_:
                        shared_span_str = shared_span_str.replace(token.text, token.lemma_, 1)
                test_pairs.append((phrase_id_0, phrase_id_1, shared_span_str, shared_spans))
        
        phrase_id2shared_text2phrase_ids = defaultdict(lambda: defaultdict(list[int]))
        for phrase_id_0, phrase_id_1, shared_span_str, _ in test_pairs:
            phrase_id2shared_text2phrase_ids[phrase_id_0][shared_span_str].append(phrase_id_1)
            
        new_shared_ngram_edges = set[tuple[int, int, str]]()
        for phrase_id_0, shared_text2phrase_ids in phrase_id2shared_text2phrase_ids.items():
            for shared_text, phrase_ids in shared_text2phrase_ids.items():
                phrase_ids.sort()
                for phrase_id_1 in phrase_ids:
                    if self.phrases[phrase_id_0].sent.start != self.phrases[phrase_id_1].sent.start:
                        new_shared_ngram_edges.add((phrase_id_0, phrase_id_1, shared_text))
        
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
        
        for phrase_id_0, phrase_id_1, shared_text, shared_spans in test_pairs:
            if (phrase_id_0, phrase_id_1, shared_text) in new_shared_ngram_edges:
                shared_span_0, shared_span_1 = shared_spans
                shared_weight_0 = tid2score[shared_span_0.start:shared_span_0.end].sum() / tid2score[self.phrases[phrase_id_0].start:self.phrases[phrase_id_0].end].sum()
                shared_weight_1 = tid2score[shared_span_1.start:shared_span_1.end].sum() / tid2score[self.phrases[phrase_id_1].start:self.phrases[phrase_id_1].end].sum()
                shared_weight = (shared_weight_0 * shared_weight_1) ** -0.5 - 1
                dkg.add_edges_from([(phrase_id_0, phrase_id_1, SHARED_TEXT), (phrase_id_1, phrase_id_0, SHARED_TEXT)], weight=shared_weight, shared_text=shared_text)
        
        self.dkg = dkg
    
    # =========================== Doc Helper Functions ===========================
    
    # --------------------------- PyMuPDF Related Functions ---------------------------
    @staticmethod
    def split_block_by_font(block:dict, normal_font:str):
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
                    # try:
                    if span_chars and (span_chars[0].isnumeric() or (span_chars[0].isalpha() and span_chars[0].isupper())) and last_span_chars[-1] not in {';', ':', ','}:
                        if new_line['spans']:
                            new_block['lines'].append(new_line)
                            new_line = copy.deepcopy(line)
                            new_line['spans'] = []
                        blocks.append(new_block)
                        new_block = copy.deepcopy(block)
                        new_block['lines'] = []
                    # except:
                    #     print(last_span_chars, span_chars)
                new_line['spans'].append(span)
                last_span_chars = span_chars
            new_block['lines'].append(new_line)
        
        blocks.append(new_block)
        return blocks
    
    @staticmethod
    def get_block_text(block:dict):
        lines, temp_line = [], ''
        for lid, line in enumerate(block['lines']):
            temp_line += ' '.join(''.join([char['c'] for char in span['chars']]) for span in line['spans'])
            if re.match(r'.*(?<!-)-$', temp_line):
                if lid != len(block['lines']) - 1:
                    temp_line = temp_line[:-1]
            else:
                lines.extend(temp_line.split())
                temp_line = ''
        lines.extend(temp_line.split())
        return ' '.join(lines).replace(' .', '.').replace(' ,', ',').replace(' ?', '?').replace(' !', '!').replace(' :', ':').replace(' ;', ';').replace(' )', ')').replace('( ', '(')

    @staticmethod
    def block_startswith(block:str, prefix:str):
        return ''.join(block.split()).lower().startswith(prefix.lower())
    
    @staticmethod
    def block_endswith(block:str, prefix:str):
        return ''.join(block.split()).lower().endswith(prefix.lower())
    
    def get_main_text_style(self):
        '''Depreciated'''
        size_cnt:dict[str, int] = Counter()
        for page in self.pdf_doc:
            for block in page.get_textpage().extractRAWDICT()['blocks']:
                block_text = DocManager.get_block_text(block)
                if DocManager.block_endswith(block_text, REF_HEADER) or DocManager.block_startswith(block_text, REF_HEADER):
                    break
                for line in block['lines']:
                    for span in line['spans']:
                        size_cnt[(round(span['size'], 0), span['font'])] += len(span['chars'])
        self.main_text_style = size_cnt.most_common()[0][0]
    
    def remove_tables(self):
        for page in self.pdf_doc:
            for table in page.find_tables():
                page.add_redact_annot(table.bbox)
            page.apply_redactions()
    
    def extract_blocks(self):
        text_block_pairs = [(DocManager.get_block_text(sub_block), sub_block)
                      for page in self.pdf_doc 
                      for block in page.get_textpage().extractRAWDICT()['blocks'] 
                      for sub_block in DocManager.split_block_by_font(block, self.main_text_style)]
        self.blocks = [DocBlock(text=t, i=bid) for bid, (t, b) in enumerate(text_block_pairs)]
        self.bid2block = [b for t, b in text_block_pairs]

    # --------------------------- Spacy Related Functions ---------------------------
    @staticmethod
    def strip_ent(noun_chunk:spacy.tokens.Span):
        start, end = 0, 0
        for start in range(len(noun_chunk)):
            if (noun_chunk[start].pos_ not in ['PUNCT', 'DET'] or (noun_chunk[start].text.strip() == '(' and ')' in noun_chunk[start:].text)) and noun_chunk[start].lemma_ not in stop_words.STOP_WORDS:
                break
        for end in range(len(noun_chunk)-1, start-1, -1):
            if (noun_chunk[end].pos_ not in ['PUNCT', 'DET'] or (noun_chunk[end].text.strip() == ')' and '(' in noun_chunk[start:end+1].text)) and noun_chunk[end].lemma_ not in stop_words.STOP_WORDS:
                break
        return noun_chunk[start:end+1]
    
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
        passages = list[langchain_core.documents.Document]()
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
            
    def get_section_for_span(self, span:spacy.tokens.Span) -> OutlineSection | None:
        if self.tid2section_id[span.start] >= 0 and self.tid2section_id[span.end-1] >= 0 and self.tid2section_id[span.start] == self.tid2section_id[span.end-1]:
            return self.sections[self.tid2section_id[span.start]]
        