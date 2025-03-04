from enum import Enum


class Tool:
    TOOL_DESCRIPTION_TEMPLATE = '''{tool_name}({tool_args}) -> {tool_outputs}\nTool description: {tool_description}'''

    def __init__(self, tool_name:str, tool_args:str, tool_outputs:str, tool_description:str):
        self.tool_name = tool_name
        self.tool_args = tool_args
        self.tool_outputs = tool_outputs
        self.tool_description = tool_description
        
    @property
    def prompt(self):
        return self.TOOL_DESCRIPTION_TEMPLATE.format(
            tool_name = self.tool_name.strip(),
            tool_args = self.tool_args.strip(),
            tool_outputs = self.tool_outputs.strip(),
            tool_description = self.tool_description.strip().replace('    ', '')
        )
        
# ---------------------------- retrieval_tools ----------------------------

class RetrieveByName(Tool):
    def __init__(self):
        super().__init__(
            tool_name='''Retrieve_Section_By_Name''',
            tool_args='''section_name: str''',
            tool_outputs='''section_text: str''',
            tool_description='''
            ### Function Description: Retrieve_Section_By_Name

            This function retrieves the text of a specified section from a larger document based on the section name provided.

            #### Parameters:
            - `section_name` (str): The name of the section to retrieve. Be sure to include the section number in this name for accurate identification.

            #### Returns:
            - `section_text` (str): The text contained within the specified section. Note that if the section contains sub-sections, the text of these sub-sections will not be included in the output.'''
        )

class RetrieveByCtrlF(Tool):
    def __init__(self):
        super().__init__(
            tool_name='''Retrieve_Section_By_Ctrl_F''',
            tool_args='''search_text: str, starting_section_name: str (optional), direction: enum['After', 'Before'] (optional)''',
            tool_outputs='''section_name: str, section_text: str''',
            tool_description='''
            ### Function Description: Retrieve_Section_By_Ctrl_F

            This function searches for a specified text within a document and retrieves the text of the first section where the specified text appears, based on given parameters.

            #### Parameters:
            - `search_text` (str): The text you want to find in the document.
            - `starting_section_name` (str, optional): The name of the section where the search begins. If not provided, the search starts from the beginning of the document.
            - `direction` (enum['After', 'Before'], optional): The direction to conduct the search in relation to the `starting_section_name`. By default, it is set to 'After'.

            #### Returns:
            - `section_name` (str): The name of the section containing the `search_text`.
            - `section_text` (str): The complete text of the section where the `search_text` is found.

            This function is useful for locating specific information within a paper and quickly identifying the relevant section based on your search text and parameters. If `starting_section_name` is specified, the search will be conducted from that section in the chosen direction. If not specified, the search will begin from the first section of the document.'''
        )
    
class RetrieveByGroupedCtrlF(Tool):
    def __init__(self):
        super().__init__(
            tool_name='''Retrieve_Section_By_Grouped_Ctrl_F''',
            tool_args='''search_text_set: set[str], starting_section_name: str (optional), direction: enum['After', 'Before'] (optional)''',
            tool_outputs='''concatenated_section_text: str''',
            tool_description='''
            ### Function Description: Retrieve_Section_By_Grouped_Ctrl_F

            This function searches a document to find and concatenate the smallest contiguous sections of text that contain all specified keywords. The search begins from a designated section and proceeds in a specified direction.

            #### Parameters:
            - `search_text_set` (set of str): A set of keywords to search for in the document (e.g., `{'model', 'state-of-the-art', 'performance'}`). Only sections containing all keywords will be considered.
            - `starting_section_name` (str, optional): The name of the section from which the search should begin. If not provided, the search will cover the entire document.
            - `direction` (enum['After', 'Before'], optional): The direction in which to search relative to the `starting_section_name`, either 'After' or 'Before'. Defaults to 'After' if not specified.

            #### Returns:
            - `concatenated_section_text` (str): A string containing the concatenated text of the smallest contiguous sections found following or preceding the starting section, based on the specified direction, that contain all keywords. If any keyword is not found in the document, it will be ignored.

            This function is useful for retrieving sections of a document that are relevant to specific search criteria, while allowing flexibility in defining where and how the search should be conducted.'''
        )
    
class RetrieveByBM25(Tool):
    def __init__(self):
        super().__init__(
            tool_name='''Retrieve_Section_by_BM25''',
            tool_args='''query_text: str''',
            tool_outputs='''section_name: str, section_text: str''',
            tool_description='''
            ### Function Description: Retrieve_Section_by_BM25

            This function identifies and returns the most relevant section of a document based on a given query. Specifically, it utilizes the BM25 algorithm to rank sections by their lexical similarity to the provided query text.

            #### Parameters:
            - `query_text` (str): The input text for which you want to find the most relevant section within the document.

            #### Returns:
            - `section_name` (str): The title or name of the section that is most closely matched to the query according to the BM25 ranking.
            - `section_text` (str): The full content of the section identified as most relevant. This function will retrieve and return this content based on its similarity to the query.

            #### Additional Notes:
            - The function is designed to return the highest-ranked section first. Repeatedly invoking the function will provide subsequent sections in descending order of relevance as determined by the BM25 algorithm.
            - The returned section will closely align in content with your search query, aiding in precise and relevant document navigation.'''
        )
            
class RetrieveByDense(Tool):
    def __init__(self):
        super().__init__(
            tool_name='''Retrieve_Section_by_Dense_Retrieval''',
            tool_args='''query_text: str''',
            tool_outputs='''section_name: str, section_text: str''',
            tool_description='''
            ### Function Description: Retrieve_Section_by_Dense_Retrieval

            This function facilitates the retrieval of a section from a document that is most relevant to the provided search query. The retrieval process utilizes a dense retrieval model, ensuring that the returned section is semantically aligned with the query.

            #### Parameters:
            - `query_text` (str): The search query for which you want to find a matching section within the document.

            #### Returns:
            - `section_name` (str): The title of the section that is ranked highest by the dense retrieval model based on semantic relevance to the query.
            - `section_text` (str): The content of the identified section, providing context and information similar to what the query seeks.

            #### Usage:
            Repeatedly invoking this function with the same query will yield the next highest-ranked section as determined by the dense retrieval model. This allows you to explore various sections that are semantically related to your query.'''
        )
    
class RetrieveByOutline(Tool):
    def __init__(self):
        super().__init__(
            tool_name='''Retrieve_Section_by_Outline''',
            tool_args='''query_text: str''',
            tool_outputs='''section_name: str, section_text: str''',
            tool_description='''
            ### Function Description: Retrieve_Section_by_Outline

            This function is designed to retrieve a section from a document based on a search query, leveraging an LLM agent to select the most relevant section from the document's outline. 

            #### Parameters:
            - `query_text` (str): The search query text used to identify sections of interest within the document.

            #### Returns:
            - `section_name` (str): The name of the section deemed relevant by the LLM agent, potentially indicating the section number or title.
            - `section_text` (str): The full text of the identified section, which the LLM agent predicts may contain information pertinent to the query.

            #### Additional Details:
            The function utilizes an LLM agent with access to the document's outline to determine which section likely contains information relevant to your search query. Repeatedly using this function will iterate over additional sections that the LLM agent identifies as potentially useful, one section at a time. This process continues until no further relevant sections can be identified.'''
        )

class RetrieveByEntity(Tool):
    def __init__(self):
        super().__init__(
            tool_name='''Retrieve_Section_by_Entity''',
            tool_args='''entity: str, starting_section_name: str (optional), direction: enum['After', 'Before'] (optional)''',
            tool_outputs='''section_name: str, section_text: str''',
            tool_description='''
            ### Function Description: Retrieve_Section_by_Entity

            This function searches for a specified entity within a document and retrieves the text of the first section where the specified entity appears, based on given parameters.

            #### Parameters:
            - `entity` (str): The entity you want to find in the document.
            - `starting_section_name` (str, optional): The name of the section where the search begins. If not provided, the search starts from the beginning of the document.
            - `direction` (enum['After', 'Before'], optional): The direction to conduct the search in relation to the `starting_section_name`. By default, it is set to 'After'.

            #### Returns:
            - `section_name` (str): The name of the section containing the `entity`.
            - `section_text` (str): The complete text of the section where the `entity` is found.

            This function is useful for locating specific information within a paper and quickly identifying the relevant section based on your search entity and parameters. If `starting_section_name` is specified, the search will be conducted from that section in the chosen direction. If not specified, the search will begin from the first section of the document.'''
        )
    
class GetEntityConnection(Tool):
    def __init__(self):
        super().__init__(
            tool_name='''Get_Entity_Connection''',
            tool_args='''search_entity_set: set[str]''',
            tool_outputs='''concatenated_text: str''',
            tool_description='''
            ### Function Description: Get_Entity_Connection

            This function extracts and concatenates text from a document to reveal connections between specified entities.

            #### Parameters:
            - `search_entity_set` (set[str]): A set of entities for which you want to discover connections within the document. For example, you might pass a set like `{'model', 'method', 'dataset'}` to explore how these entities are interlinked.

            #### Returns:
            - `concatenated_text` (str): A concatenated string containing the minimal sections of text that demonstrate the relationships between the specified entities. Each subsequent call to this function will return additional text that highlights longer and more varied connections.

            This function is particularly useful for understanding entity relationships in research papers or other structured documents by providing focused insights into how different concepts are related.'''
        )
    
class RetrieveEntityByBM25(Tool):
    def __init__(self):
        super().__init__(
            tool_name='''Retrieve_Entity_by_BM25''',
            tool_args='''search_keyword: str, num: int''',
            tool_outputs='''entities: list[str]''',
            tool_description='''
            ### Function Description: Retrieve_Entity_by_BM25

            This function uses the BM25 algorithm to retrieve the top entities that are lexically similar to a given search keyword. It returns a list of these entities, up to the specified number.

            #### Parameters:
            - `search_keyword` (str): The keyword used to search for similar entities.
            - `num` (int): The maximum number of entities to return.

            #### Returns:
            - `entities` (list[str]): A list of entities that are most similar to the `search_keyword` according to the BM25 algorithm. The list contains up to `num` entities, ranked by their similarity.'''
        )
    
class RetrieveEntityByDense(Tool):
    def __init__(self):
        super().__init__(
            tool_name='''Retrieve_Entity_by_Dense_Retrieval''',
            tool_args='''search_keyword: str, num: int''',
            tool_outputs='''entities: list[str]''',
            tool_description='''
            ### Function Description: Retrieve_Entity_by_Dense_Retrieval

            This function retrieves a list of entities that are semantically similar to a given search keyword using a dense retrieval model. It returns the specified number of top entities that closely match the search keyword in meaning.

            #### Parameters:
            - `search_keyword` (str): The keyword or phrase used to find semantically similar entities.
            - `num` (int): The number of entities to retrieve that are most closely aligned with the search keyword.

            #### Returns:
            - `entities` (list[str]): A list of entities that are semantically related to the search keyword. The list is organized from the most to least similar according to the dense retrieval model.'''
        )
    
class RetrieveEntityByLLM(Tool):
    def __init__(self):
        super().__init__(
            tool_name='''Retrieve_Entity_by_LLM''',
            tool_args='''search_keyword: str, num: int''',
            tool_outputs='''entities: list[str]''',
            tool_description='''
            ### Function Description: Retrieve_Entity_by_LLM

            This function retrieves a list of entities related to a specified keyword using a large language model. The entities returned are determined by the model's internal understanding and knowledge.

            #### Parameters:
            - `search_keyword` (str): The keyword used to search for relevant entities.
            - `num` (int): The number of entities to be returned.

            #### Returns:
            - `entities` (list[str]): A list of entities that are relevant to the provided search keyword. The list contains up to the specified number of entities, ranked by relevance as determined by the large language model.'''
        )

# ---------------------------- relation_extraction_tools ----------------------------

class RelationTripleMatching(Tool):
    def __init__(self):
        super().__init__(
            tool_name='''Relation_Triple_Matching''',
            tool_args='''head_entity_type: str, relation: str, tail_entity_type: str, context: str''',
            tool_outputs='''relation_triples: list[tuple[str, str, str]]''',
            tool_description='''
            ### Function Description: Relation_Triple_Matching

            This function identifies and extracts knowledge triples from a given context based on a specified relationship pattern. A knowledge triple is composed of a head entity, a relation, and a tail entity.

            #### Parameters:
            - `head_entity_type` (str): The specific type or category of the head entity you are interested in. This helps in filtering the entities in the context to match the desired pattern.
            - `relation` (str): The relation or connection that exists between the head and the tail entities. This is the key criterion for identifying the desired triples within the context.
            - `tail_entity_type` (str): The specific type or category of the tail entity you want to match. Like the head entity type, this helps in narrowing down the entities to extract relevant triples.
            - `context` (str): A body of text from which the function identifies and extracts the knowledge triples. This text provides the information needed to recognize entities and their relationships.

            #### Returns:
            - `relation_triples` (list[tuple[str, str, str]]): A list of tuples, where each tuple contains a matched knowledge triple in the format ("head entity", "relation", "tail entity"). This list includes all the extracted triples from the context that conform to the specified relationship pattern.'''
        )
    
class TailEntityMatching(Tool):
    def __init__(self):
        super().__init__(
            tool_name='''Tail_Entity_Matching''',
            tool_args='''head_entity: str, relation: str, tail_entity_type: str, context: str''',
            tool_outputs='''tail_entities: list[str]''',
            tool_description='''
            ### Function Description: Tail_Entity_Matching

            This function identifies and extracts all tail entities from a given piece of context that align with a specified relationship pattern. 

            #### Parameters:
            - `head_entity` (str): The initial entity in the relationship pattern.
            - `relation` (str): The type of relationship linking the head entity with the tail entity.
            - `tail_entity_type` (str): The specific type or category of the target tail entity.
            - `context` (str): A text segment from which the tail entities will be extracted based on the relationship pattern.

            #### Returns:
            - `tail_entities` (list[str]): A list of all tail entities found in the context that fit the pattern of "head_entity, relation, tail_entity_type."'''
        )

class HeadEntityMatching(Tool):
    def __init__(self):
        super().__init__(
            tool_name='''Head_Entity_Matching''',
            tool_args='''head_entity_type: str, relation: str, tail_entity: str, context: str''',
            tool_outputs='''head_entities: list[str]''',
            tool_description='''
            ### Function Description: Head_Entity_Matching

            This function identifies and extracts all head entities from a given context that match a specified relationship pattern consisting of a head entity type, relation, and tail entity.

            #### Parameters:
            - `head_entity_type` (str): Specifies the type or category of the head entity you want to match.
            - `relation` (str): The name of the relationship that links the head entity to the tail entity.
            - `tail_entity` (str): The specific tail entity involved in the relationship.
            - `context` (str): A piece of text in which the function will search for matching head entities according to the defined relationship pattern.

            #### Returns:
            - `head_entities` (list[str]): A list of all head entities from the context that align with the given pattern of "head_entity_type, relation, tail_entity."'''
        )
    
class RelationLabelExtraction(Tool):
    def __init__(self):
        super().__init__(
            tool_name='''Relation_Label_Extraction''',
            tool_args='''head_entity: str, tail_entity: str, context: str''',
            tool_outputs='''relations: list[str]''',
            tool_description='''
            ### Function Description: Relation_Label_Extraction

            This function identifies and extracts all relational associations between a specified head entity and a tail entity found within a given context.

            #### Parameters:
            - `head_entity` (str): The specific primary entity whose relationships you want to explore.
            - `tail_entity` (str): The specific secondary entity that you want to find in relation to the head entity.
            - `context` (str): The text body within which the relationships are to be identified.

            #### Returns:
            - `relations` (list[str]): A list of relationship labels describing how the head entity is connected to the tail entity within the provided context. Each relationship label corresponds to a distinct "head_entity, relation, tail_entity" triple identified in the context.'''
        )
    
class EntityExtraction(Tool):
    def __init__(self):
        super().__init__(
            tool_name='''Entity_Extraction''',
            tool_args='''entity_type: str, context: str''',
            tool_outputs='''entities: list[str]''',
            tool_description='''
            ### Function Description: Entity_Extraction

            This function extracts entities of a specified type from a given piece of context text.

            #### Parameters:
            - `entity_type` (str): The type of entity to extract from the context. This could be, for example, "person," "location," "date," etc.
            - `context` (str): The text from which to extract the entities. It should contain the context necessary to identify the specified entities.

            #### Returns:
            - `entities` (list[str]): A list of entities that match the specified entity type, extracted from the context. Each entity is represented as a string.'''
        )
    
class AttributeExtraction(Tool):
    def __init__(self):
        super().__init__(
            tool_name='''Attribute_Extraction''',
            tool_args='''entity: str, attribute_name: str, context: str''',
            tool_outputs='''attribute_value: str''',
            tool_description='''
            ### Function Description: Attribute_Extraction

            This function extracts the value of a specified attribute for a given entity from a provided piece of context text.

            #### Parameters:
            - `entity` (str): The name of the entity whose attribute value you wish to extract.
            - `attribute_name` (str): The name of the attribute whose value you are interested in retrieving.
            - `context` (str): The context text from which the function will extract the attribute value.

            #### Returns:
            - `attribute_value` (str): The extracted value of the specified attribute for the given entity as found in the context.'''
        )

# ---------------------------- question_analysis_tools ----------------------------

class QuestionRelationDecomposition(Tool):
    def __init__(self):
        super().__init__(
            tool_name='''Question_Relation_Decomposition''',
            tool_args='''question: str, known_information: str (optional)''',
            tool_outputs='''relation_tuple: tuple[str, str, str, str, str]''',
            tool_description='''
            ### Function Description: Question_Relation_Decomposition

            This function is designed to analyze a given question and optionally incorporate known information to extract a relational tuple. This tuple will help in identifying specific entities and their relationships within the question's context, thereby refining and narrowing down the question. The function returns a tuple following the format: "head entity, head entity type, relation, tail entity, tail entity type."

            #### Parameters:
            - `question` (str): The text of the question to be analyzed.
            - `known_information` (str, optional): Additional context or data relevant to the question, which can assist in accurately determining the entities and their relationships.

            #### Returns:
            - `relation_tuple` (tuple[str, str, str, str, str]): A five-element tuple where each item represents:
            - `head entity` (str): The main entity involved in the question. This may be an empty string if not directly identifiable from the input.
            - `head entity type` (str): The category or type of the head entity.
            - `relation` (str): The type of relationship connecting the head and tail entities.
            - `tail entity` (str): The secondary entity involved in the relationship. This may also be an empty string if not clearly identifiable.
            - `tail entity type` (str): The category or type of the tail entity.'''
        )
    
class QuestionAttributeDecomposition(Tool):
    def __init__(self):
        super().__init__(
            tool_name='''Question_Attribute_Decomposition''',
            tool_args='''question: str, known_information: str (optional)''',
            tool_outputs='''attribute_tuple: tuple[str, str, str]''',
            tool_description='''### Function Description: Question_Attribute_Decomposition

            This function analyzes a given question and any additional known information to identify an attribute value necessary for resolving the question. The attribute is presented in the format "entity, entity type, attribute name". The entity type describes the classification of the entity, and if the specific entity is not identifiable from the question or known information, the entity may be returned as an empty string.

            #### Parameters:
            - `question` (str): The text of the question that needs analysis to determine which attribute to extract.
            - `known_information` (str, optional): Any relevant information related to the question that may assist in identifying the required attribute.

            #### Returns:
            - `attribute_tuple` (tuple[str, str, str]): A tuple containing three elements:
            1. `entity` (str): The specific entity related to the attribute, or an empty string if the entity is not determined.
            2. `entity_type` (str): The type or category of the entity.
            3. `attribute_name` (str): The name of the attribute that needs to be extracted to assist in answering the question. 

            This function is useful for breaking down complex questions into more manageable components by identifying key attributes necessary for further analysis.'''
        )
    
# ---------------------------- evaluation_tools ----------------------------

class RelationChecking(Tool):
    def __init__(self):
        super().__init__(
            tool_name='''Relation_Checking''',
            tool_args='''head_entity: str, relation: str, tail_entity: str, context: str''',
            tool_outputs='''is_exist: bool''',
            tool_description='''
            ### Function Description: Relation_Checking

            This function determines if a specified relationship exists between two entities within a given context of text.

            #### Parameters:
            - `head_entity` (str): The initial entity involved in the relationship. This is the starting point of the relationship being assessed.
            - `relation` (str): The type or name of the relationship you are checking for.
            - `tail_entity` (str): The second entity involved in the relationship. This entity is the endpoint of the relationship being assessed.
            - `context` (str): A piece of text within which the existence of the specified relationship is to be verified.

            #### Returns:
            - `is_exist` (bool): A boolean value indicating whether the specified relationship between the `head_entity` and `tail_entity` is present in the `context`. Returns `True` if the relationship exists, otherwise `False`.'''
        )

class InformationChecking(Tool):
    def __init__(self):
        super().__init__(
            tool_name='''Information_Checking''',
            tool_args='''question: str, known_information: str''',
            tool_outputs='''is_enough: bool''',
            tool_description='''
            ### Function Description: Information_Checking

            This function evaluates whether the available known information is sufficient to answer a given question.

            #### Parameters:
            - `question` (str): The text of the question you are seeking to answer.
            - `known_information` (str): The text of the available information that is used to determine if it sufficiently addresses the question.

            #### Returns:
            - `is_enough` (bool): A boolean value indicating whether the known information is adequate to fully answer the question. If the information is sufficient, the function returns `True`; otherwise, it returns `False`.'''
        )

class TakeNote(Tool):
    def __init__(self):
        super().__init__(
            tool_name='''Take_Note''',
            tool_args='''new_information: Any, known_information: str''',
            tool_outputs='''''',
            tool_description='''
            ### Function Description: Take_Note

            This function is designed to integrate new information into an existing body of knowledge. It takes the input of any new information extracted from a document, converts it into a string format, and appends it to a given string of known information. This updated string serves as a consolidated source of information that can be utilized by other functions in the system.

            #### Parameters:
            - `new_information` (Any): The new information extracted from the document that you wish to incorporate.
            - `known_information` (str): A string containing the pre-existing information. The new information will be appended to this string.

            #### Functionality:
            The function ensures that any form of new information, regardless of its original format, is converted into a string before being appended to the existing string of known information. This seamless integration aids in maintaining a complete and accessible record for subsequent use by other components or functions within the application.'''
        )

    
class ToolSet(Enum):
    '''
    Variable types:
        - Question
            - As input: retrieval_tools, question_rewrite_tools, analysis_tools, evaluation_tools
            - As output: question_rewrite_tools
            
        - Section/Paragraph
            - As input: analysis_tools, evaluation_tools
            - As output: retrieval_tools
            
        - Section name
            - As input: retrieval_tools
            - As output: retrieval_tools
            
        - Entity
            - As input: retrieval_tools, relation_extraction_tools, evaluation_tools
            - As output: 
            
        - Entity type
            - As input: relation_extraction_tools, analysis_tools, evaluation_tools
            - As output: analysis_tools
            
        - Attribute value
            - As input: retrieval_tools, relation_extraction_tools, evaluation_tools
            - As output: 
            
        - Attribute name
            - As input: relation_extraction_tools, analysis_tools, evaluation_tools
            - As output: analysis_tools
            
        - Relation
            - As input: relation_extraction_tools, evaluation_tools
            - As output: analysis_tools
            
        - Free text
            - As input: retrieval_tools, question_rewrite_tools, evaluation_tools
            - As output: analysis_tools
            
        - Boolean
            - As input: 
            - As output: evaluation_tools
    
    Tool types:
        - retrieval_tools
            - input: Question, Section name, Entity, Free text
            - output: Section, Section name, Entity
        - relation_extraction_tools
            - input: Entity, Entity type, Relation, Section
            - output: Entity, Relation
        - question_analysis_tools
            - input: Question, Free text
            - output: Question, Entity, Entity type, Relation
        - context_analysis_tools
            - input: Section, Entity type, Question
            - output: Free text, Entity type, Relation
        - evaluation_tools
            - input: Question, Section, Entity, Entity type, Relation, Free text
            - output: Boolean
    '''
    
    # ---------------------------- retrieval_tools ----------------------------
    
    
    retrieve_by_name = RetrieveByName()
    
    retrieve_by_ctrl_f = RetrieveByCtrlF()

    retrieve_by_grouped_ctrl_f = RetrieveByGroupedCtrlF()

    retrieve_by_bm25 = RetrieveByBM25()

    retrieve_by_dense = RetrieveByDense()

    retrieve_by_outline = RetrieveByOutline()
    
    retrieve_by_entity = RetrieveByEntity()
    
    get_entity_connection = GetEntityConnection()
    
    retrieve_entity_by_bm25 = RetrieveEntityByBM25()

    retrieve_entity_by_dense = RetrieveEntityByDense()

    retrieve_entity_by_llm = RetrieveEntityByLLM()

    # ---------------------------- relation_extraction_tools ----------------------------
    
    relation_triple_matching = RelationTripleMatching()
    
    tail_ent_matching = TailEntityMatching()

    head_ent_matching = HeadEntityMatching()
    
    relation_label_extraction = RelationLabelExtraction()
    
    entity_extraction = EntityExtraction()
    
    attribute_extraction = AttributeExtraction()
    
    # ---------------------------- question_analysis_tools ----------------------------
    
    question_relation_decomposition = QuestionRelationDecomposition()
    
    question_attribute_decomposition = QuestionAttributeDecomposition()
    
    # ---------------------------- context_analysis_tools ----------------------------
    
    # class QuestionAttributeDecomposition(Tool):
    #     def __init__(self):
    #         super().__init__(
    #             tool_name='''Question_Attribute_Decomposition''',
    #             tool_args='''question: str, known_information: str (optional)''',
    #             tool_outputs='''attribute_tuple: tuple[str, str, str]''',
    #             tool_description='''Input the question text (question) and any known information (known_information) from the paper, this function will analyze the question and determine the next attribute value in the format of "entity, entity type, attribute name" that need to be extracted from the context in order to help solving the question. The entity type specifies the type of the entity and the entity may be empty string if no exact entity is known in the input question and known information.'''
    #         )
    
    # question_attribute_decomposition = QuestionAttributeDecomposition()
    
    # ---------------------------- evaluation_tools ----------------------------
    
    relation_checking = RelationChecking()
    
    information_checking = InformationChecking()
    
    take_note = TakeNote()
    


tool_description_prompt = '''You are provided with the following tools:

Passage Retrieval Tools:

'''