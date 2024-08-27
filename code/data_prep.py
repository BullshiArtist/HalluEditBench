import pandas as pd
# from fuzzywuzzy import fuzz
from bs4 import BeautifulSoup
from nltk import ngrams
import concurrent.futures
from nltk.util import ngrams
import transformers
from concurrent.futures import ThreadPoolExecutor
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer
import nltk
import time
# from gingerit.gingerit import GingerIt
import numpy as np
import torch
from flair.models import SequenceTagger
from flair.data import Sentence
from sentence_transformers import SentenceTransformer, util
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import re
import os
import sys
import json
import spacy
import random
import openai
import pickle
import logging
import asyncio
import backoff
import requests
import tiktoken
import argparse
import networkx as nx
from SPARQLWrapper import SPARQLWrapper, JSON
# from neo4j import GraphDatabase
# from googletrans import Translator


class OutOfQuotaException(Exception):
    "Raised when the key exceeded the current quota"
    def __init__(self, key, cause=None):
        super().__init__(f"No quota for key: {key}")
        self.key = key
        self.cause = cause

    def __str__(self):
        if self.cause:
            return f"{super().__str__()}. Caused by {self.cause}"
        else:
            return super().__str__()


class AccessTerminatedException(Exception):
    "Raised when the key has been terminated"
    def __init__(self, key, cause=None):
        super().__init__(f"Access terminated key: {key}")
        self.key = key
        self.cause = cause

    def __str__(self):
        if self.cause:
            return f"{super().__str__()}. Caused by {self.cause}"
        else:
            return super().__str__()


class FactChecker:
    def __init__(self, out_file_name, out_file_path, temperature, mode, hops, diversify_method,  \
                 topics_path, neo4j_uri, neo4j_username, neo4j_password, check_grammar_flag, AI_model, \
                    only_evaluate, read_graph, graph_path, limit_nodes, yes_no_number, MC_number, wh_number, use_icl) -> None:
        
        # self.out_file_name = out_file_name
        self.out_file_name = topics_path.split("/")[-1].replace(".json", "")
        self.out_file_path = out_file_path
        self.temperature = temperature
        self.mode = mode
        self.nlp_model_1 = SequenceTagger.load('ner')
        self.nlp_model_2 = spacy.load("en_core_web_lg")
        # self.api_keys = json.load(open('../apikey.json'))
        # self.parser = GingerIt()
        self.hops = hops
        self.diversify_method = diversify_method
        self.sparql = SPARQLWrapper("https://query.wikidata.org/sparql")
        self.neo4j_uri = neo4j_uri
        self.neo4j_username = neo4j_username
        self.neo4j_password = neo4j_password
        with open(topics_path, 'r', encoding='utf-8') as topics_file:
            self.topics = topics_file.readlines()
        self.topic = json.loads(self.topics[0])
        # assert self.valid_location(), "Bad location"
        self.check_grammar_flag = check_grammar_flag
        self.AI_model = AI_model
        self.evaluate = only_evaluate
        self.limit_nodes = limit_nodes
        self.read_graph = read_graph
        self.graph_path = graph_path
        self.yes_no_number = yes_no_number
        self.MC_number = MC_number
        self.wh_number = wh_number
        self.use_icl = use_icl
        self.device = 5 #device#torch.device("cuda" if torch.cuda.is_available() else "cpu")


    def remove_additional_spaces(self, text):
        pattern = r'\s{3,}'  
        modified_text = re.sub(pattern, ' ', text)
        return modified_text
    

    def identifier_conversion(self, entity, property=False):
        if not property:
            query = f"""
                SELECT ?identifier WHERE {{
                    ?identifier rdfs:label "{entity}"@en.
                }}
                """
        else:
            query = f"""
                SELECT ?identifier WHERE {{
                    ?property rdf:type wikibase:Property .
                    ?identifier rdfs:label "{entity}"@en.
                }}
                """
        property_pattern = r'^P\d+'
        node_pattern = r'^Q\d+'
        
        self.sparql.setQuery(query)
        self.sparql.setReturnFormat(JSON)
        results = self.sparql.query().convert()
        if "results" in results and "bindings" in results["results"]:
            if not property:
                for result in results["results"]["bindings"]:
                    identifier = result["identifier"]["value"].split("/")[-1]
                    if re.match(node_pattern, identifier):
                        return identifier
            else:
                for result in results["results"]["bindings"]:
                    identifier = result["identifier"]["value"].split("/")[-1]
                    if re.match(property_pattern, identifier):
                        return identifier
        return None


    def convert_topic_to_symbol(self, topic_dict):
        try:
            relation_object_pairs = []
            for key, value in topic_dict.items():
                key = self.identifier_conversion(key, True)
                value = self.identifier_conversion(value)
                if key and value:
                    relation_object_pairs.append([key, value])
                else:
                    raise Exception(f"'{key}: {value}' cannot be converted to identifier!")
            return relation_object_pairs
        except Exception as e:
            print("Error:", e)
            sys.exit(1)


    def process_result(self, result):
        subject_label = result["subjectLabel"]["value"]
        relation_label = result["relation"]["value"]
        try:
            reference_response = requests.get(relation_label)
            reference_soup = BeautifulSoup(reference_response.content, 'html.parser')
            relation_label = reference_soup.find("span", class_="wikibase-title-label")
        except requests.exceptions.RequestException as e:
            # Handle the connection error
            print(f"Connection error occurred for relation '{relation_label}': {e}")
            return None
        object_label = result["objectLabel"]["value"]

        return {
            "subjectLabel": subject_label,
            "relation": relation_label.text,
            "objectLabel": object_label
        }


    def fact_triplets_retrival(self):
        data = []
        print("====Start retrieving fact triplets!====")
        for topic in self.topics:
            if topic:
                topic = json.loads(topic)
                print(topic)
                query_part1 = "SELECT ?subjectLabel ?relation ?objectLabel WHERE {"
                query_part2 = ""
                relation_object_pairs = self.convert_topic_to_symbol(topic)
                for pair in relation_object_pairs:
                    query_part2 += f"\n?subject wdt:{pair[0]} wd:{pair[1]} ."

                query_part3 = """
                    ?subject  ?relation  ?object.
                    ?subject wikibase:identifiers ?subject_identifierCount.
                    ?object wikibase:identifiers ?object_identifierCount.
                    """
    
                query_part5 = """ 
                    FILTER (?subject_identifierCount >= 8 && ?object_identifierCount >= 5) .  
                    SERVICE wikibase:label { bd:serviceParam wikibase:language "[AUTO_LANGUAGE],en". }
                }
                LIMIT 50000
                """
                query = query_part1 + query_part2 + query_part3 + query_part5
                # Set the SPARQL query and response format
                self.sparql.setQuery(query)
                self.sparql.setReturnFormat(JSON)

                # Execute the SPARQL query
                results = self.sparql.query().convert()
                if "results" in results:
                    # Create a list to store the data
                    with ThreadPoolExecutor() as executor:
                        futures = [executor.submit(self.process_result, result) for result in results["results"]["bindings"]]
                        # Use tqdm to show the progress bar while waiting for all tasks to complete
                        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
                            result_data = future.result()
                            if result_data:
                                data.append(result_data)

        self.fact_triplets = pd.DataFrame(data)
        print("====Finished retrieving fact triplets!====")


    def graph_creating(self, encoding="utf-8"):
        self.fact_triplets.dropna(inplace=True)
        condition1 = self.fact_triplets.apply(lambda row: any(val.startswith('Q') and val[1:].isdigit() for val in row.values), axis=1)
        condition2 = self.fact_triplets.apply(lambda row: any(val.startswith('http') for val in row.values), axis=1)
        
        self.fact_triplets = self.fact_triplets[~(condition1 | condition2)]  
        # Create a directed graph
        self.directed_graph = nx.DiGraph()
        for index, row in self.fact_triplets.iterrows():
            node = row[0]
            node = self.remove_additional_spaces(node)
            self.directed_graph.add_node(node.replace('\n', ''))
            edge = self.remove_additional_spaces(row[1])
            adjacent_node = self.remove_additional_spaces(row[2])
            self.directed_graph.add_node(adjacent_node.replace('\n', ''))
            self.directed_graph.add_edge(node.replace('\n', ''), adjacent_node.replace('\n', ''), label=edge)   
        file_name = self.out_file_name.split("_")[0]
        folder_name = "graph"
        # Check if the folder exists
        if not os.path.exists(folder_name):
            # Create the folder
            os.makedirs(folder_name)
            print(f"Folder '{folder_name}' created successfully.")
        
        nx.write_gpickle(self.directed_graph, 'graph/'+file_name + '_graph' + '.gpickle')


    def read_graph_file(self):
        # self.directed_graph = nx.read_gpickle(self.graph_path)
        if self.graph_path:
            with open(self.graph_path, 'rb') as f:
                self.directed_graph = pickle.load(f)


    def visualise_fact_graph(self):        
        # Connect to Neo4j
        driver = GraphDatabase.driver(self.neo4j_uri, auth=(self.neo4j_username, self.neo4j_password))
        with driver.session() as session:
            # Delete existing graph
            session.run("MATCH (n) DETACH DELETE n")
            
            # Import nodes           
            for node in self.directed_graph.nodes():
                session.run("CREATE (:Node {id: $id})", id=node)
            
            # Import relationships
            for edge in self.directed_graph.edges(data=True):
                source = edge[0]
                target = edge[1]
                label = edge[2]['label']
                session.run("MATCH (src:Node {id: $source}), (tgt:Node {id: $target}) "
                            "CREATE (src)-[:RELATIONSHIP_TYPE {label: $label}]->(tgt)",
                            source=source, target=target, label=label)
        driver.close()
        print("Graph data imported into Neo4j successfully!")


    def yes_no_question_generation(self, subject, relation, object, answer, multi_hops=1):
        if multi_hops == 1:
            subject_doc = self.nlp_model_2(relation)
        else:
            subject_doc_list = []
            for i in range(multi_hops):
                tmp_subject_doc = self.nlp_model_2(relation['edge_labels'][i])
                subject_doc_list.append(tmp_subject_doc)

        question_answer_pair = {}
        question_answer_pair["type"] = "yes_no"
        question_answer_pair["subject"] = subject
        question_answer_pair["relation"] = relation
        question_answer_pair["object"] = object

        if multi_hops > 1:
            for i in range(multi_hops-1):
                if subject_doc_list[i][0].tag_ in ["VBN", "VBD", "VB", "VBZ", "JJ"]:
                    return None
            subject_doc = subject_doc_list[-1]
            
        relation_set = set()
        for token in subject_doc:
            relation_set.add(token.tag_)

        if multi_hops == 1:
            if subject_doc[-1].tag_ == "IN" and subject_doc[0].tag_ not in ["VBN", "VBD", "VB", "VBZ"]:
                return None
        else:
            for i in range(multi_hops):
                if subject_doc_list[i][-1].tag_ == "IN" and subject_doc_list[i][0].tag_ not in ["VBN", "VBD", "VB", "VBZ"]:
                    return None
            
        if subject_doc[0].tag_ == "VBN" and subject_doc[-1].tag_ == "IN" and all(token.tag_ not in ["NN", "NNP", "NNPS", "NNS"] for token in subject_doc[0:]):
            if multi_hops == 1:
                question_answer_pair["question"] = "Was " + subject + " " + relation + " " + object +"?"
                question_answer_pair["label"] = answer
            else:
                question_answer_pair["question"] = "Was " + subject 
                for i in range(multi_hops-1):
                    question_answer_pair["question"] += "'s " + relation['edge_labels'][i] 
                question_answer_pair["question"] += " " + relation['edge_labels'][-1] + " " + object +"?"
                question_answer_pair["label"] = answer
        elif subject_doc[0].tag_ == "JJ" and subject_doc[-1].tag_ == "IN" and all(token.tag_ not in ["NN", "NNP", "NNPS", "NNS"] for token in subject_doc[0:]):
            if multi_hops == 1:
                question_answer_pair["question"] = "Is " + subject + " " + relation + " " + object +"?"
                question_answer_pair["label"] = answer
            else:
                question_answer_pair["question"] = "Is " + subject 
                for i in range(multi_hops-1):
                     question_answer_pair["question"] += "'s " + relation['edge_labels'][i] 
                question_answer_pair["question"] += " " + relation['edge_labels'][-1] + " " + object +"?"
                question_answer_pair["label"] = answer
        elif subject_doc[0].tag_ == "VBD" and subject_doc[-1].tag_ not in ["NN", "NNP", "NNPS", "NNS"]:
            if multi_hops == 1:
                question_answer_pair["question"] = "Did " + subject + " "
            else:
                question_answer_pair["question"] = "Did " + subject 
                for i in range(multi_hops-1):
                    question_answer_pair["question"] += "'s " + relation['edge_labels'][i] 
                question_answer_pair["question"] += " "
            for token in subject_doc:
                if token.tag_ == "VBD":
                    question_answer_pair["question"] += token.lemma_ + " "
                else:
                    question_answer_pair["question"] += token.text + " "
            question_answer_pair["question"] = question_answer_pair["question"] + object +"?" 
            question_answer_pair["label"] = answer
        elif (subject_doc[0].tag_ == "VB" or subject_doc[0].tag_ == "VBZ") and subject_doc[-1].tag_ not in ["NN", "NNP", "NNPS", "NNS"]:
            if multi_hops == 1:
                question_answer_pair["question"] = "Does " + subject + " "
            else:
                question_answer_pair["question"] = "Does " + subject 
                for i in range(multi_hops-1):
                    question_answer_pair["question"] += "'s " + relation['edge_labels'][i] 
                question_answer_pair["question"] += " "
            for token in subject_doc:
                if token.tag_ == "VBZ":
                    question_answer_pair["question"] += token.lemma_ + " "
                else:
                    question_answer_pair["question"] += token.text + " "
            question_answer_pair["question"] = question_answer_pair["question"] + object +"?" 
            question_answer_pair["label"] = answer
        elif (subject_doc[-1].tag_ == "NN" or subject_doc[-1].tag_ == "NNP") and subject_doc[0].tag_ not in ["VB", "VBZ", "VBD"]: 
            if multi_hops == 1:
                question_answer_pair["question"] = "Is " + object + " the " + relation + " of " + subject + "?"
                question_answer_pair["label"] = answer
            else:
                question_answer_pair["question"] = "Is " + object + " the " + relation['edge_labels'][-1] + " of " + subject 
                for i in range(multi_hops-1):
                    question_answer_pair["question"] += "'s " + relation['edge_labels'][i] 
                question_answer_pair["question"] += "?"
                question_answer_pair["label"] = answer      
        else:
            return None
        return question_answer_pair


    def MC_question_generation(self, subject, relation, object, disturb1, disturb2, disturb3, multi_hops=1, query_subject=False):
        question_answer_pair = self.wh_question_generation(subject, relation, object, multi_hops, query_subject)
        if not question_answer_pair:
            return None

        question_answer_pair["type"] = "MC"
        question_answer_pair["subject"] = subject
        question_answer_pair["relation"] = relation
        question_answer_pair["object"] = object

        MC_dict = {"0": "A", "1": "B", "2": "C", "3": "D"}
        choice = [] 
        choice.append(object)
        choice.append(disturb1) 
        choice.append(disturb2)
        choice.append(disturb3)
        random.shuffle(choice)
        correct_answer = MC_dict[str(choice.index(object))]
        choice_str = ""
        for i in range(4):
            choice_str += (MC_dict[str(i)] + ". " + choice[i] + "  ")
        question_answer_pair["question"] = question_answer_pair["question"] + " \t\t" + choice_str
        question_answer_pair["label"] = correct_answer
        return question_answer_pair


    def wh_question_generation(self, subject, relation, object, multi_hops=1, query_subject = False):
        object_type1 = None
        object_type2 = None
        object_type = None
        discard_flag = False
        convert_dict1 = {
            "PER": "PERSON",
            "LOC": "GPE"
            }

        ####### method 1
        sentence = Sentence(object)
        # Predict entities
        self.nlp_model_1.predict(sentence)
        # Access entity annotations
        entities = sentence.get_spans('ner')
        # Print the recognized entities
        if entities:
            object_type1 = entities[0].tag
            if object_type1 == "PER" or object_type1 == "LOC":
                object_type1 = convert_dict1[object_type1]
            else:
                object_type1 = None

        ####### method 2
        object_doc = self.nlp_model_2(object)
        if object_doc.ents:
            object_type2 = object_doc.ents[0].label_

        if object_type1:        
            if object_type1 == object_type2:
                object_type = object_type1
            else:
                discard_flag = True
        else:
            if object_type2 != "GPE" and object_type2 != "PERSON":
                object_type = object_type2
            else:
                discard_flag = True
                
        if discard_flag:
            return None

        if multi_hops == 1:
            subject_doc = self.nlp_model_2(relation)
        else:
            subject_doc_list = []
            for i in range(multi_hops):
                tmp_subject_doc = self.nlp_model_2(relation['edge_labels'][i])
                subject_doc_list.append(tmp_subject_doc)

        if multi_hops > 1:
            if not query_subject:
                for i in range(multi_hops-1):
                    if subject_doc_list[i][0].tag_ in ["VBN", "VBD", "VB", "VBZ", "JJ"]:
                        return None
            else:
                for i in range(1, multi_hops):
                    if subject_doc_list[i][0].tag_ in ["VBN", "VBD", "VB", "VBZ", "JJ"]:
                        return None
            if not query_subject:
                subject_doc = subject_doc_list[-1]
            else:
                subject_doc = subject_doc_list[0]

        if multi_hops == 1:
            if subject_doc[-1].tag_ == "IN" and subject_doc[0].tag_ not in ["VBN", "VBD", "VB", "VBZ"]:
                return None
        else:
            for i in range(multi_hops):
                if subject_doc_list[i][-1].tag_ == "IN" and subject_doc_list[i][0].tag_ not in ["VBN", "VBD", "VB", "VBZ"]:
                    return None
            
        question_answer_pair = {}
        question_answer_pair["type"] = "wh"
        question_answer_pair["subject"] = subject
        question_answer_pair["relation"] = relation
        question_answer_pair["object"] = object

        relation_set = set()
        for token in subject_doc:
            relation_set.add(token.tag_)

        object_to_interrogative = {
            "PERSON": "Who",
            "DATE": "When",
        }

        default_interrogative = "What"  # Default value      
        interrogative = object_to_interrogative.get(object_type, default_interrogative)
        if query_subject:
            tmp = subject
            subject = object
            object = tmp
        if subject_doc[0].tag_ == "VBN" and subject_doc[-1].tag_ == "IN" and all(token.tag_ not in ["NN", "NNP", "NNPS", "NNS"] for token in subject_doc[0:]):
            if not query_subject:
                if multi_hops == 1:
                    question_answer_pair["question"] = interrogative + " was " + subject + " " + relation + "?"
                    question_answer_pair["label"] = object
                else:
                    question_answer_pair["question"] = interrogative + " was " + subject 
                    for i in range(multi_hops-1):
                        question_answer_pair["question"] += "'s " +  relation['edge_labels'][i] 
                    question_answer_pair["question"] += " "  + relation['edge_labels'][-1] + "?"
                    question_answer_pair["label"] = object
            else:
                if multi_hops == 1:
                    if object_type != "PERSON":
                        first_pair = next(iter(self.topic.items()))
                        if first_pair[1] != "revolution":
                            interrogative = "Which " + first_pair[1]
                        else:
                            interrogative = "Which revolution or war"
                    
                    question_answer_pair["question"] = interrogative + " was " + relation + " " + object + "?"
                    question_answer_pair["label"] = subject
                else:
                    first_pair = next(iter(self.topic.items()))
                    question_answer_pair["question"] = interrogative 
                    for i in range(multi_hops-1, 0, -1):
                        if first_pair[1] == "human" and i == multi_hops-1:
                            question_answer_pair["question"] += "se " +  relation['edge_labels'][i]
                        elif i == multi_hops-1:
                            first_pair = next(iter(self.topic.items()))
                            if first_pair[1] != "revolution":
                                interrogative = "Which " + first_pair[1]
                            else:
                                interrogative = "Which revolution or war"
                        else:
                            question_answer_pair["question"] += " " +  relation['edge_labels'][i]
                    question_answer_pair["question"] += " was " + relation['edge_labels'][0] + " " + object + "?"
                    question_answer_pair["label"] = subject
        elif subject_doc[0].tag_ == "JJ" and subject_doc[-1].tag_ == "IN" and all(token.tag_ not in ["NN", "NNP", "NNPS", "NNS"] for token in subject_doc[0:]):
            if not query_subject:
                if multi_hops == 1:
                    question_answer_pair["question"] = interrogative + " is " + subject + " "+ relation + "?"
                    question_answer_pair["label"] = object
                else:
                    question_answer_pair["question"] = interrogative + " is " + subject 
                    for i in range(multi_hops-1):
                        question_answer_pair["question"] += "'s " +  relation['edge_labels'][i] 
                    question_answer_pair["question"] += " "  + relation['edge_labels'][-1] + "?"
                    question_answer_pair["label"] = object
            else:
                if multi_hops == 1:
                    if object_type != "PERSON":
                        first_pair = next(iter(self.topic.items()))
                        if first_pair[1] != "revolution":
                            interrogative = "Which " + first_pair[1]
                        else:
                            interrogative = "Which revolution or war"
                    question_answer_pair["question"] = interrogative + " is " + " " + relation + " " + object + "?"
                    question_answer_pair["label"] = subject
                else:
                    question_answer_pair["question"] = interrogative 
                    first_pair = next(iter(self.topic.items()))
                    for i in range(multi_hops-1, 0, -1):
                        if first_pair[1] == "human" and i == multi_hops-1:
                            question_answer_pair["question"] += "se " +  relation['edge_labels'][i]
                        elif i == multi_hops-1:
                            first_pair = next(iter(self.topic.items()))
                            if first_pair[1] != "revolution":
                                interrogative = "Which " + first_pair[1]
                            else:
                                interrogative = "Which revolution or war"
                        else:
                            question_answer_pair["question"] += " " +  relation['edge_labels'][i]
                    question_answer_pair["question"] += " is "  + relation['edge_labels'][0] + " " + object + "?"
                
        elif subject_doc[0].tag_ == "VBD" and subject_doc[-1].tag_ not in ["NN", "NNP", "NNPS", "NNS"]:
            if not query_subject:
                if multi_hops == 1:
                    question_answer_pair["question"] = interrogative + " did " + subject + " " 
                else:
                    question_answer_pair["question"] = interrogative + " did " + subject
                    for i in range(multi_hops-1):
                        question_answer_pair["question"] += "'s " +  relation['edge_labels'][i] 
                    question_answer_pair["question"] += " " 
                for token in subject_doc:
                    if token.tag_ == "VBD":
                        question_answer_pair["question"] += token.lemma_ + " "
                    else:
                        question_answer_pair["question"] += token.text + " "
                question_answer_pair["question"] = question_answer_pair["question"][:-1] + "?"
                question_answer_pair["label"] = object
            else:
                if multi_hops == 1:
                    if object_type != "PERSON":
                        first_pair = next(iter(self.topic.items()))
                        if first_pair[1] != "revolution":
                            interrogative = "Which " + first_pair[1]
                        else:
                            interrogative = "Which revolution or war"
                    question_answer_pair["question"] = interrogative + " " + relation + " " + object + "?"
                    question_answer_pair["label"] = subject
                else:
                    first_pair = next(iter(self.topic.items()))
                    question_answer_pair["question"] = interrogative 
                    for i in range(multi_hops-1, 0, -1):
                        if first_pair[1] == "human" and i == multi_hops-1:
                            question_answer_pair["question"] += "se " +  relation['edge_labels'][i]
                        elif i == multi_hops-1:
                            first_pair = next(iter(self.topic.items()))
                            if first_pair[1] != "revolution":
                                interrogative = "Which " + first_pair[1]
                            else:
                                interrogative = "Which revolution or war"
                        else:
                            question_answer_pair["question"] += " " +  relation['edge_labels'][i]
                    question_answer_pair["question"] += " "  + relation['edge_labels'][0] + " " + object + "?"
                    question_answer_pair["label"] = subject
        elif (subject_doc[0].tag_ == "VB" or subject_doc[0].tag_ == "VBZ") and subject_doc[-1].tag_ not in ["NN", "NNP", "NNPS", "NNS"]:
            if not query_subject:
                if multi_hops == 1:
                    question_answer_pair["question"] = interrogative + " does " + subject + " "
                else:
                    question_answer_pair["question"] = interrogative + " does " + subject 
                    for i in range(multi_hops-1):
                        question_answer_pair["question"] += "'s " +  relation['edge_labels'][i] 
                    question_answer_pair["question"] += " " 
                for token in subject_doc:
                    if token.tag_ == "VBZ":
                        question_answer_pair["question"] += token.lemma_ + " "
                    else:
                        question_answer_pair["question"] += token.text + " "
                question_answer_pair["question"] = question_answer_pair["question"][:-1] + "?"
                question_answer_pair["label"] = object
            else:
                if multi_hops == 1:
                    if object_type != "PERSON":
                        first_pair = next(iter(self.topic.items()))
                        if first_pair[1] != "revolution":
                            interrogative = "Which " + first_pair[1]
                        else:
                            interrogative = "Which revolution or war"
                    question_answer_pair["question"] = interrogative + " " + relation + " " + object + "?"
                    question_answer_pair["label"] = subject
                else:
                    first_pair = next(iter(self.topic.items()))
                    question_answer_pair["question"] = interrogative 
                    for i in range(multi_hops-1, 0, -1):
                        if first_pair[1] == "human" and i == multi_hops-1:
                            question_answer_pair["question"] += "se " +  relation['edge_labels'][i]
                        elif i == multi_hops-1:
                            first_pair = next(iter(self.topic.items()))
                            if first_pair[1] != "revolution":
                                interrogative = "Which " + first_pair[1]
                            else:
                                interrogative = "Which revolution or war"
                        else:
                            question_answer_pair["question"] += " " +  relation['edge_labels'][i]
                    question_answer_pair["question"] += " "  + relation['edge_labels'][0] + " " + object + "?"
                    question_answer_pair["label"] = subject
        elif (subject_doc[-1].tag_ == "NN" or subject_doc[-1].tag_ == "NNP") and subject_doc[0].tag_ not in ["VB", "VBZ", "VBD"]: 
            if not query_subject:
                if multi_hops == 1:
                    question_answer_pair["question"] = interrogative + " is the " + relation + " of " + subject + "?"
                    question_answer_pair["label"] = object
                else:
                    question_answer_pair["question"] = interrogative + " is the " + relation['edge_labels'][1] + " of " + subject 
                    for i in range(multi_hops-1):
                        question_answer_pair["question"] += "'s " + relation['edge_labels'][i] 
                    question_answer_pair["question"] += "?"
                    question_answer_pair["label"] = object
            else:
                first_pair = next(iter(self.topic.items()))
                if multi_hops == 1:
                    if first_pair[1] == "human":
                        question_answer_pair["question"] = interrogative + "se " + relation + " is " + object + "?"
                    else:
                        first_pair = next(iter(self.topic.items()))
                        if first_pair[1] != "revolution":
                            interrogative = "Which " + first_pair[1]
                        else:
                            interrogative = "Which revolution or war"
                        question_answer_pair["question"] = interrogative + "'s " + relation + " is " + object + "?"
                    question_answer_pair["label"] = subject
                else:
                    question_answer_pair["question"] = interrogative 
                    for i in range(multi_hops-1, -1, -1):
                        if first_pair[1] == "human" and i == multi_hops-1:
                            question_answer_pair["question"] += "se " +  relation['edge_labels'][i]
                        elif i == multi_hops-1:
                            first_pair = next(iter(self.topic.items()))
                            if first_pair[1] != "revolution":
                                interrogative = "Which " + first_pair[1]
                            else:
                                interrogative = "Which revolution or war"
                        else:
                            question_answer_pair["question"] += "'s " +  relation['edge_labels'][i]
                    question_answer_pair["question"] += " is " + object + "?"
                    question_answer_pair["label"] = subject
        else:
            return None
       
        return question_answer_pair


    def generate_question_sets(self):
        print("====Start generating questions!====")
        relation_set = {}
        for node in self.directed_graph.nodes():
            in_edges = self.directed_graph.in_edges(node, data=True)
            for edge in in_edges:
                if relation_set.get(edge[2]['label']) is None:
                    relation_set[edge[2]['label']] = [edge]
                else:
                    relation_set[edge[2]['label']].append(edge)
                    
        yes_no_question_set = []
        MC_question_set = []
        wh_question_set = []
        remove_relation = ["topic's main category", "topic's main template", "described by source", "Commons category", "on focus list of Wikimedia project"]
        tmp_list = list(self.directed_graph.nodes())
        random.shuffle(tmp_list)
        wh_flag = False
        MC_flag = False
        YN_flag = False
        for node in tqdm(tmp_list[:self.limit_nodes]):            
            if wh_flag and MC_flag and YN_flag:
                break
            relation_query_object = []
            relation_query_subject = []
            duplicate_relation_query_object = []
            duplicate_relation_query_subject = []
            for _, target, edge_data in self.directed_graph.out_edges(node, data=True):
                label = edge_data['label']
                if label in relation_query_object:
                    duplicate_relation_query_object.append(label)
                else:
                    relation_query_object.append(label)
            for _, target, edge_data in self.directed_graph.in_edges(node, data=True):
                label = edge_data['label']
                if label in relation_query_subject:
                    duplicate_relation_query_subject.append(label)
                else:
                    relation_query_subject.append(label)
            for _, target, edge_data in self.directed_graph.out_edges(node, data=True):
                label = edge_data['label']
                if label in remove_relation:
                    continue
                if label not in duplicate_relation_query_object:
                    if not wh_flag:
                        wh_question = self.wh_question_generation(node, label, target)
                        if wh_question:
                            wh_question_set.append(wh_question)
                            # self.question_set.append(wh_question)
                if len(wh_question_set) >= self.wh_number:
                    wh_flag = True
                if not YN_flag:
                    yes_no_question = self.yes_no_question_generation(node, label, target, "Yes")
                    if yes_no_question:
                        yes_no_question_set.append(yes_no_question)
                        # self.question_set.append(yes_no_question)
                if len(yes_no_question_set) >= self.yes_no_number:
                    YN_flag = True
                disturb_nodes = []
                count = 0
                outedges = [v for u, v, attr in self.directed_graph.out_edges(node, data=True) if attr['label'] == label]

                if len(relation_set[label]) >= 4:
                    flag = 0
                    while count < 3 and flag < 100:
                        random_edge = random.choice(relation_set[label])
                        random_node = random_edge[1]
                        if random_node not in disturb_nodes and random_node != target and random_node not in outedges and random_node != node:
                            disturb_nodes.append(random_node)
                            count += 1
                        flag += 1
                    if count > 0:
                        disturb_node = random.choice(disturb_nodes)
                        if not YN_flag:
                            yes_no_question = self.yes_no_question_generation(node, label, disturb_node, "No")
                            if yes_no_question:
                                yes_no_question_set.append(self.yes_no_question_generation(node, label, disturb_node, "No"))
                                # self.question_set.append(self.yes_no_question_generation(node, label, disturb_node, "No"))
                    if len(yes_no_question_set) >= self.yes_no_number:
                        YN_flag = True
                
                    if count >= 3:
                        if not MC_flag:
                            MC_question = self.MC_question_generation(node, label, target, disturb_nodes[0], disturb_nodes[1], disturb_nodes[2])
                            if MC_question:
                                MC_question_set.append(MC_question)
                                # self.question_set.append(MC_question)
                    if len(MC_question_set) >= self.MC_number:
                        MC_flag = True
                elif len(relation_set[label]) >=2:
                    flag = 0
                    while count < 1 and flag < 100:
                        random_edge = random.choice(relation_set[label])
                        random_node = random_edge[1]
                        if random_node not in outedges:
                            disturb_nodes.append(random_node)
                            count += 1
                        flag += 1
                    if count > 0:
                        disturb_node = disturb_nodes[0]
                        if not YN_flag:
                            yes_no_question = self.yes_no_question_generation(node, label, disturb_node, "No")
                            if yes_no_question:
                                yes_no_question_set.append(self.yes_no_question_generation(node, label, disturb_node, "No"))
                                # self.question_set.append(self.yes_no_question_generation(node, label, disturb_node, "No"))
                    if len(yes_no_question_set) >= self.yes_no_number:
                        YN_flag = True

            for target, _, edge_data in self.directed_graph.in_edges(node, data=True):
                label = edge_data['label']
                if label in remove_relation:
                    continue
                if label not in duplicate_relation_query_subject:
                    if not wh_flag:
                        wh_question = self.wh_question_generation(node, label, target, query_subject=True)
                        if wh_question:
                            wh_question_set.append(wh_question)
                            # self.question_set.append(wh_question)
                if len(wh_question_set) >= self.wh_number:
                    wh_flag = True
                disturb_nodes = []
                count = 0
                inedges = [u for u, v, attr in self.directed_graph.in_edges(node, data=True) if attr['label'] == label]
                
                if len(relation_set[label]) >= 4:
                    flag = 0
                    while count < 3 and flag < 100:
                        random_edge = random.choice(relation_set[label])
                        random_node = random_edge[0]
                        if random_node not in disturb_nodes and random_node != target and random_node not in inedges and random_node != node:
                            disturb_nodes.append(random_node)
                            count += 1
                        flag += 1
                    if count >= 3:
                        if not MC_flag:
                            MC_question = self.MC_question_generation(node, label, target, disturb_nodes[0], disturb_nodes[1], disturb_nodes[2], query_subject=True)
                            if MC_question:
                                MC_question_set.append(MC_question)
                                # self.question_set.append(MC_question)
                    if len(MC_question_set) >= self.MC_number:
                        MC_flag = True
        self.question_set = yes_no_question_set + MC_question_set + wh_question_set
        if not self.check_grammar_flag and not self.diversify_method:
            local_out_file_name = self.out_file_name.replace("topic_","")#.split("_")[0] 
            with open(self.out_file_path + local_out_file_name + "_questions.json", "w", encoding='utf-8') as json_file:
                for record in tqdm(self.question_set, desc="Saving questions", unit="question"):
                    if record['type'] == "MC":
                        record["question"] = record['question'].split("\t\t")[0] + "        " + record['question'].split("\t\t")[-1]
                    json_string = json.dumps(record)
                    json_file.write(json_string + "\n")
                    count += 1
        print("====Finished generating questions!====")
        return self.question_set
    

    def generate_multi_hops_questions(self):
        print("====Start generating questions!====")
        remove_relation = ["topic's main category", "topic's main template", "described by source", "Commons category", "on focus list of Wikimedia project"]
        relation_set = {}
        for node in self.directed_graph.nodes():
            in_edges = self.directed_graph.in_edges(node, data=True)
            for edge in in_edges:
                if relation_set.get(edge[2]['label']) is None:
                    relation_set[edge[2]['label']] = [edge]
                else:
                    relation_set[edge[2]['label']].append(edge)
        yes_no_question_set = []
        MC_question_set = []
        MC_flag = False
        YN_flag = False
        tmp_list = list(self.directed_graph.nodes())
        random.shuffle(tmp_list)
        for source_node in tqdm(tmp_list[:self.limit_nodes]):
            if MC_flag and YN_flag:
                break
            for node, path in nx.single_source_shortest_path(self.directed_graph, source_node, cutoff=self.hops).items():
                if len(path) == self.hops+1:
                    # Retrieve the edge labels based on the path
                    edge_labels = [self.directed_graph[path[i]][path[i+1]]['label'] for i in range(len(path)-1)]
                    path_info = {'node': node, 'path': path[1:], 'edge_labels': edge_labels}
                    nodes_less_than_or_equal_to_dist_hops = [my_node for my_node, path in nx.single_source_shortest_path(self.directed_graph, source_node, cutoff=self.hops).items()]
                    label = path_info['edge_labels'][self.hops-1]
                    flag = False
                    for edge in path_info['edge_labels']:
                        if edge in remove_relation:
                            flag = True
                    if flag:
                        continue
                    disturb_count = 0
                    disturb_nodes = []
                    if not YN_flag:
                        yes_no_question = self.yes_no_question_generation(source_node, path_info, node, "Yes", self.hops)
                        if yes_no_question:
                            # self.question_set.append(yes_no_question)
                            yes_no_question_set.append(yes_no_question)
                    if len(yes_no_question_set) >= self.yes_no_number:
                        YN_flag = True
                
                    if len(relation_set[label]) >= 4:
                        count = 0
                        while disturb_count < 3 and count < 100:
                            random_edge = random.choice(relation_set[label])
                            random_node = random_edge[1]
                            if random_node not in nodes_less_than_or_equal_to_dist_hops and random_node not in disturb_nodes and random_node != source_node:
                                disturb_count += 1
                                disturb_nodes.append(random_node)
                            count += 1
                        if disturb_count > 0:
                            disturb_node = random.choice(disturb_nodes)
                            if not YN_flag:
                                yes_no_question = self.yes_no_question_generation(source_node, path_info, disturb_node, "No",  self.hops)
                                if yes_no_question:
                                    #self.question_set.append(self.yes_no_question_generation(source_node, path_info, disturb_node, "No", self.hops))
                                    yes_no_question_set.append(self.yes_no_question_generation(source_node, path_info, disturb_node, "No", self.hops))
                        if len(yes_no_question_set) >= self.yes_no_number:
                            YN_flag = True

                        if disturb_count >= 3:
                            if not MC_flag:
                                MC_question = self.MC_question_generation(source_node, path_info, node, disturb_nodes[0], disturb_nodes[1], disturb_nodes[2],  self.hops)
                                if MC_question:                 
                                    #self.question_set.append(MC_question)
                                    MC_question_set.append(MC_question)
                        if len(MC_question_set) >= self.MC_number:
                            MC_flag = True

                    elif len(relation_set[label]) >= 2:
                        count = 0
                        while disturb_count < 1 and count < 100:
                            random_edge = random.choice(relation_set[label])
                            random_node = random_edge[1]
                            if random_node not in nodes_less_than_or_equal_to_dist_hops:
                                disturb_count += 1
                                disturb_nodes.append(random_node)
                            count += 1
                        if disturb_count > 0:
                            disturb_node = disturb_nodes[0]
                            if not YN_flag:
                                yes_no_question = self.yes_no_question_generation(source_node, path_info, disturb_node, "No", self.hops)
                                if yes_no_question:
                                    #self.question_set.append(self.yes_no_question_generation(source_node, path_info, disturb_node, "No", self.hops))
                                    yes_no_question_set.append(self.yes_no_question_generation(source_node, path_info, disturb_node, "No", self.hops))
                        if len(yes_no_question_set) >= self.yes_no_number:
                            YN_flag = True
            
            for node, path in nx.single_source_shortest_path(self.directed_graph.reverse(), source_node, cutoff=self.hops).items():
                if len(path) == self.hops+1:
                    # Retrieve the edge labels based on the path
                    edge_labels = [self.directed_graph.reverse()[path[i]][path[i+1]]['label'] for i in range(len(path)-1)]
                    path_info = {'node': node, 'path': path[1:], 'edge_labels': edge_labels}
                    nodes_less_than_or_equal_to_dist_hops = [my_node for my_node, path in nx.single_source_shortest_path(self.directed_graph.reverse(), source_node, cutoff=self.hops).items()]
                    label = path_info['edge_labels'][self.hops-1]
                    flag = False
                    for edge in path_info['edge_labels']:
                        if edge in remove_relation:
                            flag = True
                    if flag:
                        continue
                    disturb_count = 0
                    disturb_nodes = []    
                    if len(relation_set[label]) >= 4:
                        count = 0
                        while disturb_count < 3 and count < 100:
                            random_edge = random.choice(relation_set[label])
                            random_node = random_edge[0]
                            if random_node not in nodes_less_than_or_equal_to_dist_hops and random_node not in disturb_nodes and random_node != source_node:
                                disturb_count += 1
                                disturb_nodes.append(random_node)
                            count += 1
                        if disturb_count >= 3:
                            if not MC_flag:
                                MC_question = self.MC_question_generation(source_node, path_info, node, disturb_nodes[0], disturb_nodes[1], disturb_nodes[2], self.hops, query_subject=True)
                                if MC_question:                 
                                    #self.question_set.append(MC_question) 
                                    MC_question_set.append(MC_question)
                        if len(MC_question_set) >= self.MC_number:
                            MC_flag = True

        self.question_set = yes_no_question_set + MC_question_set
        if not self.check_grammar_flag and not self.diversify_method:
            local_out_file_name = self.out_file_name.replace("topic_","")#.split("_")[0] 
            with open(self.out_file_path + local_out_file_name + "_questions.json", "w", encoding='utf-8') as json_file:
                for record in tqdm(self.question_set, desc="Saving questions", unit="question"):
                    if record['type'] == "MC":
                        record["question"] = record['question'].split("\t\t")[0] + "        " + record['question'].split("\t\t")[-1]
                    json_string = json.dumps(record)
                    json_file.write(json_string + "\n")
                    
                        
        print("====Finished generating questions!====")
        if len(self.question_set) == 0:
            print(f"{self.hops}-hops questions cannot be generated by the current constructed knowledge graph!") 
            sys.exit(1)
        return self.question_set


    def check_grammar(self, sentence):
        try:
            result = self.parser.parse(sentence)
            if len(result['corrections']) > 0:
                # There are grammar corrections
                return False
            else:
                # The sentence is grammatically correct
                return True
        except Exception as e:
            print(f"An error occurred while checking grammar: {e}")
            return False


    def discard_low_quality_question(self):
        print("====Start evaluating the quality of questions!====")
        filtered_questions = []
        count = 0 
        for question in tqdm(self.question_set):
            # if self.check_grammar(question['question']):
            #     filtered_questions.append(question)
            count += 1
            if count%30 == 0:
                time.sleep(5)
        
        # with concurrent.futures.ThreadPoolExecutor() as executor:
        #     futures = []
        #     count = 0
        #     for q in tqdm(self.question_set):
        #         count += 1
        #         future = executor.submit(self.check_grammar, q["question"])
        #         futures.append((future, q))
        #         if count%10 == 0:
        #             time.sleep(10)

        #     for future, question in tqdm(futures, total=len(futures)):
        #         if future.result():
        #             filtered_questions.append(question)

        print("====Low-quality questions have been discarded!====")
        if not diversify_method:
            with open(self.out_file_path + self.out_file_name + "_question_answer_pair.json", "w", encoding='utf-8') as json_file:
                for record in tqdm(filtered_questions, desc="Saving questions", unit="question"):
                    if record['type'] == "MC":
                        record["question"] = record['question'].split("\t\t")[0] + "        " + record['question'].split("\t\t")[-1]
                    json_string = json.dumps(record)
                    json_file.write(json_string + "\n")
                    

        self.question_set = filtered_questions
        print("[INFO] Generated questions have been saved!")
        return filtered_questions


    async def diversify_questions(self):
        if self.diversify_method == "GPT":
            print("====Start paraphrasing questions using GPT method!====")
        else:
            print("====Start paraphrasing questions using translation method!====")
        total = len(self.question_set)
        done_flag = [False for _ in range(total)]
        output_file = self.out_file_path + self.out_file_name + "_question_answer_pair.json"
        if os.path.exists(output_file):
            with open(output_file, 'r', encoding='utf-8') as out_file:
                out_file_lines = out_file.readlines()
                out_file_lines = [json.loads(l) for l in out_file_lines]
                assert len(out_file_lines) == total
                for idx, out_smp in enumerate(out_file_lines):
                    done_flag[idx] = self.done(out_smp, idx)
        else:
            with open(output_file, 'w', encoding='utf-8') as out_file:
                out_file.write("{}\n" * total)
        with tqdm(total=total) as pbar:
            pbar.update(len([0 for e in done_flag if e]))

            async def translate_remaining(api_key):
                while not all(done_flag):
                    to_be_translated_idx = done_flag.index(False)
                    done_flag[to_be_translated_idx] = True 
                    to_be_translated_smp = self.question_set[to_be_translated_idx]
                    if self.diversify_method == "GPT":
                        message = to_be_translated_smp['type'] + "\n\n\n" + to_be_translated_smp['question'].split("\t\t")[0]    
                        messages = self.get_messages(message, self.mode, self.diversify_method)
                        len_prompt = self.num_tokens_from_string(message, encoding_name="p50k_base") # wrong encoding
                        
                        try:
                            gen = await self.translate_with_backoff(
                                messages=messages,
                                len_prompt=len_prompt,
                                api_key=api_key,
                                temperature=random.random()
                            )
                            if to_be_translated_smp['type'] == "MC":
                                self.question_set[to_be_translated_idx]["question"] = gen + "       " + to_be_translated_smp['question'].split("\t\t")[-1]
                            else:
                                self.question_set[to_be_translated_idx]["question"] = gen
                        
                            self.write_to_specified_lines(
                                output_file,
                                to_be_translated_smp,
                                to_be_translated_idx
                            )
                            pbar.update(1)
                        except (OutOfQuotaException) as e:
                            done_flag[to_be_translated_idx] = False
                            logging.warning(e)
                            return
                        except openai.error.OpenAIError as e:
                            # Other error: mark done_flag as False and sleep a while
                            done_flag[to_be_translated_idx] = False
                            # logging.warning(e)
                            await asyncio.sleep(60)
                    else:
                        message = to_be_translated_smp['question'].split("\t\t")[0]
                        translator = Translator()
                        translated = translator.translate(message, src='en', dest="zh-CN")
                        translated_back = translator.translate(translated.text, src='zh-CN', dest="en")    
                        if to_be_translated_smp['type'] == "MC":
                            self.question_set[to_be_translated_idx]["question"] = translated_back.text + "      " + to_be_translated_smp['question'].split("\t\t")[-1]
                        else:
                            self.question_set[to_be_translated_idx]["question"] = translated_back.text  
                        self.write_to_specified_lines(
                                output_file,
                                to_be_translated_smp,
                                to_be_translated_idx
                            )
                        pbar.update(1)       
            await asyncio.gather(*[translate_remaining(k) for k in self.api_keys])

        # check if all done
        assert all(done_flag), f"Not all done. Check api-keys and rerun."
        print("====Finished paraphrasing questions!====")


    def num_tokens_from_string(self, string: str, encoding_name: str) -> int:
        """Returns the number of tokens in a text string."""
        encoding = tiktoken.get_encoding(encoding_name)
        num_tokens = len(encoding.encode(string))
        return num_tokens


    def get_messages(self, prompt, mode, diversify=False):
        if mode == 'user':
            messages = []
            tmp = prompt.split('\n\n\n')
            if tmp[0] == "MC":
                prompt = tmp[1]
            elif tmp[0] == "wh":
                prompt = tmp[1] + "(Directly give me the answer and don't explain)"
            elif tmp[0] == "yes_no":
                prompt = tmp[1] + "(Only need to answer 'Yes' or 'No', and don't explain)"
            return [{"role": "user", "content": prompt}]
        elif mode == 'sys-user-ass':
            messages = []
            prompt = prompt.split('\n\n\n')
            for id, smp in enumerate(prompt):
                if id == 0:
                    first_pair = next(iter(self.topic.items()))
                    if first_pair[1] == "human":
                        tmp_topic = self.topic["occupation"]
                    elif first_pair[1] == "revolution":
                        tmp_topic = "revolution and war"
                    else:
                        tmp_topic = first_pair[1]
                    
                    MC_content = "The following question's topic is about " + tmp_topic + ". Choose the only correct option for the multiple choice problem. (Answer 'A', 'B', 'C' or 'D')(Don't explain)"
                    Wh_content = "The following question's topic is about " + tmp_topic + ". Directly give me the answer in 'phrase' or 'word' format. Don't give me a sentence or explain"
                    yes_no_content = "The following question's topic is about " + tmp_topic + ". Only need to answer 'Yes' or 'No', and don't explain"
                    if self.use_icl:
                        MC_content = f"""
                            Choose the only correct option ('A', 'B', 'C' or 'D') for the multiple choice problem in topic {tmp_topic}. (Don't explain)
                            Below are some example problems for your reference
                            Examples:
                            1. Whose place of birth is Cleveland?  A. Mary Tyler Moore  B. Wes Craven  C. Todd Phillips  D. Cat Power 
                                B
                            2. Which film festival's country is France?  A. Confrontation  B. Nashville Film Festival  C. Taormina Film Fest  D. Jackson Hole Wildlife Film Festival 
                                A
                            3. Which recurring sporting event's location is Scotland?  A. Tokyo Marathon  B. The Ocean Race  C. parkrun Orsk  D. European Challenge Cup 
                                D
                            4. What is the twinned administrative body of Taipei?         A. Lisbon  B. San Francisco de Campeche  C. San Jos\u00e9  D. Krak\u00f3w 
                                C
                        """
                        Wh_content = f"""
                            Answer the following wh question in topic {tmp_topic} (Give the answer in 'phrase' or 'word' format. Don't give me a sentence or explain)
                            Below are some example problems for your reference
                            Examples:
                            1. What is the cause of death of Gracie Fields?
                                pneumonia
                            2. Who is the author of Brothers in Arms?
                                Lois McMaster Bujold
                            3. What is the country of Scottish Queer International Film Festival?"
                                United Kingdom
                            4. What is the location of Paraguayan War? 
                                Southern Cone
                        """
                        yes_no_content = f"""
                            Answer the following yes no question in topic {tmp_topic}:
                            Below are some example problems for your reference  (Only need to answer 'Yes' or 'No', and don't explain")
                            Examples:
                            1. Is United Kingdom the country of citizenship of Catherine Cookson?
                                Yes 
                            2. Was Alan Jay Lerner educated at Princeton University? 
                                No 
                            3. Is San Salvador the twinned administrative body of Monterrey?
                                Yes
                            4. Is Peru the country of Lima Film Festival?
                                Yes
                        """
                    
                    if not diversify:
                        if smp == "MC":
                            messages.append(
                                {"role": "system", "content": MC_content}
                            )
                        elif smp == "wh":
                            messages.append(
                                {"role": "system", "content": Wh_content}
                            )
                        elif smp == "yes_no":
                            messages.append(
                                {"role": "system", "content": yes_no_content}
                            )
                    else:
                        messages.append(
                                {"role": "system", "content": "The following question's topic is about " + tmp_topic + ". Paraphrase the following question for me"}
                            )
                elif id % 2 == 1:
                    messages.append(
                        {"role": "user", "content": smp}
                    )
                elif id % 2 == 0:
                    messages.append(
                        {"role": "assistant", "content": smp}
                    )
        elif mode == 'user-ass':
            messages = []
            prompt = prompt.split('\n\n\n')
            for id, smp in enumerate(prompt):
                if id % 2 == 0:
                    messages.append(
                        {"role": "user", "content": smp}
                    )
                elif id % 2 == 1:
                    messages.append(
                        {"role": "assistant", "content": smp}
                    )
        else:
            raise NotImplementedError
        return messages

    @backoff.on_exception(backoff.expo, (openai.RateLimitError, openai.APIError, openai.APIConnectionError), max_tries=5)
    async def translate_with_backoff(self, messages, len_prompt, api_key, temperature=0, AI_model="ChatGPT"):

        try:
            if AI_model == "ChatGPT" or AI_model == "ChatGPT-0613":
                if AI_model == "ChatGPT":
                    model_name = 'gpt-3.5-turbo'
                    if self.use_icl:
                        model_name = 'gpt-3.5-turbo-16k'
                elif AI_model == "ChatGPT-0613":
                    model_name = 'gpt-3.5-turbo-0613'
                    if self.use_icl:
                        model_name = 'gpt-3.5-turbo-0613-16k'
                response = await openai.ChatCompletion.acreate(
                    model=model_name,
                    messages=messages,
                    temperature=temperature,
                    top_p=1.0,
                    max_tokens=4000-len_prompt,
                    api_key=api_key,
                )
                gen = response['choices'][0]['message']['content'].strip().replace('\n\n\n', '\n\n')
            
            elif AI_model == "GPT4":
                model_name = "gpt-4"
                completion = await openai.ChatCompletion.create(
                                    model="gpt-4",
                                    messages=messages)
                gen = completion.choices[0].message["content"]
            if gen == "":
                gen = " "
            return gen

        except openai.error.RateLimitError as e:
            if "You exceeded your current quota, please check your plan and billing details" in e.user_message:
                raise OutOfQuotaException(api_key)
            elif "Your access was terminated due to violation of our policies" in e.user_message:
                raise AccessTerminatedException(api_key)
            else:
                raise e


    def done(self, out_smp, idx):
        if out_smp == {}:
            return False
        elif 'output' in out_smp and out_smp['output']!=' ':
            return True
        else:
            raise Exception(f"Check output file. Line: {idx+1}")


    def write_to_specified_lines(self, file_path, sample, line_id):
        with open(file_path, 'r+', encoding='utf-8', errors='replace') as f:
            all_lines = f.readlines()
            sample = json.dumps(sample, ensure_ascii=False)
            all_lines[line_id] = sample+'\n'
            f.seek(0)
            f.writelines(all_lines)
    

    async def question_asking(self):
        print("====Start generating outputs!====")
        if self.evaluate:
            self.question_set = []
            local_out_file_name = self.out_file_name#.split("_")[0] 
            input_file = self.out_file_path + local_out_file_name + "_question_answer_pair.json"
            with open(input_file, 'r', encoding='utf-8') as my_file:
                line_count = sum(1 for _ in my_file)  # Count the lines in the file
                my_file.seek(0)  # Reset the file pointer to the beginning
                for line in tqdm(my_file, total=line_count):
                    data = json.loads(line)
                    self.question_set.append(data)

        total = len(self.question_set)
        done_flag = [False for _ in range(total)]
        output_file = self.out_file_path + self.out_file_name + "_results.json"
        if os.path.exists(output_file):
            with open(output_file, 'r', encoding='utf-8') as out_file:
                out_file_lines = out_file.readlines()
                out_file_lines = [json.loads(l) for l in out_file_lines]
                assert len(out_file_lines) == total
                for idx, out_smp in enumerate(out_file_lines):
                    done_flag[idx] = self.done(out_smp, idx)
        else:
            with open(output_file, 'w', encoding='utf-8') as out_file:
                out_file.write("{}\n" * total)
        with tqdm(total=total) as pbar:
            pbar.update(len([0 for e in done_flag if e]))

            async def translate_remaining(api_key):
                
                while not all(done_flag):
                    to_be_translated_idx = done_flag.index(False)
                    done_flag[to_be_translated_idx] = True
                    to_be_translated_smp = self.question_set[to_be_translated_idx]
                    first_pair = next(iter(self.topic.items()))
                    if first_pair[1] == "human":
                        tmp_topic = self.topic["occupation"]
                    elif first_pair[1] == "revolution":
                        tmp_topic = "revolution and war"
                    else:
                        tmp_topic = first_pair[1]
                    message = to_be_translated_smp['type'] + "\n\n\n" + to_be_translated_smp['question']
                    
                    MC_content = "The following question's topic is about " + tmp_topic + ". Choose the only correct option for the multiple choice problem. (Answer 'A', 'B', 'C' or 'D')(Don't explain)"
                    Wh_content = "The following question's topic is about " + tmp_topic + ". Directly give me the answer in 'phrase' or 'word' format. Don't give me a sentence or explain"
                    yes_no_content = "The following question's topic is about " + tmp_topic + ". Only need to answer 'Yes' or 'No', and don't explain"
                    if self.use_icl:
                        MC_content = f"""
                            Choose the only correct option ('A', 'B', 'C' or 'D') for the multiple choice problem in topic {tmp_topic}. (Don't explain)
                            Below are some example problems for your reference
                            Examples:
                            1. Whose place of birth is Cleveland?  A. Mary Tyler Moore  B. Wes Craven  C. Todd Phillips  D. Cat Power 
                                B
                            2. Which film festival's country is France?  A. Confrontation  B. Nashville Film Festival  C. Taormina Film Fest  D. Jackson Hole Wildlife Film Festival 
                                A
                            3. Which recurring sporting event's location is Scotland?  A. Tokyo Marathon  B. The Ocean Race  C. parkrun Orsk  D. European Challenge Cup 
                                D
                            4. What is the twinned administrative body of Taipei?         A. Lisbon  B. San Francisco de Campeche  C. San Jos\u00e9  D. Krak\u00f3w 
                                C
                        """
                        Wh_content = f"""
                            Answer the following wh question in topic {tmp_topic} (Give the answer in 'phrase' or 'word' format. Don't give me a sentence or explain)
                            Below are some example problems for your reference
                            Examples:
                            1. What is the cause of death of Gracie Fields?
                                pneumonia
                            2. Who is the author of Brothers in Arms?
                                Lois McMaster Bujold
                            3. What is the country of Scottish Queer International Film Festival?"
                                United Kingdom
                            4. What is the location of Paraguayan War? 
                                Southern Cone
                        """
                        yes_no_content = f"""
                            Answer the following yes no question in topic {tmp_topic}:
                            Below are some example problems for your reference  (Only need to answer 'Yes' or 'No', and don't explain"
                            Examples:
                            1. Is United Kingdom the country of citizenship of Catherine Cookson?
                                Yes 
                            2. Was Alan Jay Lerner educated at Princeton University? 
                                No 
                            3. Is San Salvador the twinned administrative body of Monterrey?
                                Yes
                            4. Is Peru the country of Lima Film Festival?
                                Yes
                        """
                
                    messages = self.get_messages(message, self.mode)
                    len_prompt = self.num_tokens_from_string(message, encoding_name="p50k_base") # wrong encoding
                    if to_be_translated_smp['type'] == "MC":
                        current_prompt = MC_content + "\nQuestion:" + to_be_translated_smp['question'] 
                    elif to_be_translated_smp['type'] == "yes_no":
                        current_prompt = yes_no_content + "\nQuestion:" +  to_be_translated_smp['question'] 
                    else:
                        current_prompt = Wh_content + "\nQuestion:" +  to_be_translated_smp['question'] 
                    try:
                        if self.AI_model == "ChatGPT" or self.AI_model == "GPT4" or self.AI_model == "ChatGPT-0613" :
                            gen = await self.translate_with_backoff(
                                messages=messages,
                                len_prompt=len_prompt,
                                api_key=api_key,
                                temperature=self.temperature,
                                AI_model=self.AI_model
                            )
                            to_be_translated_smp['output'] = gen
                        elif self.AI_model == "GPT3-002":
                            openai.api_key = api_key
                            completion = openai.Completion.create(
                                    model="text-davinci-002",
                                    prompt=current_prompt,
                                    temperature=0,
                                    max_tokens=150,
                                    top_p=1,
                                    frequency_penalty=0.0,
                                    presence_penalty=0.6,
                                    )
                            to_be_translated_smp['output'] = completion.choices[0].text.replace('\n\n', '')    
                        elif self.AI_model == "GPT3-003":
                            openai.api_key = api_key
                            completion = openai.Completion.create(
                                    model="text-davinci-003",
                                    prompt=current_prompt,
                                    temperature=0,
                                    max_tokens=150,
                                    top_p=1,
                                    frequency_penalty=0.0,
                                    presence_penalty=0.6,
                                    )
                            to_be_translated_smp['output'] = completion.choices[0].text.replace('\n\n', '')             
                        elif self.AI_model == "Vicuna": # , cache_dir="../cache"
                            vicuna_model = transformers.AutoModelForCausalLM.from_pretrained("eachadea/vicuna-13b-1.1").to(self.device)
                            vicuna_tokenizer = transformers.AutoTokenizer.from_pretrained("eachadea/vicuna-13b-1.1")
                            inputs = vicuna_tokenizer(current_prompt, return_tensors="pt").to(self.device)
                            generate_ids = vicuna_model.generate(inputs.input_ids, max_length=500)
                            to_be_translated_smp['output'] = vicuna_tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].replace(current_prompt, "")
                        elif self.AI_model == "Alpaca":
                            alpaca_model = transformers.AutoModelForCausalLM.from_pretrained("../alpaca-7b", cache_dir="../cache").to(self.device)
                            alpaca_tokenizer = transformers.AutoTokenizer.from_pretrained("../alpaca-7b", cache_dir="../cache")   
                            inputs = alpaca_tokenizer(current_prompt, return_tensors="pt").to(self.device)
                            generate_ids = alpaca_model.generate(inputs.input_ids, max_length=500)
                            to_be_translated_smp['output'] = alpaca_tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].replace(current_prompt, "")
                        elif self.AI_model == "Llama":
                            llama_model = transformers.AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct").to(self.device)
                            tok = transformers.AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")   
                            inputs = tok(current_prompt, return_tensors="pt").to(self.device)
                            generate_ids = llama_model.generate(inputs.input_ids, max_length=500)
                            to_be_translated_smp['output'] = tok.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].replace(current_prompt, "")

                            # messages = [{"role": "system", "content": system_msg_qa}, {"role": "user", "content": f'Question: {prompt}. Answer:'}]
                            # msg_tokenized = tok.apply_chat_template(messages, add_generation_prompt=True, return_tensors='pt')
                            # output_ids = model.generate(msg_tokenized.to(device), max_new_tokens=16, eos_token_id=terminators, do_sample=False, pad_token_id=tok.eos_token_id)
                            # output_decoded = tok.decode(output_ids[0][msg_tokenized.shape[-1]:], skip_special_tokens=True)
                            
                        self.write_to_specified_lines(
                            output_file,
                            to_be_translated_smp,
                            to_be_translated_idx
                        )
                        pbar.update(1)
                    except (OutOfQuotaException) as e:
                        done_flag[to_be_translated_idx] = False
                        logging.warning(e)
                        return
                    except openai.error.OpenAIError as e:
                        # Other error: mark done_flag as False and sleep a while
                        done_flag[to_be_translated_idx] = False
                        # logging.warning(e)
                        await asyncio.sleep(60)
                       
            await asyncio.gather(*[translate_remaining(k) for k in self.api_keys])

        # check if all done
        assert all(done_flag), f"Not all done. Check api-keys and rerun."
        print("====Finished generating outputs!====")

    
    async def GPT_evaluate_result(self, data, current_api_key):
        messages = []
        first_pair = next(iter(self.topic.items()))
        info = "Question: " + data["question"], "\nModel Answer: " + data["answer"] + "\nAI model generated answer: " + data["output"] \
            + "\n. The above question's topic is " + first_pair[1]  + ". "
        string = """\n Please evaluate the correctness of the AI model's answer compared to the model answer. 
            Consider the following criteria and provide your judgment:
            If the AI's answer is a more specific version of the model answer, please respond with: "Correct"
            If the AI's answer is a more general version of the model answer, please respond with: "Correct".
            If the AI's answer is a closely related to the model answer, please respond with: "Correct".
            If the AI's answer and the model answer are entirely different entities with no direct relationship, please respond with: "Incorrect".
            """
        info += string
       
        messages.append({"role": "user", "content": str(info)})

        len_prompt = self.num_tokens_from_string(str(info), encoding_name="p50k_base") # wrong encoding
        
        try:
            gen = await self.translate_with_backoff(
                messages=messages,
                len_prompt=len_prompt,
                api_key=current_api_key,
                temperature=self.temperature
            )
            return gen        
        except (OutOfQuotaException) as e:
            logging.warning(e)
            return
        except openai.error.OpenAIError as e:
            # Other error: mark done_flag as False and sleep a while
            # logging.warning(e)
            await asyncio.sleep(60)        


    def evaluation_for_MC_TF_question(self):
        print("====Start evaluating AI conversational model's performance on MC and yes-no questions!====")
        MC_count = 0
        MC_correct = 0
        yes_no_count = 0
        yes_no_correct = 0
        input_file = self.out_file_path + self.out_file_name + "_results.json"
        with open(self.out_file_path + self.out_file_name + "_TF_MC_evaluate.json", "w", encoding='utf-8') as json_file:
            with open(input_file, 'r', encoding='utf-8') as my_file:
                line_count = sum(1 for _ in my_file)  # Count the lines in the file
                my_file.seek(0)  # Reset the file pointer to the beginning
                for line in tqdm(my_file, total=line_count):
                    data = json.loads(line)
                    answer = data['answer']
                    output = data['output']
                    flag = False
                    if data["type"] == "MC":
                        MC_count += 1
                        if output.lower().split(".")[0] == answer.lower():
                            MC_correct += 1
                            flag = True
                    elif data["type"] == "yes_no":
                        yes_no_count += 1
                        if answer == output.rstrip("."):
                            yes_no_correct += 1
                            flag = True

                    if not flag and data["type"] in ["yes_no", "MC"]:
                        json_string = json.dumps(data)
                        json_file.write(json_string + "\n")

            if MC_count != 0:
                MC_acc_dict = {"MC_accuracy": MC_correct/MC_count}
                json_string = json.dumps(MC_acc_dict)
                json_file.write(json_string + "\n")
            if yes_no_count != 0:
                TF_acc_dict = {"yes_no__accuracy": yes_no_correct/yes_no_count}
                json_string = json.dumps(TF_acc_dict)
                json_file.write(json_string + "\n")
        if MC_count != 0:
            print(f"[INFO] The MC question accuracy of the language model is {MC_correct/MC_count}")
        else:
            print("[INFO] MC questions do not exist!")  
        if yes_no_count != 0:
            print(f"[INFO] The yes no question accuracy of the language model is {yes_no_correct/yes_no_count}")
        else:
            print("[INFO] yes no questions do not exist!")  
        print("====Finished evaluating AI conversational model's performance on MC and yes-no questions!====")


    def Levenshtein_distance_evaluation(self):
        wh_count = 0
        wh_correct = 0
        input_file = self.out_file_path + self.out_file_name + "_results.json"
        with open(self.out_file_path + self.out_file_name + "_wh_Levenshtein_distance_evaluate.json", "w") as json_file:
            with open(input_file, 'r', encoding='utf-8') as my_file:
                line_count = sum(1 for _ in my_file)  # Count the lines in the file
                my_file.seek(0)  # Reset the file pointer to the beginning
                for line in tqdm(my_file, total=line_count):
                    data = json.loads(line)
                    answer = data['answer']
                    output = data['output']
                    flag = False
                    if data["type"] == "wh":
                        wh_count += 1
                        # Calculate similarity score using fuzzywuzzy
                        similarity_score = fuzz.ratio(answer, output.rstrip('.'))
                        # Define a threshold for similarity
                        threshold = 80  # Adjust as needed
                        # Compare the similarity score with the threshold
                        if similarity_score >= threshold:
                            wh_correct += 1
                            flag = True

                    if not flag and data["type"] == "wh":
                        json_string = json.dumps(data)
                        json_file.write(json_string + "\n")
            if wh_count != 0:
                wh_acc_dict = {"wh_accuracy": wh_correct/wh_count}
                json_string = json.dumps(wh_acc_dict)
                json_file.write(json_string + "\n")
                      
        if wh_count != 0:
            print(f"[INFO] [Levenshtein distance] The wh question accuray of language model is ", str(wh_correct/wh_count))  
        else:
            print("[INFO] wh questions do not exist!")     
       

    def N_grams_evaluation(self):
        wh_count = 0
        wh_correct = 0    
        input_file = self.out_file_path + self.out_file_name + "_results.json"
        with open(self.out_file_path + self.out_file_name + "_wh_N_grams_evaluate.json", "w", encoding='utf-8') as json_file:
            with open(input_file, 'r', encoding='utf-8') as my_file:
                line_count = sum(1 for _ in my_file)  # Count the lines in the file
                my_file.seek(0)  # Reset the file pointer to the beginning
                for line in tqdm(my_file, total=line_count):
                    data = json.loads(line)
                    answer = data['answer']
                    output = data['output']
                    flag = False
                    if data["type"] == "wh":
                        wh_count += 1
                        answer_tokens = answer.split()
                        output_tokens = output.split()
                        answer_ngrams = set(ngrams(answer_tokens, 1))
                        output_ngrams = set(ngrams(output_tokens, 1))
                        intersection = len(answer_ngrams.intersection(output_ngrams))
                        union = len(answer_ngrams) + len(output_ngrams) - intersection
                        similarity_score = intersection / union
                        threshold = 0.7
                        if similarity_score >= threshold:
                            wh_correct += 1
                            flag = True        

                    if not flag and data["type"] == "wh":
                        json_string = json.dumps(data)
                        json_file.write(json_string + "\n")
            if wh_count != 0:
                wh_acc_dict = {"wh_accuracy": wh_correct/wh_count}
                json_string = json.dumps(wh_acc_dict)
                json_file.write(json_string + "\n")
        if wh_count != 0:
            print(f"[INFO] [N grams] The wh question accuray of language model is ", str(wh_correct/wh_count))
        else:
            print("[INFO] wh questions do not exist!")


    def Word_embeddings_evaluation(self):
        wh_count = 0
        wh_correct = 0
        nlp = spacy.load('en_core_web_lg')
        input_file = self.out_file_path + self.out_file_name + "_results.json"
        with open(self.out_file_path + self.out_file_name + "_wh_word_embeddings_evaluate.json", "w", encoding='utf-8') as json_file:
            with open(input_file, 'r', encoding='utf-8') as my_file:
                line_count = sum(1 for _ in my_file)  # Count the lines in the file
                my_file.seek(0)  # Reset the file pointer to the beginning
                for line in tqdm(my_file, total=line_count):
                    data = json.loads(line)
                    answer = data['answer']
                    output = data['output']
                    flag = False

                    if data["type"] == "wh":
                        wh_count += 1
                        answer_embedding = nlp(answer).vector
                        output_embedding = nlp(output.rstrip('.')).vector
                        similarity_score = cosine_similarity(answer_embedding.reshape(1, -1), output_embedding.reshape(1, -1))[0][0]
                        threshold = 0.7   
                        if similarity_score >= threshold:
                            wh_correct += 1
                            flag = True

                    if not flag and data["type"] == "wh":
                        json_string = json.dumps(data)
                        json_file.write(json_string + "\n")
            if wh_count != 0:
                wh_acc_dict = {"wh_accuracy": wh_correct/wh_count}
                json_string = json.dumps(wh_acc_dict)
                json_file.write(json_string + "\n")

        if wh_count != 0:
            print(f"[INFO] [Word embedding] The wh question accuracy of the language model is {wh_correct / wh_count}")            
        else:
            print("[INFO] wh questions do not exist!")


    def Sentence_transformer_evaluation(self):
        wh_count = 0
        wh_correct = 0
        model_name = 'paraphrase-MiniLM-L6-v2'
        model = SentenceTransformer(model_name)
        input_file = self.out_file_path + self.out_file_name + "_results.json"
        with open(self.out_file_path + self.out_file_name + "_wh_sentence_transformer_evaluate.json", "w", encoding='utf-8') as json_file:
            with open(input_file, 'r', encoding='utf-8') as my_file:
                line_count = sum(1 for _ in my_file)  # Count the lines in the file
                my_file.seek(0)  # Reset the file pointer to the beginning
                for line in tqdm(my_file, total=line_count):
                    data = json.loads(line)
                    answer = data['answer']
                    output = data['output']
                    flag = False

                    if data["type"] == "wh":
                        wh_count += 1
                        embeddings = model.encode([answer, output])
                        similarity_score = util.cos_sim(embeddings[0], embeddings[1])
                        threshold = 0.6   
                        if similarity_score >= threshold:
                            wh_correct += 1
                            flag = True
                        
                    if not flag and data["type"] == "wh":
                        json_string = json.dumps(data)
                        json_file.write(json_string + "\n")
            if wh_count != 0:
                wh_acc_dict = {"wh_accuracy": wh_correct/wh_count}
                json_string = json.dumps(wh_acc_dict)
                json_file.write(json_string + "\n")

        if wh_count != 0:
            print(f"[INFO] [Sentence Transformer] The wh question accuracy of the language model is {wh_correct / wh_count}")
        else:
            print("[INFO] wh questions do not exist!")
        
       
    async def ChatGPT_evaluation(self):
        input_file = self.out_file_path + self.out_file_name + "_results.json"
        self.question_output_set = []
        with open(input_file, 'r', encoding='utf-8') as my_file:
            total = sum(1 for _ in my_file) 
            my_file.seek(0)  
            for line in my_file:
                data = json.loads(line)
                self.question_output_set.append(data)
        
        done_flag = [False for _ in range(total)]
        output_file = self.out_file_path + self.out_file_name + "_wh_ChatGPT_evaluate.json"        

        wh_count = 0
        wh_correct = 0
        with tqdm(total=total) as pbar:
            pbar.update(len([0 for e in done_flag if e]))
            wh_count_lock = asyncio.Lock()
            wh_correct_lock = asyncio.Lock()
            output_file_lock = asyncio.Lock()

            async def translate_remaining(api_key):
                nonlocal wh_count, wh_correct
                while not all(done_flag):
                    to_be_translated_idx = done_flag.index(False)
                    done_flag[to_be_translated_idx] = True 
                    to_be_translated_smp = self.question_output_set[to_be_translated_idx]
                    flag = False
                    try:
                        if to_be_translated_smp["type"] == "wh":
                            async with wh_count_lock:
                                wh_count += 1
                            result = await self.GPT_evaluate_result(to_be_translated_smp, api_key)
                            if result and result.rstrip('.') == "Correct":
                                async with wh_correct_lock:
                                    wh_correct += 1
                                flag = True

                        if not flag and to_be_translated_smp["type"] == "wh":
                            json_string = json.dumps(to_be_translated_smp)
                            async with output_file_lock:
                                with open(output_file, 'a', encoding='utf-8') as out_file:
                                    out_file.write(json_string + "\n")
                    
                        pbar.update(1)
                    except (OutOfQuotaException) as e:
                        done_flag[to_be_translated_idx] = False
                        logging.warning(e)
                        return
                    except openai.error.OpenAIError as e:
                        # Other error: mark done_flag as False and sleep a while
                        done_flag[to_be_translated_idx] = False
                        # logging.warning(e)
                        await asyncio.sleep(60)
                        
            await asyncio.gather(*[translate_remaining(k) for k in self.api_keys])
        # check if all done
        assert all(done_flag), f"Not all done. Check api-keys and rerun."
        if wh_count != 0:
            wh_acc_dict = {"wh_accuracy": wh_correct/wh_count}
            json_string = json.dumps(wh_acc_dict)
            with open(output_file, 'a', encoding='utf-8') as out_file:
                out_file.write(json_string + "\n")

        if wh_count != 0:
            print(f"[INFO] [ChatGPT 3.5] The wh question accuracy of the language model is {wh_correct/wh_count}")
        else:
            print("[INFO] wh questions do not exist!") 

 
def parse_args():
    parser = argparse.ArgumentParser("", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-n", "--name", type=str, default="", #required=True,
        help="Output file name")
    parser.add_argument("-o", "--output", type=str, required=True,
        help="Output file path")
    parser.add_argument("-t", "--temperature", type=float, default=0,
        help="Sampling temperature")
    parser.add_argument("-m", "--mode", type=str, default='sys-user-ass',
        choices=['user', 'user-ass', 'sys-user-ass'],
        help="Turbo message mode")
    parser.add_argument("-hops", "--set_multihops", type=int, default=1,
        help="Set the hop of generated questions")
    parser.add_argument("-v", "--visualise", type=bool, default=False,
        help="Visualise the knowledge graph")
    parser.add_argument("-diver", "--diversify", type=str, default=None,
        choices=['GPT', 'Translate'],
        help="Choose paraphrasing method: [GPT method] [Translate method]")
    parser.add_argument("--topics", type=str, required=True,
        help="The topics json file path")
    parser.add_argument("--neo4j_uri", type=str, default="bolt://127.0.0.1:7687",
        help="Set URI for your neo4j database")
    parser.add_argument("--neo4j_username", type=str, default="neo4j",
        help="Set username for your neo4j database")
    parser.add_argument("--neo4j_password", type=str, default="FactChecker",
        help="Set password for your neo4j database")
    parser.add_argument("--check_grammar_flag", type=bool, default=False,
        help="Decide whether to discard questions with grammar mistakes")
    parser.add_argument("--model", type=str, default='ChatGPT',
        choices=['GPT4', 'ChatGPT', 'BARD', 'GPT3-002', 'GPT3-003','Vicuna', 'Alpaca', 'ChatGPT-0613', 'Llama'],
        help="Choose AI model for evaluation")
    parser.add_argument("--evaluate", type=bool, default=False,
        help="Choose whether to generate questions or not")
    parser.add_argument("--read_graph", type=bool, default=False,
        help="Decide whether directly read the knolwedge graph")
    parser.add_argument("--graph_path", type=str,  default=None,
        help="Provide the path for the knowlege graph")
    parser.add_argument("--only_check_performance", type=bool, default=False,
        help="Only check the final performance of the given chatbot")
    parser.add_argument("--limit_nodes", type=int, default=5000,
        help="The maximun number of nodes traversed for generating questions")
    parser.add_argument("--yes_no_number", type=int, default=500,
        help="The number of generated yes_no questions")
    parser.add_argument("--MC_number", type=int, default=500,
        help="The number of generated MC questions")
    parser.add_argument("--wh_number", type=int, default=500,
        help="The number of generated wh_questions")
    parser.add_argument("--use_icl", type=bool, default=False,
        help="Determine whether to use in-context learning")
    parser.add_argument("--generate_questions_only", type=bool, default=False,
        help="Decide whether to generate questions only")
    
    parser.add_argument("--device", type=str)
    
    return parser.parse_args()


if __name__ == "__main__":
    data_path = '../data/'
    args = parse_args()
    out_file_name = args.name
    out_file_path = args.output
    temperature = args.temperature
    mode = args.mode
    multi_hops = args.set_multihops
    visualise = args.visualise
    diversify_method = args.diversify
    topics_path = data_path+args.topics
    neo4j_uri = args.neo4j_uri
    neo4j_username = args.neo4j_username
    neo4j_password = args.neo4j_password
    check_grammar_flag = args.check_grammar_flag
    read_graph = args.read_graph
    graph_path = data_path+args.graph_path if args.graph_path else None
    AI_model = args.model
    only_evaluate = args.evaluate
    yes_no_number = args.yes_no_number
    MC_number = args.MC_number
    wh_number = args.wh_number
    limit_nodes = args.limit_nodes
    only_check_performance = args.only_check_performance
    use_icl = args.use_icl
    generate_questions_only = args.generate_questions_only
    asker = FactChecker(out_file_name, out_file_path, temperature, mode, multi_hops, diversify_method,  \
                      topics_path, neo4j_uri, neo4j_username, neo4j_password, check_grammar_flag, AI_model,\
                    only_evaluate, read_graph, graph_path, limit_nodes, yes_no_number, MC_number, wh_number, use_icl)
    device = args.device

    async def main():
        if not only_check_performance:
            if not only_evaluate:
                if not read_graph:
                    asker.fact_triplets_retrival()
                    asker.graph_creating()
                    if visualise:
                        asker.visualise_fact_graph()
                else:
                    asker.read_graph_file()
                if multi_hops == 1:
                    asker.generate_question_sets()
                else:
                    asker.generate_multi_hops_questions()
                if check_grammar_flag:
                    asker.discard_low_quality_question()
                if diversify_method:
                    await asker.diversify_questions()
            if not generate_questions_only:
                await asker.question_asking()
        if not generate_questions_only:
            asker.evaluation_for_MC_TF_question()
            if multi_hops == 1:
                print("====Start evaluating AI conversational model's performance on wh-questions!====")
                # asker.Levenshtein_distance_evaluation()
                # asker.N_grams_evaluation()
                # asker.Word_embeddings_evaluation()
                asker.Sentence_transformer_evaluation()
                # await asker.ChatGPT_evaluation()
                print("====Finished evaluating AI conversational model's performance on wh-questions!====")

    asyncio.run(main())