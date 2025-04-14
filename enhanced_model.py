import torch
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import os
import pickle
import re
import time
import spacy
from enum import Enum
from transformers import T5ForConditionalGeneration, T5Tokenizer, RobertaForSequenceClassification, RobertaTokenizer
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
import torch
import pandas as pd
from werkzeug.utils import secure_filename
from model import PDFStandardizer

class RAGEnhancedPDFStandardizer(PDFStandardizer):
    def __init__(self):
        # Commenting out the super().__init__() to stop loading the model
        # super().__init__()
        # Initialize sentence transformer for embedding
        self.sentence_encoder = SentenceTransformer('all-MiniLM-L6-v2')
        # Knowledge base storage
        self.knowledge_chunks = []
        self.knowledge_embeddings = None
    
    def add_to_knowledge_base(self, documents):
        """
        Add documents to the knowledge base
        """
        # Break documents into chunks
        chunks = []
        for doc in documents:
            # Simple chunking by paragraphs
            paragraphs = [p for p in doc.split('\n\n') if p.strip()]
            chunks.extend(paragraphs)
        
        # Add to existing chunks
        self.knowledge_chunks.extend(chunks)
        
        # Update embeddings
        self._update_knowledge_embeddings()
    
    def _update_knowledge_embeddings(self):
        """
        Update the vector embeddings for the knowledge base
        """
        if not self.knowledge_chunks:
            return
        
        # Generate embeddings for all chunks
        self.knowledge_embeddings = self.sentence_encoder.encode(
            self.knowledge_chunks, 
            convert_to_tensor=True
        )
    
    def retrieve_relevant_context(self, query, top_k=3):
        """
        Retrieve the most relevant context chunks for a query
        """
        if not self.knowledge_chunks or self.knowledge_embeddings is None:
            return []
        
        # Encode the query
        query_embedding = self.sentence_encoder.encode(query, convert_to_tensor=True)
        
        # Calculate similarity
        similarities = cosine_similarity(
            query_embedding.cpu().numpy().reshape(1, -1),
            self.knowledge_embeddings.cpu().numpy()
        )[0]
        
        # Get top-k chunks
        top_indices = similarities.argsort()[-top_k:][::-1]
        
        # Return relevant chunks
        return [self.knowledge_chunks[i] for i in top_indices]
    
    def generate_with_rag(self, query):
        """
        Generate a response using RAG to reduce hallucination
        """
        # Retrieve relevant context
        context_chunks = self.retrieve_relevant_context(query)
        
        if not context_chunks:
            return "No relevant information found in the knowledge base."
        
        # Combine context chunks
        context = "\n\n".join(context_chunks)
        
        # Create RAG prompt
        rag_prompt = f"""
        Answer the following query based ONLY on the provided context information.
        If the context doesn't contain the information needed, respond with "I don't have enough information to answer this question."
        
        Context:
        {context}
        
        Query: {query}
        """
        
        # Process with T5 model
        input_ids = self.t5_tokenizer(rag_prompt, return_tensors="pt").input_ids
        
        # Generate with constraints
        outputs = self.t5_model.generate(
            input_ids,
            max_length=512,
            num_beams=4,
            no_repeat_ngram_size=3,
            early_stopping=True
        )
        
        response = self.t5_tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        return response
    
    
       #This untested enhanced implementation below is intended to:
       #Uses the transformer-based NER pipeline for high-accuracy entity extraction
       #Includes confidence scores for each entity
       #Uses spaCy as a backup to catch entities the transformer might miss
       #Marks the source of each entity for traceability

def extract_entities_with_confidence(self, text):
    #Extract entities with confidence scores using transformer-based models
    # Get entities from Hugging Face pipeline
    ner_results = self.ner_pipeline(text)
    # Initialize entity categories with confidence scores
    entities = {
        "Locations": [], 
        "Dates": [], 
        "Quantities": [], 
        "Other": []
    }
# Process entities from transformer model
    for entity in ner_results:
        entity_text = entity["word"]
        entity_type = entity["entity_group"]
        confidence = entity["score"]
        entity_data = {
            "text": entity_text,
            "confidence": confidence
        }
        # Map to our categories
        if entity_type in ["LOC", "GPE"]:
            entities["Locations"].append(entity_data)
        elif entity_type == "DATE":
            entities["Dates"].append(entity_data)
        elif entity_type in ["CARDINAL", "QUANTITY", "MONEY", "PERCENT"]:
            entities["Quantities"].append(entity_data)
        else:
            entities["Other"].append(entity_data)
    # Also use spaCy as backup for any entities the transformer might miss
    doc = self.nlp(text)
    for ent in doc.ents:
        # Check if this entity is already captured by the transformer
        already_captured = False
        for category in entities:
            for existing_entity in entities[category]:
                if ent.text in existing_entity["text"] or existing_entity["text"] in ent.text:
                    already_captured = True
                    break
            if already_captured:
                break
    
        if not already_captured:
            # Add with a default confidence score
            entity_data = {
                "text": ent.text,
                "confidence": 0.75,  # Default confidence for spaCy entities
                "source": "spacy"    # Mark the source
            }
            # Map to our categories
            if ent.label_ in ["GPE", "LOC"]:
                entities["Locations"].append(entity_data)
            elif ent.label_ == "DATE":
                entities["Dates"].append(entity_data)
            elif ent.label_ in ["CARDINAL", "QUANTITY"]:
                entities["Quantities"].append(entity_data)
            else:
                entities["Other"].append(entity_data)
    return entities

def disambiguate_entities(self, entities, context_text):
    # Process the context with spaCy for linguistic features
    context_doc = self.nlp(context_text)
    # Initialize disambiguated entities
    disambiguated = {
        "Locations": [],
        "Dates": [],
        "Quantities": [],
        "Other": []
    }
    # Process each category
    for category, entity_list in entities.items():
        for entity in entity_list:
            # Skip high-confidence entities (they're likely correct)
            if entity.get("confidence", 0) > 0.95:
                disambiguated[category].append(entity)
                continue
            # For locations with medium confidence, check if they might be person names
            if category == "Locations" and entity.get("confidence", 0) < 0.9:
                # Check if this appears as a person name in the context
                if self._is_likely_person(entity["text"], context_doc):
                    # Move to Other category and mark as disambiguated
                    entity["disambiguated"] = True
                    entity["original_category"] = "Locations"
                    disambiguated["Other"].append(entity)
                else:
                    disambiguated[category].append(entity)
            # For dates with low confidence, verify format
            elif category == "Dates" and entity.get("confidence", 0) < 0.85:
                if self._verify_date_format(entity["text"]):
                    disambiguated[category].append(entity)
                else:
                    # Move to Other if it doesn't match date patterns
                    entity["disambiguated"] = True
                    entity["original_category"] = "Dates"
                    disambiguated["Other"].append(entity)
            else:
                disambiguated[category].append(entity)
    return disambiguated

def _is_likely_person(self, text, context_doc):
    # Check if an entity is more likely to be a person than a location
    # Look for the entity in the context
    for ent in context_doc.ents:
        if ent.text == text or text in ent.text or ent.text in text:
            # If it's labeled as a person in spaCy, return True
            if ent.label_ == "PERSON":
                return True
    # Check for common person name patterns
    # (This could be enhanced with a name database or more sophisticated checks)
    words = text.split()
    if len(words) == 2:  # Potential first name + last name
        # Check if first word is capitalized and not a common location prefix
        if (words[0][0].isupper() and 
            words[0].lower() not in ["north", "south", "east", "west", "new", "old", "san", "los"]):
            return True
    return False

def _verify_date_format(self, text):
    #Verify if text matches common date formats
    # Use regex patterns to check for date formats
    import re
    # Common date patterns
    date_patterns = [
        r'\d{1,2}/\d{1,2}/\d{2,4}',  # MM/DD/YYYY or DD/MM/YYYY
        r'\d{1,2}-\d{1,2}-\d{2,4}',  # MM-DD-YYYY or DD-MM-YYYY
        r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{1,2},? \d{4}\b',  # Month DD, YYYY
        r'\b\d{1,2} (?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{4}\b',  # DD Month YYYY
        r'\b\d{4}-\d{2}-\d{2}\b'     # YYYY-MM-DD (ISO format)
    ]
    for pattern in date_patterns:
        if re.search(pattern, text, re.IGNORECASE):
            return True
    return False




#This implementation adds Retrieval-Augmented Generation capabilities to ground the model's responses in factual information from a knowledge base, significantly reducing hallucination.
### 3. Chain-of-Thought Reasoning

def generate_with_reasoning(self, text, query):
    """
    Generate responses using chain-of-thought reasoning to reduce hallucinations
    """
    # Create a prompt that encourages step-by-step reasoning
    cot_prompt = f"""
    Based on the following text, answer the query step by step.
    
    Text: {text}
    
    Query: {query}
    
    Let's think through this systematically:
    1. First, identify what information from the text is relevant to the query
    2. Then, determine what facts we can extract from the text
    3. Finally, formulate an answer based only on these facts
    
    If the text doesn't contain information to answer the query, state that explicitly.
    """
    
    # Process with T5 model
    input_ids = self.t5_tokenizer(cot_prompt, return_tensors="pt").input_ids 
    # Generate with constraints
    outputs = self.t5_model.generate(
        input_ids,
        max_length=1024,
        num_beams=4,
        temperature=0.7,  # Slightly higher temperature for more detailed reasoning
        no_repeat_ngram_size=3,
        early_stopping=True
    )
    reasoning = self.t5_tokenizer.decode(outputs[0], skip_special_tokens=True)
    return reasoning

#This implementation uses chain-of-thought prompting to encourage the model to break down its reasoning process, which helps reveal and prevent logical errors that lead to hallucinations.
### 4. Fact Verification System

def verify_generated_content(self, source_text, generated_text):
    """
    Verify that generated content doesn't contradict or hallucinate beyond source text
    """
    # Extract entities from source and generated text
    source_entities = self.extract_entities_with_confidence(source_text)
    generated_entities = self.extract_entities_with_confidence(generated_text)
    # Track potential hallucinations
    hallucinations = []
    # Check each category of entities
    for category in generated_entities:
        for gen_entity in generated_entities[category]:
            # Skip low-confidence entities
            if gen_entity.get("confidence", 0) < 0.7:
                continue
            # Check if this entity appears in source
            found_in_source = False
            for src_entity in source_entities.get(category, []):
                # Check for exact or partial matches
                if (gen_entity["text"] == src_entity["text"] or 
                    gen_entity["text"] in src_entity["text"] or 
                    src_entity["text"] in gen_entity["text"]):
                    found_in_source = True
                    break
            # If not found in the same category, check other categories
            if not found_in_source:
                for other_category, src_entities in source_entities.items():
                    if other_category == category:
                        continue
                    for src_entity in src_entities:
                        if (gen_entity["text"] == src_entity["text"] or 
                            gen_entity["text"] in src_entity["text"] or 
                            src_entity["text"] in gen_entity["text"]):
                            found_in_source = True
                            # Note category mismatch but not a hallucination
                            break
                    if found_in_source:
                        break
            # If still not found, it might be a hallucination
            if not found_in_source:
                hallucinations.append({
                    "entity": gen_entity["text"],
                    "category": category,
                    "confidence": gen_entity.get("confidence", 0)
                })
    # Check for factual contradictions using NLI
    contradictions = self._check_contradictions(source_text, generated_text)
    return {
        "is_verified": len(hallucinations) == 0 and len(contradictions) == 0,
        "hallucinations": hallucinations,
        "contradictions": contradictions
    }

def _check_contradictions(self, source_text, generated_text):
    """
    Check for factual contradictions between source and generated text
    """
    # This could be implemented using a Natural Language Inference model
    # For simplicity, we'll use a basic approach here
    # Extract sentences from both texts
    source_sentences = [sent.strip() for sent in source_text.split('.') if sent.strip()]
    generated_sentences = [sent.strip() for sent in generated_text.split('.') if sent.strip()]
    contradictions = []
    # For a more sophisticated implementation, you would use an NLI model here
    # This is a placeholder for the concept
    return contradictions

## Testing and Evaluation

### NER Evaluation


def evaluate_ner(self, test_data):
    """
    Evaluate NER performance on test data
    
    test_data should be a list of dicts with:
    - "text": the input text
    - "entities": ground truth entities
    """
    results = {
        "precision": 0,
        "recall": 0,
        "f1": 0,
        "by_category": {}
    }
    
    total_true_positives = 0
    total_false_positives = 0
    total_false_negatives = 0
    
    # Initialize category-specific metrics
    categories = ["Locations", "Dates", "Quantities", "Other"]
    for category in categories:
        results["by_category"][category] = {
            "precision": 0,
            "recall": 0,
            "f1": 0,
            "true_positives": 0,
            "false_positives": 0,
            "false_negatives": 0
        }
    
    # Process each test example
    for example in test_data:
        text = example["text"]
        ground_truth = example["entities"]
        
        # Get predictions
        predictions = self.extract_entities_with_confidence(text)
        
        # Evaluate by category
        for category in categories:
            true_entities = set([e["text"].lower() for e in ground_truth.get(category, [])])
            pred_entities = set([e["text"].lower() for e in predictions.get(category, [])])
            
            # Calculate metrics
            true_positives = len(true_entities.intersection(pred_entities))
            false_positives = len(pred_entities - true_entities)
            false_negatives = len(true_entities - pred_entities)
            
            # Update category metrics
            results["by_category"][category]["true_positives"] += true_positives
            results["by_category"][category]["false_positives"] += false_positives
            results["by_category"][category]["false_negatives"] += false_negatives
            
            # Update totals
            total_true_positives += true_positives
            total_false_positives += false_positives
            total_false_negatives += false_negatives
    
    # Calculate overall metrics
    if total_true_positives + total_false_positives > 0:
        results["precision"] = total_true_positives / (total_true_positives + total_false_positives)
    
    if total_true_positives + total_false_negatives > 0:
        results["recall"] = total_true_positives / (total_true_positives + total_false_negatives)
    
    if results["precision"] + results["recall"] > 0:
        results["f1"] = 2 * (results["precision"] * results["recall"]) / (results["precision"] + results["recall"])
    
    # Calculate category-specific metrics
    for category in categories:
        cat_tp = results["by_category"][category]["true_positives"]
        cat_fp = results["by_category"][category]["false_positives"]
        cat_fn = results["by_category"][category]["false_negatives"]
        
        if cat_tp + cat_fp > 0:
            results["by_category"][category]["precision"] = cat_tp / (cat_tp + cat_fp)
        
        if cat_tp + cat_fn > 0:
            results["by_category"][category]["recall"] = cat_tp / (cat_tp + cat_fn)
        
        p = results["by_category"][category]["precision"]
        r = results["by_category"][category]["recall"]
        if p + r > 0:
            results["by_category"][category]["f1"] = 2 * (p * r) / (p + r)
    
    return results

### Hallucination Evaluation


def evaluate_hallucination(self, test_data):
    """
    Evaluate hallucination rate on test data
    
    test_data should be a list of dicts with:
    - "source": the source text
    - "query": the query to generate a response for
    """
    results = {
        "hallucination_rate": 0,
        "by_type": {
            "fact_conflicts": 0,
            "input_conflicts": 0,
            "context_conflicts": 0
        },
        "examples": []
    }
    
    total_hallucinations = 0
    
    for example in test_data:
        source = example["source"]
        query = example["query"]
        
        # Generate response
        response = self.generate_with_rag(query)
        
        # Verify response
        verification = self.verify_generated_content(source, response)
        
        # Track results
        if not verification["is_verified"]:
            total_hallucinations += 1
            
            # Categorize hallucination types
            if verification.get("contradictions"):
                results["by_type"]["fact_conflicts"] += 1
            
            if verification.get("hallucinations"):
                results["by_type"]["input_conflicts"] += 1
            
            # Store example for analysis
            results["examples"].append({
                "source": source,
                "query": query,
                "response": response,
                "verification": verification
            })
    
    # Calculate overall hallucination rate
    if test_data:
        results["hallucination_rate"] = total_hallucinations / len(test_data)
    
    return results