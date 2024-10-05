import json
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM
from pinecone_text.sparse import SpladeEncoder


class InferlessPythonModel:

    # Implement the Load function here for the model
    def initialize(self):
        self.splade = SpladeEncoder(max_seq_length=256, device="cuda" if torch.cuda.is_available() else "cpu")

    
    # Function to perform inference 
    def infer(self, inputs):
        # inputs is a dictonary where the keys are input names and values are actual input data
        # e.g. in the below code the input name is "prompt"
        prompt = inputs["prompt"]
        
        # Encode the prompt using SPLADE
        sparse_vector = self.splade.encode_queries(prompt)
        
        # Convert the sparse vector to a dictionary for JSON serialization
        result = {
            "indices": sparse_vector["indices"],
            "values": sparse_vector["values"]
        }
        
        return {"sparse_vector": json.dumps(result)}

    # perform any cleanup activity here
    def finalize(self,args):
        self.splade = None
