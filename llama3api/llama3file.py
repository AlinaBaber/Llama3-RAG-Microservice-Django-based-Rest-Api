# from transformers import AutoTokenizer, AutoModelForCausalLM
# import torch
# from llama3api.model_loader import llm_model 
# from langchain import HuggingFacePipeline, PromptTemplate
# from langchain.chains import RetrievalQA
# from langchain.document_loaders import PyPDFLoader
# from langchain.embeddings.huggingface import HuggingFaceEmbeddings
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.vectorstores import Chroma
# from transformers import AutoTokenizer, TextStreamer, pipeline
# from transformers import AutoModelForCausalLM
# import requests


# class ChatbotModel:
#     def __init__(self):
#         self.model = llm_model
#         # self.model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
#         self.tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
#         self.SERPER_API_KEY = '476b5c3fcf8cb826dab5c8cfeb98e69d0d38edfe'


#      # Function to perform a search using Google Serper API
#     def serper_search(self, query):
#         url = 'https://google.serper.dev/search'
#         headers = {
#             'X-API-KEY': self.SERPER_API_KEY,
#             'Content-Type': 'application/json'
#         }
#         payload = {
#             'q': query
#         }
        
#         response = requests.post(url, json=payload, headers=headers)
#         if response.status_code == 200:
#                 return response.json()
#         else:
#             response.raise_for_status()



#     # Function to enhance LLM response using search results
#     def enhance_response_with_search(self, query, search_data):
#         print('\n\nSearch_data: ', search_data)
#         # Extract useful information from search results
#         snippets = [result['snippet'] for result in search_data.get('organic', [])]
#         top_snippets = " ".join(snippets[:5])  # Take top 5 snippets for context
        
#         # Extract key points from 'peopleAlsoAsk' section
#         people_also_ask = [item['snippet'] for item in search_data.get('peopleAlsoAsk', [])]
#         related_questions = " ".join(people_also_ask[:5])  # Take top 5 related questions and answers
    
#         # Combine the information
#         combined_context = f"{top_snippets} {related_questions}"
    
#         print('\n\nCombined_context: ', combined_context)

#         return combined_context
    
#         # # Prepare a prompt for the LLM
#         # prompt = f"User query: {query}\n\n"
#         # prompt += f"Here is some relevant information from the web:\n{combined_context}\n\n"
#         # prompt += "Provide a detailed and accurate response to the user query. Consider the relevant information if required."
    
#         # # Get response from the LLM
#         # tokenizer = self.tokenizer
#         # model = self.model
#         # inputs = tokenizer(prompt, return_tensors='pt').to(self.model.device)
#         # outputs = model.generate(inputs.input_ids, max_length=5000, num_return_sequences=1)

#         # response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
#         # del inputs
#         # del outputs
#         # torch.cuda.empty_cache()

#         # response= 'Chatbot response: '+response
        
#         # return response


#     # # Main function to get enhanced response
#     # def get_enhanced_response(self, user_query):
#     #     try:
#     #         # Perform search using Serper API
#     #         search_results = self.serper_search(user_query)
#     #         # Get enhanced response from the LLM
#     #         enhanced_response = self.enhance_response_with_search(user_query, search_results)
#     #         return enhanced_response
#     #     except Exception as e:
#     #         return f"An error occurred: {str(e)}"



     
#     def generate_response(self, user_input):


#         search_results = self.serper_search(user_input)

#         enhanced_response = self.enhance_response_with_search(user_input, search_results)

#         user_input += f"\n\nHere is some relevant information from the web:\n{enhanced_response}\n\n"
        
#         messages = [
#             {"role": "system", "content": "You are the knowledgeable chatbot to answer the user query perfectly"},
#             {"role": "user", "content": user_input},
#         ]
        
#         input_ids = self.tokenizer.apply_chat_template(
#             messages,
#             add_generation_prompt=True,
#             return_tensors="pt"
#         ).to(self.model.device)

#         eos_token_id = self.tokenizer.eos_token_id

#         outputs = self.model.generate(
#             input_ids,
#             max_new_tokens=500,
#             eos_token_id=eos_token_id,
#             do_sample=True,
#             temperature=0.6,
#             top_p=0.9,
#         )
#         response = outputs[0][input_ids.shape[-1]:]
#         result = self.tokenizer.decode(response, skip_special_tokens=True)
        
#         # Clean up to free GPU memory
#         del input_ids
#         del outputs
#         torch.cuda.empty_cache()
        
#         return result


#     def initialize_components(self,file_path,question,user_prompt=None,botname="Chatbot:"):

#             search_results = self.serper_search(user_input)

#             enhanced_response = self.enhance_response_with_search(user_input, search_results)
        
#             user_prompt += f"\n\nHere is some relevant information from the web:\n{enhanced_response}\n\n"


#             loader = PyPDFLoader(file_path)
#             data = loader.load()
        
#             embeddings = HuggingFaceEmbeddings(
#                 model_name="sentence-transformers/all-MiniLM-L6-v2",
#                 model_kwargs={"device": "cuda"}
#             )
        
#             text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=64)
#             texts = text_splitter.split_documents(data)
        
#             db = Chroma.from_documents(texts, embeddings)
        
        
#             streamer = TextStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
#             text_pipeline = pipeline(
#                 "text-generation",
#                 model=self.model,
#                 tokenizer=self.tokenizer,
#                 max_new_tokens=1024,
#                 temperature=0.5,
#                 top_p=0.95,
#                 repetition_penalty=1.15,
#                 streamer=streamer,
#             )
#             llm = HuggingFacePipeline(pipeline=text_pipeline, model_kwargs={"temperature": 0.7})
            
#             # Default user prompt if none is provided
#             if user_prompt is None:
#                 user_prompt = """You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.
#                 If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."""
    
#             context_template = f"\nContext: {{context}}\nUser: {{question}}\n{botname}"

#         # Combine user-defined prompt with the context template
#             combined_prompt = f"{user_prompt}{context_template}"
        
#             prompt = PromptTemplate(template=combined_prompt, input_variables=["context", "question"])
        
#             qa_chain = RetrievalQA.from_chain_type(
#                 llm=llm,
#                 chain_type="stuff",
#                 retriever=db.as_retriever(search_kwargs={"k": 2}),
#                 return_source_documents=True,
#                 chain_type_kwargs={"prompt": prompt},
#             )
        
#                     # Execute a query (you can customize this part as needed)
#             query_result = qa_chain(question)

#             response_content = query_result['result']
#             answer_prefix = botname
#             answer_start_index = response_content.find(answer_prefix)
#             if answer_start_index != -1:
#                 answer = response_content[answer_start_index + len(answer_prefix):].strip()
#                 print(answer)
#                 return answer
#             else:
#                 print("No answer found in the response.")
#                 return response_content



#     def serper_search(self, query):
#         url = 'https://google.serper.dev/search'
#         headers = {
#             'X-API-KEY': self.SERPER_API_KEY,
#             'Content-Type': 'application/json'
#         }
#         payload = {
#             'q': query
#         }
        
#         response = requests.post(url, json=payload, headers=headers)
#         if response.status_code == 200:
#                 return response.json()
#         else:
#             response.raise_for_status()

    
#     # Function to enhance LLM response using search results
#     def enhance_response_with_search(self, query, search_data):
#         print('\n\nSearch_data: ', search_data)
#         # Extract useful information from search results
#         snippets = [result['snippet'] for result in search_data.get('organic', [])]
#         top_snippets = " ".join(snippets[:5])  # Take top 5 snippets for context
        
#         # Extract key points from 'peopleAlsoAsk' section
#         people_also_ask = [item['snippet'] for item in search_data.get('peopleAlsoAsk', [])]
#         related_questions = " ".join(people_also_ask[:5])  # Take top 5 related questions and answers
    
#         # Combine the information
#         combined_context = f"{top_snippets} {related_questions}"
    
#         print('\n\nCombined_context: ', combined_context)
    
#         # Prepare a prompt for the LLM
#         prompt = f"User query: {query}\n\n"
#         prompt += f"Here is some relevant information from the web:\n{combined_context}\n\n"
#         prompt += "Provide a detailed and accurate response to the user query. Consider the relevant information if required."
    
#         # Get response from the LLM
#         tokenizer = self.tokenizer
#         model = self.model
#         inputs = tokenizer(prompt, return_tensors='pt').to(self.model.device)
#         outputs = model.generate(inputs.input_ids, max_length=50000, num_return_sequences=1)

#         response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
#         del inputs
#         del outputs
#         torch.cuda.empty_cache()

#         response= 'Chatbot response: '+response
        
#         return response








# from transformers import AutoTokenizer, AutoModelForCausalLM
# import torch
# from llama3api.model_loader import llm_model 
# from langchain import HuggingFacePipeline, PromptTemplate
# from langchain.chains import RetrievalQA
# from langchain.document_loaders import PyPDFLoader
# from langchain.embeddings.huggingface import HuggingFaceEmbeddings
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.vectorstores import Chroma
# from transformers import AutoTokenizer, TextStreamer, pipeline
# from transformers import AutoModelForCausalLM
# import requests
# import json

# class ChatbotModel:
#     def __init__(self):
#         self.model = llm_model
#         self.tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
#         self.SERPER_API_KEY = '476b5c3fcf8cb826dab5c8cfeb98e69d0d38edfe'

#     # Function to perform a search using Google Serper API
#     def serper_search(self, query):
#         url = 'https://google.serper.dev/search'
#         headers = {
#             'X-API-KEY': self.SERPER_API_KEY,
#             'Content-Type': 'application/json'
#         }
#         payload = {
#             'q': query
#         }
        
#         response = requests.post(url, json=payload, headers=headers)
#         if response.status_code == 200:
#             return response.json()
#         else:
#             response.raise_for_status()

#     # Function to enhance LLM response using search results
#     def enhance_response_with_search(self, query, search_data):
#         # Extract useful information from search results
#         snippets = [result['snippet'] for result in search_data.get('organic', [])]
#         top_snippets = " ".join(snippets[:5])  # Take top 5 snippets for context
        
#         # Extract key points from 'peopleAlsoAsk' section
#         people_also_ask = [item['snippet'] for item in search_data.get('peopleAlsoAsk', [])]
#         related_questions = " ".join(people_also_ask[:5])  # Take top 5 related questions and answers
    
#         # Combine the information
#         combined_context = f"{top_snippets} {related_questions}"
    
#         return combined_context

#     # Function to clean and format the response text
#     def clean_response(self, response_text):
#         import re

#         # Replace escaped newlines with actual newlines
#         response_text = response_text.replace('\\n', '\n')

#         # Fix indentation for code blocks
#         def fix_indentation(match):
#             code_block = match.group(0)
#             lines = code_block.split('\n')
#             fixed_lines = [lines[0]] + [line if line.startswith('    ') else '    ' + line for line in lines[1:]]
#             return '\n'.join(fixed_lines)

#         response_text = re.sub(r'```.*?```', fix_indentation, response_text, flags=re.DOTALL)

#         return response_text
    
#     # Function to generate the chatbot response
#     def generate_response(self, user_input):
#         search_results = self.serper_search(user_input)
#         enhanced_response = self.enhance_response_with_search(user_input, search_results)
#         user_input += f"\n\nHere is some relevant information from the web:\n{enhanced_response}\n\n"
        
#         messages = [
#             {"role": "system", "content": "You are the knowledgeable chatbot to answer the user query perfectly"},
#             {"role": "user", "content": user_input},
#         ]
        
#         input_ids = self.tokenizer.apply_chat_template(
#             messages,
#             add_generation_prompt=True,
#             return_tensors="pt"
#         ).to(self.model.device)

#         eos_token_id = self.tokenizer.eos_token_id

#         outputs = self.model.generate(
#             input_ids,
#             max_new_tokens=500,
#             eos_token_id=eos_token_id,
#             do_sample=True,
#             temperature=0.6,
#             top_p=0.9,
#         )
#         response = outputs[0][input_ids.shape[-1]:]
#         result = self.tokenizer.decode(response, skip_special_tokens=True)
        
#         # Clean up to free GPU memory
#         del input_ids
#         del outputs
#         torch.cuda.empty_cache()
        
#         # Clean the response text
#         cleaned_result = self.clean_response(result)
        
#         return cleaned_result

#     # Function to initialize components and execute a query
#     def initialize_components(self, file_path, question, user_prompt=None, botname="Chatbot:"):
#         search_results = self.serper_search(question)
#         enhanced_response = self.enhance_response_with_search(question, search_results)
        
#         user_prompt = user_prompt if user_prompt else ""
#         user_prompt += f"\n\nHere is some relevant information from the web:\n{enhanced_response}\n\n"

#         loader = PyPDFLoader(file_path)
#         data = loader.load()
    
#         embeddings = HuggingFaceEmbeddings(
#             model_name="sentence-transformers/all-MiniLM-L6-v2",
#             model_kwargs={"device": "cuda"}
#         )
    
#         text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=64)
#         texts = text_splitter.split_documents(data)
    
#         db = Chroma.from_documents(texts, embeddings)
    
#         streamer = TextStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
#         text_pipeline = pipeline(
#             "text-generation",
#             model=self.model,
#             tokenizer=self.tokenizer,
#             max_new_tokens=1024,
#             temperature=0.5,
#             top_p=0.95,
#             repetition_penalty=1.15,
#             streamer=streamer,
#         )
#         llm = HuggingFacePipeline(pipeline=text_pipeline, model_kwargs={"temperature": 0.7})
        
#         # Default user prompt if none is provided
#         if user_prompt is None:
#             user_prompt = """You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.
#             If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."""

#         context_template = f"\nContext: {{context}}\nUser: {{question}}\n{botname}"

#         # Combine user-defined prompt with the context template
#         combined_prompt = f"{user_prompt}{context_template}"
    
#         prompt = PromptTemplate(template=combined_prompt, input_variables=["context", "question"])
    
#         qa_chain = RetrievalQA.from_chain_type(
#             llm=llm,
#             chain_type="stuff",
#             retriever=db.as_retriever(search_kwargs={"k": 2}),
#             return_source_documents=True,
#             chain_type_kwargs={"prompt": prompt},
#         )
    
#         # Execute a query
#         query_result = qa_chain(question)
#         response_content = query_result['result']
#         answer_prefix = botname
#         answer_start_index = response_content.find(answer_prefix)
#         if answer_start_index != -1:
#             answer = response_content[answer_start_index + len(answer_prefix):].strip()
#             cleaned_answer = self.clean_response(answer)
#             print(cleaned_answer)
#             return cleaned_answer
#         else:
#             print("No answer found in the response.")
#             return response_content

# # # Example of using the ChatbotModel class
# # chatbot = ChatbotModel()
# # user_input = "How to implement a content-based movie recommendation system in Python?"
# # response = chatbot.generate_response(user_input)
# # print(response)

# # file_path = "path/to/your/document.pdf"
# # question = "What are the key points in this document?"
# # response = chatbot.initialize_components(file_path, question)
# # print(response)






from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from llama3api.model_loader import llm_model 
from langchain import HuggingFacePipeline, PromptTemplate
from langchain.chains import RetrievalQA
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from transformers import AutoTokenizer, TextStreamer, pipeline
import requests


class ChatbotModel:
    def __init__(self):
        self.model = llm_model
        self.tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
        self.SERPER_API_KEY = '476b5c3fcf8cb826dab5c8cfeb98e69d0d38edfe'


    def serper_search(self, query):
        url = 'https://google.serper.dev/search'
        headers = {
            'X-API-KEY': self.SERPER_API_KEY,
            'Content-Type': 'application/json'
        }
        payload = {'q': query}
        
        response = requests.post(url, json=payload, headers=headers)
        if response.status_code == 200:
            return response.json()
        else:
            response.raise_for_status()


    def enhance_response_with_search(self, query, search_data):
        snippets = [result['snippet'] for result in search_data.get('organic', [])]
        top_snippets = " ".join(snippets[:5])
        
        people_also_ask = [item['snippet'] for item in search_data.get('peopleAlsoAsk', [])]
        related_questions = " ".join(people_also_ask[:5])
    
        combined_context = f"{top_snippets} {related_questions}"
        return combined_context


    def generate_response(self, user_input):
        search_results = self.serper_search(user_input)
        enhanced_response = self.enhance_response_with_search(user_input, search_results)

        user_input += f"\n\nHere is some relevant information from the web:\n{enhanced_response}\n\n"
        
        messages = [
            {"role": "system", "content": "You are the knowledgeable chatbot to answer the user query perfectly"},
            {"role": "user", "content": user_input},
        ]
        
        input_ids = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to(self.model.device)

        eos_token_id = self.tokenizer.eos_token_id

        outputs = self.model.generate(
            input_ids,
            max_new_tokens=500,
            eos_token_id=eos_token_id,
            do_sample=True,
            temperature=0.6,
            top_p=0.9,
        )
        response = outputs[0][input_ids.shape[-1]:]
        result = self.tokenizer.decode(response, skip_special_tokens=True)
        
        del input_ids
        del outputs
        torch.cuda.empty_cache()
        
        return self.format_response(result)


    def format_response(self, response):
        import re

        # Clean and structure the response
        response = response.replace("\\n", "\n")
        response = re.sub(r'\s+', ' ', response).strip()

        # Format the response
        formatted_response = ""
        code_block = False

        for line in response.split("\n"):
            if line.startswith("```"):
                code_block = not code_block
                formatted_response += "\n" + line + "\n"
            elif code_block:
                formatted_response += line + "\n"
            else:
                formatted_response += "\n" + line.strip() + "\n"
        
        return formatted_response.strip()


    def initialize_components(self, file_path, question, user_prompt=None, botname="Chatbot:"):
        search_results = self.serper_search(question)
        enhanced_response = self.enhance_response_with_search(question, search_results)

        user_prompt = user_prompt or """You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature. If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."""
        user_prompt += f"\n\nHere is some relevant information from the web:\n{enhanced_response}\n\n"

        loader = PyPDFLoader(file_path)
        data = loader.load()
        
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": "cuda"}
        )
        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=64)
        texts = text_splitter.split_documents(data)
        
        db = Chroma.from_documents(texts, embeddings)
        
        streamer = TextStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
        text_pipeline = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            max_new_tokens=1024,
            temperature=0.5,
            top_p=0.95,
            repetition_penalty=1.15,
            streamer=streamer,
        )
        llm = HuggingFacePipeline(pipeline=text_pipeline, model_kwargs={"temperature": 0.7})
        
        context_template = f"\nContext: {{context}}\nUser: {{question}}\n{botname}"
        combined_prompt = f"{user_prompt}{context_template}"
        
        prompt = PromptTemplate(template=combined_prompt, input_variables=["context", "question"])
        
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=db.as_retriever(search_kwargs={"k": 2}),
            return_source_documents=True,
            chain_type_kwargs={"prompt": prompt},
        )
        
        query_result = qa_chain(question)
        response_content = query_result['result']
        answer_prefix = botname
        answer_start_index = response_content.find(answer_prefix)
        
        if answer_start_index != -1:
            answer = response_content[answer_start_index + len(answer_prefix):].strip()
            return self.format_response(answer)
        else:
            return self.format_response(response_content)

# # # Example usage
# # chatbot = ChatbotModel()
# # response = chatbot.generate_response("Explain a content-based movie recommendation system with an example code.")
# # print(response)




# from transformers import AutoTokenizer, AutoModelForCausalLM
# import torch
# from llama3api.model_loader import llm_model 
# from langchain import HuggingFacePipeline, PromptTemplate
# from langchain.chains import RetrievalQA
# from langchain.document_loaders import PyPDFLoader
# from langchain.embeddings.huggingface import HuggingFaceEmbeddings
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.vectorstores import Chroma
# from transformers import AutoTokenizer, TextStreamer, pipeline
# import requests
# import json


# class ChatbotModel:
#     def __init__(self):
#         self.model = llm_model
#         self.tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
#         self.SERPER_API_KEY = '476b5c3fcf8cb826dab5c8cfeb98e69d0d38edfe'


#     def serper_search(self, query):
#         url = 'https://google.serper.dev/search'
#         headers = {
#             'X-API-KEY': self.SERPER_API_KEY,
#             'Content-Type': 'application/json'
#         }
#         payload = {'q': query}
        
#         response = requests.post(url, json=payload, headers=headers)
#         if response.status_code == 200:
#             return response.json()
#         else:
#             response.raise_for_status()


#     def enhance_response_with_search(self, query, search_data):
#         snippets = [result['snippet'] for result in search_data.get('organic', [])]
#         top_snippets = " ".join(snippets[:5])
        
#         people_also_ask = [item['snippet'] for item in search_data.get('peopleAlsoAsk', [])]
#         related_questions = " ".join(people_also_ask[:5])
    
#         combined_context = f"{top_snippets} {related_questions}"
#         return combined_context


#     def generate_response(self, user_input):
#         search_results = self.serper_search(user_input)
#         enhanced_response = self.enhance_response_with_search(user_input, search_results)

#         user_input += f"\n\nHere is some relevant information from the web:\n{enhanced_response}\n\n"
        
#         messages = [
#             {"role": "system", "content": "You are the knowledgeable chatbot to answer the user query perfectly"},
#             {"role": "user", "content": user_input},
#         ]
        
#         input_ids = self.tokenizer.apply_chat_template(
#             messages,
#             add_generation_prompt=True,
#             return_tensors="pt"
#         ).to(self.model.device)

#         eos_token_id = self.tokenizer.eos_token_id

#         outputs = self.model.generate(
#             input_ids,
#             max_new_tokens=500,
#             eos_token_id=eos_token_id,
#             do_sample=True,
#             temperature=0.6,
#             top_p=0.9,
#         )
#         response = outputs[0][input_ids.shape[-1]:]
#         result = self.tokenizer.decode(response, skip_special_tokens=True)
        
#         del input_ids
#         del outputs
#         torch.cuda.empty_cache()
        
#         return self.clean_response(result)


#     def clean_response(self, response_text):
#         # Perform any necessary cleanup
#         # For example, replace '\n' with actual newlines and handle special formatting
#         clean_text = response_text.replace('\\n', '\n')
        
#         # Additional formatting can be added here if needed
#         return clean_text


#     def initialize_components(self, file_path, question, user_prompt=None, botname="Chatbot:"):
#         search_results = self.serper_search(question)
#         enhanced_response = self.enhance_response_with_search(question, search_results)

#         user_prompt = user_prompt or """You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature. If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."""
#         user_prompt += f"\n\nHere is some relevant information from the web:\n{enhanced_response}\n\n"

#         loader = PyPDFLoader(file_path)
#         data = loader.load()
        
#         embeddings = HuggingFaceEmbeddings(
#             model_name="sentence-transformers/all-MiniLM-L6-v2",
#             model_kwargs={"device": "cuda"}
#         )
        
#         text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=64)
#         texts = text_splitter.split_documents(data)
        
#         db = Chroma.from_documents(texts, embeddings)
        
#         streamer = TextStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
#         text_pipeline = pipeline(
#             "text-generation",
#             model=self.model,
#             tokenizer=self.tokenizer,
#             max_new_tokens=1024,
#             temperature=0.5,
#             top_p=0.95,
#             repetition_penalty=1.15,
#             streamer=streamer,
#         )
#         llm = HuggingFacePipeline(pipeline=text_pipeline, model_kwargs={"temperature": 0.7})
        
#         context_template = f"\nContext: {{context}}\nUser: {{question}}\n{botname}"
#         combined_prompt = f"{user_prompt}{context_template}"
        
#         prompt = PromptTemplate(template=combined_prompt, input_variables=["context", "question"])
        
#         qa_chain = RetrievalQA.from_chain_type(
#             llm=llm,
#             chain_type="stuff",
#             retriever=db.as_retriever(search_kwargs={"k": 2}),
#             return_source_documents=True,
#             chain_type_kwargs={"prompt": prompt},
#         )
        
#         query_result = qa_chain(question)
#         response_content = query_result['result']
#         answer_prefix = botname
#         answer_start_index = response_content.find(answer_prefix)
        
#         if answer_start_index != -1:
#             answer = response_content[answer_start_index + len(answer_prefix):].strip()
#             return self.clean_response(answer)
#         else:
#             return self.clean_response(response_content)


# # # Example usage
# # chatbot = ChatbotModel()
# # response = chatbot.generate_response("Explain a content-based movie recommendation system with an example code.")
# # print(response)

