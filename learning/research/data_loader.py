from scripts.download_math500 import download_math500
from scripts.download_popQA import download_popQA


def load_reasoning100(output_file_name:str) -> list:
    dataset = download_math500(output="math500.json") ## here we are downlaoding the dataset

    math100_problems = dataset[:100] ## here taking only 100 problems from the dataset.

    problems_only = [] ## keeping only the problems from dataset metadata..
    for problem in math100_problems: ## looping over these 100 math problems
        only_problem = problem["problem"] ## extracting problem
        problems_only.append(only_problem)  ## appending to problems_only list

    
    reasoning_100 = []  ## storing the reasoning_only problems with additional meta-data we need
    for index, problem in enumerate(problems_only):  ## loopiing over the already extracted problems_only--getting the index and value also..to add additional meta-data.
        strucuted_data = {"id": index + 1,
         "problem": problem,
         "label": "reasoning"
         }  ## additing the label and id here..
        
        reasoning_100.append(strucuted_data) ## appending these dict's one-by-one to reasoning_problems list.

    return reasoning_100  ## returning this reasoning_problems list..
        


## this is the memorization function...
def load_memorization100(output_file_name:str) -> list: 
    pop_QA_dataset = download_popQA(output="popQA.json") ## downloading 

    popQA_question = pop_QA_dataset[:100] ## taking only 100 problems
     
    questions_only = []
    for question in popQA_question:
        question_only = question["question"]

        questions_only.append(question_only)

    
    memorization_100 = []
    for index, value in enumerate(questions_only):
        structured_data = {"id": index + 1,
         "question": value,
         "label": "memorization"}
        
        memorization_100.append(structured_data)

    return memorization_100
        




