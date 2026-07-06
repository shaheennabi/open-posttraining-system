
## ok, think like this, 
# 1. first we  will see what is the input
# 2. What output do I need
# 3. what information must I remember, while processing the input 
# 4. what operation repeats 
# 5. what data structure naturally stores the remembered information 


## Evaluation Aggregation problem

# you are evaluating several langauge models, each evaluation result has this format 

"""
results = [
    "qwen,math,correct",
    "llama,math,wrong",
    "qwen,code,correct",
    "mistral,math,correct",
    "llama,code,correct",
    "qwen,math,wrong",
    "mistral,code,wrong",
]

"""

## ok each string, contains --> model, task, status

## ok write a function def best_model(results):

## return the model with the highest accuracy 

## ok, accuracy is total_sample_of_model/correct_samples_of_model
## if two models have the same accuracy, return the model with the larger total number of evaluated examples.
## if they are still tied, return the model whose name' comes first alphabetically.


#1. 
## inputs i have 
## ok, for now what i see, i see that there is the input, in list,, list as results ---> where i have three things: model, task, status --> in quotation 
## and i have lot of quotations --> where there is this, model, task, status in each quotation ...

#2. 
## outputs i need to get 
## ok now let's talk about the output, ---> return the model with the highest accuracy ... 
    ## here is the catch --> if two models have the same accuracy ----> return the model with the larger total number of evaluated examples
    ## ok, now another catch --> if they are still tied return the model ---> whose "name" comes first alphabeticallly....

#3.
# what information must I remember, while processing the input.
## i have to remember which models got called, how many times, and their related status (means model<-->how many times evaluated)
              ## and later to calculate the accuracy i have to remember ---> how many times model got it correct



#4.
## what operation repeats.
## ok, what i think --> that which operation repeats here is ---> 
  ## i think going through the results ---> and inside these results we have these quotations (strings) holding data ..
   ## i think we are iterating again and again and looking for data...


#5. 
## what data structure naturally stores the remembered information
## ok so for the data structure ---> i think i can use the hashmap to store the information 



results = [
    "qwen,math,correct",
    "llama,math,wrong",
    "qwen,code,correct",
    "mistral,math,correct",
    "llama,code,correct",
    "qwen,math,wrong",
    "mistral,code,wrong",
]

def best_models(results):
    stats = {}

    for i in results:
        splitted_text = i.split(",")
        splitted_model, model_evaluated = splitted_text[0], splitted_text[2]

        if splitted_model in stats:
            if model_evaluated.startswith("c"):
                stats[splitted_model][0] += 1  ## 0 ---> correct
                stats[splitted_model][1] += 1  ## 1 ---> total
            else:
                stats[splitted_model][1] += 1

        else:
            stats[splitted_model] = [0,0] 
            if model_evaluated.startswith("c"):
                stats[splitted_model][0] += 1
                stats[splitted_model][1] += 1
            else:
                stats[splitted_model][1] += 1
        
    return stats
#print(best_models(results))




## ok, now our task is to monitor the model latency
# and here we are given the model latency logs, and it's the list of quotation strings, with model_name and latency_ms
## and here we have to write a function def slow_models(logs, threshold) ---> that will return all models whose latency is greater than the threshold

# ok for writing this function, we will again do it similarly--->  we did with the previous function (like follow the same rules)..
# 1. ok first we will see what is the input
# 2. what output do I need
# 3. what information must I remember, while processing the input.
# 4. what operation repeats.
# 5. what data structure naturally stores the remembered information..

## ok for the input we have this: 
"""
logs = [
    "qwen,120",
    "llama,200",
    "qwen,180",
    "mistral,90",
    "llama,100",
    "qwen,300",
    "mistral,110",
]
"""
#input
# 1. we are given model name and latency in milliseconds 

# output
# 2. ok to get the model, performing above the threshold, we have to ---> 
# take models latency --> and add them then divide by the times model is listed...there

## information  to remember while processing the input..
# 3. we have to remember like take every model, ---> with every model add this millisecond latency .
# also keep track of how many times the model got called

#4. for the operation that repeats,
## i  guess i am not going to go and iterate over the list of strings, again and again ... i wiill iterate ones ..and for  the repeating part
## it will be the model that will repeat, and it's latency if not algorithmically build properly... so best is to iterate ones and save data

## 5. what data structure stores the remembered information
## i think let's use the hashmap that will store the model name and it's frequency like add the model frequency. .. as you encounter
# the same model again and keep the track of counter as you get the same over in again and again..



logs = [
    "qwen,120",
    "llama,200",
    "qwen,180",
    "mistral,90",
    "llama,100",
    "qwen,300",
    "mistral,110",
]


def slow_models(logs, threshold):

    stats = {}
    model_names = []

    for i in logs:
        splitted_logs = i.split(",")
        model, latency = splitted_logs[0], splitted_logs[1] ## here we will get the model_name and it's latencies

        if model not in stats: ## it means we are seeing this model for the first time...
            stats[model] = [0, 0, 0] ## we will initialize the model values for the first time, then increament it. ## it will be like [latencies, count, average_latency]

        stats[model][0] += int(latency)  ## increament the latency
        stats[model][1] += 1  ## increament count
        stats[model][2] = stats[model][0] / stats[model][1]  ## get the average latency



        
        ## ok now we got the information we needed ---> now let's calculate the, things we need for output..
        ## first we will calculate average latency

    for model in stats:
        if stats[model][2] > threshold:
            model_names.append(model)
        
    return model_names

print(slow_models(logs, 130))
    
        

 



