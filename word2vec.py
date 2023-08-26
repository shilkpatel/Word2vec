import math
import typing



class word2vec:
    def __init__(self,vector_dict: dict[str,list[list[int]]],embed: int):
        self.vector=vector_dict
        self.number_of_embedding=embed
    
    def dot(vec1: list[float],vec2: list[float]) -> float:
        ans=0

        for i in range(len(vec1)):
            ans+=(vec1[i]*vec2[i])
        
        return ans

    def sigmoid(x: float) -> float:
        return (1/(1+math.exp(-x)))



class trainer:
    def __init__(self,learning_rate,model: word2vec):
        self.learning=learning_rate
        self.vec_model=model
        self.adjust={}

        for i in self.vec_model.vector:
            self.adjust[i]=[[0 for j in model.number_of_embedding] for k in range(2)]

    def learn_single_input(self,training_data):
        result= word2vec.dot(self.vec_model.vector[training_data[0]][0],self.vec_model.vector[training_data[1]][1])
        result= word2vec.sigmoid(result)
        error=(int(training_data[2])-result)**2
        
        




    
    
