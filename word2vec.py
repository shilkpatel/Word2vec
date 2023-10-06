import math
import typing
import random
import pickle



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
    
    def save_to_file(self,model_name):
        with open('C:/Users/shilp/OneDrive/Desktop/New Github/Word2vec/models/'+model_name,'wb') as data_file:
            pickle.dump(self.vector,data_file)

    def load_from_file(self,model_name):
        with open('C:/Users/shilp/OneDrive/Desktop/New Github/Word2vec/models/'+model_name,'rb') as data_file:
            self.vector=pickle.load(data_file)
        




class trainer:
    def __init__(self,learning_rate: float,model: word2vec):
        self.learning=learning_rate
        self.vec_model=model
        self.adjust={}
        self.last_average_error=1

        for i in self.vec_model.vector:
            self.adjust[i]=[[0 for j in range(model.number_of_embedding)] for k in range(2)]

    def learn_single_input(self,training_data):#training set is (main,context,label)
        main=self.vec_model.vector[training_data[0]][0]
        context=self.vec_model.vector[training_data[1]][1]
        result= word2vec.dot(main,context)
        result= word2vec.sigmoid(result)
        #error=float(training_data[2])-result
        error=result-float(training_data[2])
        error_to_vector=(2*error)*result*(1-result)

        #adjust the main vector
        for i in range(len(main)):
            self.adjust[training_data[0]][0][i]+=(error_to_vector*context[i]*(50*self.last_average_error)*-1)##removing minus
        #adjust in the context vector
        for i in range(len(context)):
            self.adjust[training_data[1]][1][i]+=(error_to_vector*main[i]*(50*self.last_average_error**2)*-1)##removing minus
        

        return (self.adjust,math.pow(error,2))

    def epoch(self,dataset,batch_size):
        sub_set=[]
        average_error=0
        #creates subset of dataset which will be fed into the model
        for i in range(batch_size):
            sub_set.append(random.choice(dataset))

        for i in sub_set:
            learn=self.learn_single_input(i)
            average_error+=learn[1]
            self.adjust=learn[0]
            
        
        
        for i in self.adjust:
            for j in range(len(self.adjust[i])):
                for k in range(len(self.adjust[i][j])):         
                    self.adjust[i][j][k]/=batch_size

        for i in self.vec_model.vector:
            for j in range(len(self.adjust[i])):
                for k in range(len(self.adjust[i][j])):
                    self.vec_model.vector[i][j][k]+=self.adjust[i][j][k]
                    self.adjust[i][j][k]=0
                        
        self.last_average_error=average_error/batch_size

        print("average_error:", average_error/batch_size)
        print("learning rate:", self.last_average_error*50)











        




    
    
