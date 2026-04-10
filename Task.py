class Task:
    def __init__(self, description, function_name, tests,answer):
        self.description = description
        self.tests = tests
        self.answer = answer
        self.function_name = function_name

    def Description(self):
        return self.description
    
    def Tests(self):
        return self.tests 
    
    def Answer(self):
        return self.answer
    
    def Function_Name(self):
        return self.function_name
    
    def Prompt(self):
        text =  f"""Реши задачу на питоне: {self.description}\nПолучившуюся функцию назови {self.function_name}.\nНазад отправь только код самой задачи."""
        return text