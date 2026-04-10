class Task:
    def __init__(self, description, tests,answer):
        self.description = description
        self.tests = tests
        self.answer = answer

    def Description(self):
        return self.description
    
    def Tests(self):
        return self.tests 
    
    def Answer(self):
        return self.answer
    
    def Prompt(self):
        text =  f"""Реши задачу на питоне: {self.description}.\n Назад отправь только код самой задачи."""
        return text