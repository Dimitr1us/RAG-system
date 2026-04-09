class Task:
    def __init__(self, description, tests):
        self.description = description
        self.tests = tests

    def Description(self):
        return self.description
    
    def Tests(self):
        return self.tests 
    
    def Prompt(self):
        text =  f"""Реши задачу на питоне: {self.description}.\n Назад отправь только код самой задачи."""
        return text