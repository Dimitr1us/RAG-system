class Task:
    def __init__(self, description, tests):
        self.description = description
        self.tests = tests

    def description(self):
        return self.description
    
    def tests(self):
        return self.tests 