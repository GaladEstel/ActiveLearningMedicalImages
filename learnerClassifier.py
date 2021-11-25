class learner:
    def __init__(self, X, y, estimator, query_strategy):
        self.estimator = estimator
        self.X = X
        self.y = y
        self.query_strategy = query_strategy

    #called several times incrementing the training data with the new data
    def teach(self):
        pass

    #add newly labeled instances to the dataset
    def add_training(self, X, y):
        pass

    #compute the uncertainty and give back the indices of the new data with highest uncertainty
    def query(self):
        pass
