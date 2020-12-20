from agents.NonlinearQ import NonlinearQ

def getAgent(name):
    if name == 'NonlinearQ':
        return NonlinearQ

    raise NotImplementedError()
