from .q_learner import QLearner
from .qtran_learner import QLearner as QTranLearner
from .riskq_learner import RiskQLearner

REGISTRY = {}

REGISTRY["q_learner"] = QLearner
REGISTRY["qtran_learner"] = QTranLearner
REGISTRY['riskq_learner'] = RiskQLearner
