REGISTRY = {}

from .rnn_agent import RNNAgent
from .iqn_rnn_agent import IQNRNNAgent
from .central_rnn_agent import CentralRNNAgent
from .riskq_agent import RISKRNNAgent 
from .riskq_ncqrdqn_agent import RISKNCQRDQNRNNAgent
from .riskq_agent_qrdqn import RISKRNNQRAgent

REGISTRY["rnn"] = RNNAgent
REGISTRY["iqn_rnn"] = IQNRNNAgent
REGISTRY["central_rnn"] = CentralRNNAgent
REGISTRY["risk_rnn"] = RISKRNNAgent 

REGISTRY["ncqrdqn_rnn"] = RISKNCQRDQNRNNAgent
REGISTRY["qrdqn_rnn"] = RISKRNNQRAgent

