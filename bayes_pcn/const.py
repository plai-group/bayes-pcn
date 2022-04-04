from enum import Enum
"""
NOTE: Enum value must be lowercase of its key!!!
"""


class ArgParseEnum(Enum):
    def __str__(self):
        return self.value

    @classmethod
    def get_enum_from_value(cls, value: str):
        return cls.__dict__[value.upper()]


class LayerLogProbStrat(ArgParseEnum):
    MAP = 'map'        # Assume no parameter uncertainty when doing energy minimization
    P_PRED = 'p_pred'  # Assume parameter uncertainty when doing energy minimization


class LayerSampleStrat(ArgParseEnum):
    MAP = 'map'        # Sample from posterior
    P_PRED = 'p_pred'  # Sample from posterior predictive via marginalization


class LayerUpdateStrat(ArgParseEnum):
    ML = 'ml'        # Gradient descent update on weights given activations
    BAYES = 'bayes'  # Conjugate Bayesian update on weights given activations


class EnsembleLogJointStrat(ArgParseEnum):
    SHARED = 'shared'          # Fit a single proposal distribution for all PCNets in the ensemble
    INDIVIDUAL = 'individual'  # Fit a proposal distribution per PCNet in the ensemble


class EnsembleProposalStrat(ArgParseEnum):
    MODE = 'mode'  # Proposal distribution always returns the mean
    FULL = 'full'  # Proposal distribution uses a full covariance matrix


class ActInitStrat(ArgParseEnum):
    FIXED = 'fixed'    # Initialize hidden activation values to 1 (0 doesn't work cuz gradient)
    RANDN = 'randn'    # Initialize hidden activation values to normal samples
    SAMPLE = 'sample'  # Initialize hidden activation values to model samples


class ActFn(ArgParseEnum):
    NONE = 'none'
    RELU = 'relu'
    GELU = 'gelu'
    SOFTMAX = 'softmax'
