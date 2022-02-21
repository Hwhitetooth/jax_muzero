"""Customized data structures."""
import collections


ActorOutput = collections.namedtuple('ActorOutput', [
    'action_tm1',
    'reward',
    'observation',
    'first',
    'last',
])


AgentOutput = collections.namedtuple('AgentOutput', [
    'state',
    'logits',
    'value_logits',
    'value',
    'reward_logits',
    'reward',
])


Params = collections.namedtuple('Params', [
    'encoder',
    'prediction',
    'transition',
])


Tree = collections.namedtuple(
    'Tree', [
        'state',
        'logits',
        'prob',
        'reward_logits',
        'reward',
        'value_logits',
        'value',
        'action_value',
        'depth',
        'parent',
        'parent_action',
        'child',
        'visit_count',
    ]
)
