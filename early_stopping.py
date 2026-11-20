"""Validation-based stopping state; independent of PPO and best-weight selection."""


def initial_state():
    # This plain dictionary is stored in the same atomic training checkpoint as
    # model/optimizer/RNG state. Never reset patience when resuming a run.
    return {'anchor_score': None, 'bad_validations': 0, 'monitoring': False,
            'last_validation_update': None, 'stopped': False}


def observe_validation(state, completed_updates, score, best_score, config):
    """Consume one successful validation (score=None means ineligible, not crashed).

    best_score is the lowest eligible score observed so far. It controls best
    weight selection elsewhere; anchor_score only controls meaningful progress.
    The first validation strictly AFTER min_updates counts toward patience.
    """
    if state['stopped']:
        return True
    previous = state['last_validation_update']
    if previous is not None and completed_updates <= previous:
        raise ValueError('Early stopping must consume each validation update exactly once')
    state['last_validation_update'] = completed_updates
    if not config['enabled']:
        return False
    if completed_updates <= config['min_updates']:
        state['anchor_score'] = best_score
        state['bad_validations'] = 0
        return False
    # No qualified model is not convergence. Keep trying until the budget cap.
    if best_score is None:
        return False
    state['monitoring'] = True
    if state['anchor_score'] is None:
        state['anchor_score'] = best_score
        return False
    anchor = state['anchor_score']
    # Strict decrease matters for zero queues and a zero improvement threshold.
    significant = (score is not None and score < anchor
                   and score <= anchor * (1 - config['min_relative_improvement']))
    if significant:
        state['anchor_score'] = best_score
        state['bad_validations'] = 0
    else:
        state['bad_validations'] += 1
    state['stopped'] = state['bad_validations'] >= config['patience']
    return state['stopped']
