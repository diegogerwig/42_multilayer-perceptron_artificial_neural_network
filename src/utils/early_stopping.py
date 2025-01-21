def check_early_stopping(val_loss, state=None, patience=5, min_delta=0.001):
    """
    Check if training should be stopped early based on validation loss.
    """
    # Initialize state if None
    if state is None:
        state = {
            'best_loss': None,
            'counter': 0,
            'early_stop': False
        }
    
    # Initialize best_loss if not set
    if state['best_loss'] is None:
        state['best_loss'] = val_loss
        return False, state
    
    # Check if current loss is better than best_loss by min_delta
    if val_loss > state['best_loss'] - min_delta:
        state['counter'] += 1
        if state['counter'] >= patience:
            state['early_stop'] = True
    else:
        state['best_loss'] = val_loss
        state['counter'] = 0
    
    return state['early_stop'], state