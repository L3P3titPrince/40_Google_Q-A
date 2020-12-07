class AnalysisAndPlot(object):
    def __init__(self):
        pass

    def get_scores(y_true, y_pred) -> Dict[str, float]:
        # y_true, y_pred: np.ndarray with shape (sample_size, num_targets)
        assert y_true.shape == y_pred.shape
        assert y_true.shape[1] == num_targets
        scores = {}
        for target_name, i in zip(target_names, range(y_true.shape[1])):
            scores[target_name] = scipy.stats.spearmanr(y_true[:, i], y_pred[:, i])[0]
        return scores
