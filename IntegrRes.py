class IntegrRes:
    """Data object for classifier integration results

    """

    def __init__(self, mv_score, mv_score_std, mv_mcc, mv_mcc_std, i_score, i_score_std, i_mcc, i_mcc_std):
        self.mv_score = mv_score
        self.mv_score_std = mv_score_std
        self.mv_mcc = mv_mcc
        self.mv_mcc_std = mv_mcc_std
        self.i_score = i_score
        self.i_score_std = i_score_std
        self.i_mcc = i_mcc
        self.i_mcc_std = i_mcc_std
