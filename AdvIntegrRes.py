from IntegrRes import IntegrRes


class AdvIntegrRes(IntegrRes):

    def __init__(self,
                 mv_score,
                 mv_score_std,
                 mv_mcc,
                 mv_mcc_std,
                 i_score,
                 i_score_std,
                 i_mcc,
                 i_mcc_std,
                 n_class,
                 n_best,
                 i_meth,
                 bagging,
                 space_parts,
                 filename):
        super().__init__(mv_score, mv_score_std, mv_mcc, mv_mcc_std, i_score, i_score_std, i_mcc, i_mcc_std)
        self.n_class = n_class
        self.n_best = n_best
        self.i_meth = i_meth
        self.bagging = bagging
        self.space_parts = space_parts
        self.filename = filename
