class DtsRes:
    """Data object for classifier integration results

    """

    def __init__(self, mv_score, mv_mcc, i_score, i_mcc, n_clf, alpha, filename):
        self.mv_score = mv_score
        self.mv_mcc = mv_mcc
        self.i_score = i_score
        self.i_mcc = i_mcc
        self.n_clf = n_clf
        self.alpha = alpha
        self.filename = filename

    def __str__(self) -> str:
        return str(self.filename) + ", " + str(self.n_clf) + ", " + str(self.alpha) + ":::\t" + str(self.i_score) + "\t:\t" + str(self.mv_score) + \
               ",\t" + str(self.i_mcc) + "\t:\t" + str(self.mv_mcc)

