class DtsRes:
    """Data object for classifier integration results

    """

    def __init__(self, mv_acc, mv_mcc, i_acc, i_mcc, n_clf, alpha, filename):
        self.mv_acc = mv_acc
        self.mv_mcc = mv_mcc
        self.i_acc = i_acc
        self.i_mcc = i_mcc
        self.n_clf = n_clf
        self.alpha = alpha
        self.filename = filename

    def __str__(self) -> str:
        return str(self.filename) + ", " + str(self.n_clf) + ", " + self.alpha + ":::\t" + str(self.i_acc) + "\t:\t" + str(self.mv_acc) + \
               ",\t" + str(self.i_mcc) + "\t:\t" + str(self.mv_mcc)

