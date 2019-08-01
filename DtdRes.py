class DtdRes:
    """Data object for classifier integration results

    """

    def __init__(self, mv_acc, mv_mcc, rf_acc, rf_mcc, i_acc, i_mcc, n_clf, n_fea, n_div, filename):
        self.mv_acc = mv_acc
        self.mv_mcc = mv_mcc
        self.rf_acc = rf_acc
        self.rf_mcc = rf_mcc
        self.i_acc = i_acc
        self.i_mcc = i_mcc
        self.n_clf = n_clf
        self.n_fea = n_fea
        self.n_div = n_div
        self.filename = filename

    def __str__(self) -> str:
        return str(self.filename) + ", " + str(self.n_clf) + ", " + str(self.n_fea) + ", " + str(self.n_div) + ":::\t" + str(self.i_acc) + "\t:\t" + str(self.mv_acc) + \
               ",\t" + str(self.i_mcc) + "\t:\t" + str(self.mv_mcc)

