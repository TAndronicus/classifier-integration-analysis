class DynamicDtreeRes:
    """Data object for classifier integration results

    """

    def __init__(self,
                 mv_acc, mv_mcc, mv_f1, mv_aur,
                 rf_acc, rf_mcc, rf_f1, rf_aur,
                 i_acc, i_mcc, i_f1, i_aur,
                 n_clf, metric, mapping, filename):
        self.mv_acc = mv_acc
        self.mv_mcc = mv_mcc
        self.mv_f1 = mv_f1
        self.mv_aur = mv_aur

        self.rf_acc = rf_acc
        self.rf_mcc = rf_mcc
        self.rf_f1 = rf_f1
        self.rf_aur = rf_aur

        self.i_acc = i_acc
        self.i_mcc = i_mcc
        self.i_f1 = i_f1
        self.i_aur = i_aur

        self.n_clf = n_clf
        self.metric = metric
        self.mapping = mapping
        self.filename = filename

    def __str__(self) -> str:
        return str(self.filename) + ", " + str(self.n_clf) + ", " + self.metric + ", " + self.mapping

