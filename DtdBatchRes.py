class DtdBatchRes:
    """Data object for classifier integration results

    """

    def __init__(self, mv_acc, mv_mcc, rf_acc, rf_mcc, wmv_vol_acc, wmv_vol_mcc, i_vol_acc, i_vol_mcc, wmv_inv_acc, wmv_inv_mcc, i_inv_acc, i_inv_mcc, n_clf, n_fea, n_div, filename):
        self.mv_acc = mv_acc
        self.mv_mcc = mv_mcc
        self.rf_acc = rf_acc
        self.rf_mcc = rf_mcc
        self.wmv_vol_acc = wmv_vol_acc
        self.wmv_vol_mcc = wmv_vol_mcc
        self.i_vol_acc = i_vol_acc
        self.i_vol_mcc = i_vol_mcc
        self.wmv_inv_acc = wmv_inv_acc
        self.wmv_inv_mcc = wmv_inv_mcc
        self.i_inv_acc = i_inv_acc
        self.i_inv_mcc = i_inv_mcc
        self.n_clf = n_clf
        self.n_fea = n_fea
        self.n_div = n_div # array with divisions
        self.filename = filename

    def __str__(self) -> str:
        return 'Results file: ' + str(self.filename) + ", n: " + str(self.n_clf) + ", dim: " + str(self.n_fea) + ", div: " + str('_'.join(self.n_div))

