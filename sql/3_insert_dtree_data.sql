-- Warning: Some data might be populated before running this procedure by 3_insert_base_data.
-- Especially regarding mv and rf.
-- Both ring and dtree should operate on the same data (clfs, metrics, mappings)
create or replace procedure
    insert_dtree_data()
    language sql
as
$$
update dynamic_ring_stats drs
set io_acc         = ddr.i_acc,
    io_precisionmi = ddr.i_precisionmi,
    io_recallmi    = ddr.i_fscoremi,
    io_fscoremi    = ddr.i_fscoremi,
    io_precisionm  = ddr.i_precisionm,
    io_recallm     = ddr.i_recallm,
    io_fscorem     = ddr.i_fscorem,
    mv_acc         = ddr.mv_acc,
    mv_precisionmi = ddr.mv_precisionmi,
    mv_recallmi    = ddr.mv_fscoremi,
    mv_fscoremi    = ddr.mv_fscoremi,
    mv_precisionm  = ddr.mv_precisionm,
    mv_recallm     = ddr.mv_recallm,
    mv_fscorem     = ddr.mv_fscorem,
    rf_acc         = ddr.rf_acc,
    rf_precisionmi = ddr.rf_precisionmi,
    rf_recallmi    = ddr.rf_fscoremi,
    rf_fscoremi    = ddr.rf_fscoremi,
    rf_precisionm  = ddr.rf_precisionm,
    rf_recallm     = ddr.rf_recallm,
    rf_fscorem     = ddr.rf_fscorem
from dynamic_dtree_raw ddr
where (drs.file, drs.clfs, drs.metric, drs.mapping) = (ddr.file, ddr.clfs, ddr.metric, ddr.mapping)
$$;