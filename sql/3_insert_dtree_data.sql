create or replace procedure
    insert_dtree_data()
    language sql
as
$$
update dynamic_ring_stats drs
set ir_acc         = ddr.i_acc,
    ir_precisionmi = ddr.i_precisionmi,
    ir_recallmi    = ddr.i_fscoremi,
    ir_fscoremi    = ddr.i_fscoremi,
    ir_precisionm  = ddr.i_precisionm,
    ir_recallm     = ddr.i_recallm,
    ir_fscorem     = ddr.i_fscorem
from dynamic_dtree_raw ddr
where (drs.file, drs.clfs, drs.metric, drs.mapping) = (ddr.file, ddr.clfs, ddr.metric, ddr.mapping)
$$;