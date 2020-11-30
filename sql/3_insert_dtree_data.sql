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
    io_fscorem     = ddr.i_fscorem
from dynamic_dtree_raw ddr
where (drs.file, drs.clfs, drs.metric, drs.mapping) = (ddr.file, ddr.clfs, ddr.metric, ddr.mapping)
$$;