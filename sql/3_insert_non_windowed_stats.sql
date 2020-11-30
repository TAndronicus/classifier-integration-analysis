create or replace procedure
    insert_non_windowed_stats()
    language sql
as
$$
update dynamic_ring_stats drs
set mv_precisionMi = drr.mv_precisionMi,
    mv_recallMi    = drr.mv_recallMi,
    mv_fScoreM     = drr.mv_fScoreM,
    mv_fScoreMi    = drr.mv_fScoreMi,
    rf_precisionMi = drr.rf_precisionMi,
    rf_recallMi    = drr.rf_recallMi,
    rf_fScoreM     = drr.rf_fScoreM,
    rf_fScoreMi    = drr.rf_fScoreMi,
    i_precisionMi  = drr.i_precisionMi,
    i_recallMi     = drr.i_recallMi,
    i_fScoreM      = drr.i_fScoreM,
    i_fScoreMi     = drr.i_fScoreMi
from dynamic_ring_raw drr
where (drr.file, drr.clfs, drr.metric, drr.mapping) = (drs.file, drs.clfs, drs.metric, drs.mapping);
$$;