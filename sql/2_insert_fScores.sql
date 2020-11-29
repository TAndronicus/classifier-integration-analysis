create or replace procedure
    insert_fScores(in target_clfs smallint)
    language sql
as
$$
update dynamic_ring_stats
set mv_fScoreMi = 2 * mv_precisionMi * mv_recallMi / (mv_precisionMi + mv_recallMi),
    mv_fScoreM  = 2 * mv_precisionM * mv_recallM / (mv_precisionM + mv_recallM),
    rf_fScoreMi = 2 * rf_precisionMi * rf_recallMi / (rf_precisionMi + rf_recallMi),
    rf_fScoreM  = 2 * rf_precisionM * rf_recallM / (rf_precisionM + rf_recallM),
    i_fScoreMi  = 2 * i_precisionMi * i_recallMi / (i_precisionMi + i_recallMi),
    i_fScoreM   = 2 * i_precisionM * i_recallM / (i_precisionM + i_recallM)
where clfs = target_clfs
$$;