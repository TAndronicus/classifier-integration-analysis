create or replace procedure
    insert_base_data()
    language sql
as
$$
insert into dynamic_ring_stats (id, file, clfs, metric, mapping, mv_acc, mv_precisionMi, mv_recallMi, mv_fScoreMi, mv_precisionM, mv_recallM, mv_fScoreM, rf_acc, rf_precisionMi,
                                rf_recallMi, rf_fScoreMi, rf_precisionM, rf_recallM, rf_fScoreM, i_acc, i_precisionMi, i_recallMi, i_fScoreMi, i_precisionM, i_recallM, i_fScoreM,
                                io_acc, io_precisionmi, io_recallmi, io_fscoremi, io_precisionm, io_recallm, io_fscorem)
select nextval('mes_seq'),
       file,
       clfs,
       metric,
       mapping,
       (mv_tp + mv_tn)::double precision / (mv_tp + mv_fn + mv_fp + mv_tn),
       0,
       0,
       0,
       mv_tp::double precision / (mv_tp + mv_fp),
       mv_tp::double precision / (mv_tp + mv_fn),
       0,
       (rf_tp + rf_tn)::double precision / (rf_tp + rf_fn + rf_fp + rf_tn),
       0,
       0,
       0,
       rf_tp::double precision / (rf_tp + rf_fp),
       rf_tp::double precision / (rf_tp + rf_fn),
       0,
       (i_tp + i_tn)::double precision / (i_tp + i_fn + i_fp + i_tn),
       0,
       0,
       0,
       i_tp::double precision / (i_tp + i_fp),
       i_tp::double precision / (i_tp + i_fn),
       0,
       0,
       0,
       0,
       0,
       0,
       0,
       0
from dynamic_ring;
$$;