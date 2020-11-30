create or replace procedure
    insert_windowed_stats(in target_clfs smallint)
    language sql
as
$$
with progressions as (
    select id,
           mv_precisionM * lag(mv_precisionMi) over clf / lag(mv_precisionM) over clf as mv_precisionMi,
           mv_recallM * lag(mv_recallMi) over clf / lag(mv_recallM) over clf          as mv_recallMi,
           rf_precisionM * lag(rf_precisionMi) over clf / lag(rf_precisionM) over clf as rf_precisionMi,
           rf_recallM * lag(rf_recallMi) over clf / lag(rf_recallM) over clf          as rf_recallMi,
           i_precisionM * lag(i_precisionMi) over clf / lag(i_precisionM) over clf    as i_precisionMi,
           i_recallM * lag(i_recallMi) over clf / lag(i_recallM) over clf             as i_recallMi
    from dynamic_ring_stats drs
        window clf as (partition by file, metric, mapping order by clfs)
)
update dynamic_ring_stats drs
set mv_precisionMi = p.mv_precisionMi,
    mv_recallMi    = p.mv_recallMi,
    rf_precisionMi = p.rf_precisionMi,
    rf_recallMi    = p.rf_recallMi,
    i_precisionMi  = p.i_precisionMi,
    i_recallMi     = p.i_recallMi
from progressions p
where p.id = drs.id
  and drs.clfs = target_clfs;
$$;