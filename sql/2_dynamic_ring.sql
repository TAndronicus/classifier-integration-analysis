create table dynamic_ring
(
    id      bigint primary key,
    file    bigint           not null references files,
    clfs    smallint         not null,
    metric  smallint         not null references metrics,
    mapping smallint         not null references mappings,
    mv_tp   smallint         not null,
    mv_fp   double precision not null,
    mv_fn   double precision not null,
    mv_tn   double precision not null,
    rf_tp   smallint         not null,
    rf_fp   double precision not null,
    rf_fn   double precision not null,
    rf_tn   double precision not null,
    i_tp    smallint         not null,
    i_fp    double precision not null,
    i_fn    double precision not null,
    i_tn    double precision not null
);