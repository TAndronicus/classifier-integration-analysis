drop table if exists dynamic_ring_raw;
drop table if exists dynamic_ring;
drop table if exists dynamic_ring_stats;
drop table if exists dynamic_dtree_raw;

drop table if exists files;
drop table if exists metrics;
drop table if exists mappings;

drop sequence files_seq;
drop sequence mes_seq;

drop procedure if exists insert_base_data();
drop procedure if exists insert_non_windowed_stats();
drop procedure if exists insert_windowed_stats(target_clfs smallint);
drop procedure if exists insert_fscores(target_clfs smallint);
drop procedure if exists insert_dtree_data();
