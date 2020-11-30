create or replace procedure
    insert_dtree_data()
    language sql
as
$$
update dynamic_ring_stats drs
set
$$