create table metrics
(
    id           bigint primary key,
    name         character varying not null,
    abbreviation character varying not null
);
insert into metrics (id, name, abbreviation)
values (1, 'Euclidean', 'e');