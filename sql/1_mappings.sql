create table mappings
(
    id           bigint primary key,
    name         character varying not null,
    abbreviation character varying not null
);
insert into mappings (id, name, abbreviation)
values (1, 'Half by distance', 'hbd');