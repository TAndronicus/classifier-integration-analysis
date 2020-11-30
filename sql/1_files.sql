create table files
(
    id           bigint primary key,
    name         character varying not null unique,
    abbreviation character varying not null unique,
    size         smallint          not null,
    attributes   smallint          not null,
    classes      smallint          not null,
    major        smallint          not null,
    minor        smallint          not null,
    url          character varying not null,
    description  character varying,
    check ( classes <> 2 or (minor + major = size) ),
    check ( minor + major <= size ),
    check ( size > 1 ),
    check ( attributes > 1 ),
    check ( classes > 1 ),
    check ( minor > 0 ),
    check ( major > 0 )
);
create sequence if not exists files_seq
    minvalue 0
    no maxvalue
    cycle
    increment by 1
    start 0;

insert into files (id, name, abbreviation, size, attributes, classes, minor, major, url)
values (nextval('files_seq'), 'Indoor Channel Measurements', 'aa', 7840, 5, 19, 1, 208, 'https://archive.ics.uci.edu/ml/datasets/2.4+GHZ+Indoor+Channel+Measurements'),
       (nextval('files_seq'), 'Appendicitis', 'ap', 106, 7, 2, 21, 85, 'https://sci2s.ugr.es/keel/dataset.php?cod=183'),
       (nextval('files_seq'), 'Balance Scale', 'ba', 625, 4, 3, 49, 288, 'https://archive.ics.uci.edu/ml/datasets/Balance+Scale'),
       (nextval('files_seq'), 'QSAR Biodegegradation', 'bi', 1055, 41, 2, 356, 699, 'https://archive.ics.uci.edu/ml/datasets/QSAR+biodegradation'),
       (nextval('files_seq'), 'Liver Disorders (BUPA)', 'bu', 345, 7, 2, 145, 200, 'https://archive.ics.uci.edu/ml/datasets/liver+disorders'),
       (nextval('files_seq'), 'Cryotherapy', 'c', 90, 7, 2, 42, 48, 'https://archive.ics.uci.edu/ml/datasets/Cryotherapy+Dataset+'),
       (nextval('files_seq'), 'Data banknote authentication', 'd', 1372, 5, 2, 610, 762, 'https://archive.ics.uci.edu/ml/datasets/banknote+authentication'),
       (nextval('files_seq'), 'Ecoli', 'e', 336, 8, 8, 2, 143, 'https://archive.ics.uci.edu/ml/datasets/ecoli'),
       (nextval('files_seq'), 'Habermans Survival', 'h', 306, 3, 2, 81, 225, 'https://archive.ics.uci.edu/ml/datasets/Haberman%27s+Survival'),
       (nextval('files_seq'), 'Ionosphere', 'io', 351, 34, 2, 126, 225, 'https://archive.ics.uci.edu/ml/datasets/ionosphere'),
       (nextval('files_seq'), 'Iris', 'ir', 150, 4, 3, 50, 50, 'https://archive.ics.uci.edu/ml/datasets/Iris'),
       (nextval('files_seq'), 'Magic', 'ma', 19020, 11, 2, 6688, 12332, 'https://archive.ics.uci.edu/ml/datasets/magic+gamma+telescope'),
       (nextval('files_seq'), 'Ultrasonic flowmeter diagnostics', 'me', 540, 143, 2, 220, 320, 'https://archive.ics.uci.edu/ml/datasets/Ultrasonic+flowmeter+diagnostics'),
       (nextval('files_seq'), 'Phoneme', 'ph', 5404, 5, 2, 1586, 3818, 'https://sci2s.ugr.es/keel/dataset.php?cod=105'),
       (nextval('files_seq'), 'Pima', 'pi', 768, 9, 2, 268, 500, 'https://www.kaggle.com/uciml/pima-indians-diabetes-database'),
       (nextval('files_seq'), 'Climate model simulation crashes', 'po', 540, 18, 2, 46, 494, 'https://archive.ics.uci.edu/ml/datasets/climate+model+simulation+crashes'),
       (nextval('files_seq'), 'Ring', 'r', 7400, 20, 2, 3664, 3736, 'https://sci2s.ugr.es/keel/dataset.php?cod=106'),
       (nextval('files_seq'), 'Spambase', 'sb', 4601, 57, 2, 1816, 2785, 'https://archive.ics.uci.edu/ml/datasets/spambase'),
       (nextval('files_seq'), 'Seismic-bumps', 'se', 2584, 19, 2, 170, 2414, 'https://archive.ics.uci.edu/ml/datasets/seismic-bumps'),
       (nextval('files_seq'), 'Texture', 'te', 5500, 40, 11, 500, 500, 'https://sci2s.ugr.es/keel/dataset.php?cod=72'),
       (nextval('files_seq'), 'Thyroid', 'th', 7200, 21, 3, 166, 6666, 'https://archive.ics.uci.edu/ml/datasets/Thyroid+Disease'),
       (nextval('files_seq'), 'Titanic', 'ti', 2201, 3, 2, 711, 1490, 'https://sci2s.ugr.es/keel/dataset.php?cod=189'),
       (nextval('files_seq'), 'Twonorm', 'tw', 7400, 20, 2, 3697, 3703, 'https://sci2s.ugr.es/keel/dataset.php?cod=110'),
       (nextval('files_seq'), 'Breast Cancer (Diagnostic)', 'wd', 569, 32, 2, 212, 357, 'https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic)'),
       (nextval('files_seq'), 'Breast Cancer (Original)', 'wi', 699, 10, 2, 256, 443, 'https://archive.ics.uci.edu/ml/datasets/breast+cancer+wisconsin+(original)'),
       (nextval('files_seq'), 'Wine quality – red', 'wr', 1599, 11, 6, 10, 681, 'https://sci2s.ugr.es/keel/dataset.php?cod=210'),
       (nextval('files_seq'), 'Wine quality – white', 'ww', 4898, 11, 7, 5, 2198, 'https://sci2s.ugr.es/keel/dataset.php?cod=209'),
       (nextval('files_seq'), 'Yeast', 'y', 1484, 8, 10, 5, 463, 'https://archive.ics.uci.edu/ml/datasets/Yeast');
