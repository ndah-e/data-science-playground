-- noinspection SqlNoDataSourceInspectionForFile

-- Investigator (doctors) table
CREATE TABLE IF NOT EXISTS investigators (
    id integer PRIMARY KEY AUTOINCREMENT,
    title varchar(5),
    names varchar(20) NOT NULL,
    address varchar(20),
    average_rr integer NOT NULL
);

-- Site (hospitals) table
CREATE TABLE IF NOT EXISTS sites (
    id integer PRIMARY KEY AUTOINCREMENT,
    center varchar(20) NOT NULL,
    address varchar(20),
    site_rr real
);

-- Trail table
CREATE TABLE IF NOT EXISTS trials (
    id integer PRIMARY KEY AUTOINCREMENT,
    trial varchar(20) NOT NULL,
    begin_date date,
    end_date date,
    trial_weight real
);

-- recruitment_rate
CREATE TABLE IF NOT EXISTS investigators_to_site (
    site_id integer NOT NULL ,
    investigator_id integer NOT NULL ,
    trial_id integer NOT NULL,
    average_rr integer,
    trial_weight real,
    FOREIGN KEY (site_id) REFERENCES sites (id),
    FOREIGN KEY (investigator_id) REFERENCES investigators (id),
    FOREIGN KEY (trial_id) REFERENCES trials (id)
);

-- recruitment_rate
CREATE TABLE IF NOT EXISTS recruitment_rate (
    site_id integer NOT NULL ,
    investigator_id integer NOT NULL ,
    trial_id integer NOT NULL,
    month integer NOT NULL,
    sim_average integer NOT NULL,
    FOREIGN KEY (site_id) REFERENCES sites (id),
    FOREIGN KEY (investigator_id) REFERENCES investigators (id),
    FOREIGN KEY (trial_id) REFERENCES trials (id)
);