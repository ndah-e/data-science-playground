import sqlite3
import logging
import argparse
import math
import pymc3 as pm
import numpy as np
import pandas as pd
from datetime import datetime
from dateutil.rrule import rrule, MONTHLY

import warnings
warnings.warn("deprecated", DeprecationWarning)

# We set a seed so that the results are reproducible.
np.random.seed(545)


class CreateDatabase:
    """Initialize database."""
    def __init__(self, filename):
        self.db_filename = filename
        self.conn = self.create_connection()

    def create_connection(self):
        """ create a database connection to db_file
        :return: conn: Connection object or None
        """
        conn = None
        try:
            conn = sqlite3.connect(self.db_filename)
        except sqlite3.Error as error:
            logging.CRITICAL("Error, connecting to sqlite database {}".format(error))

        return conn

    def create_database(self, sql_file):
        """ create a database connection to db_file
        :param sql_file: database file
        :return: conn: Connection object or None
        """
        conn = None
        try:
            # create SQLite database
            cur = self.conn.cursor()

            # parse query file and create tables
            sql_queries = self.parse_sql(sql_file)
            cur.executescript(sql_queries)
            self.conn.commit()

        except sqlite3.Error as error:
            logging.CRITICAL("Error, cannot connect to sqlite database {}".format(error))

    def parse_sql(self, sql_file):
        """Parse sql_file to create tables
        :param sql_file: database file
        :return: query_string: Connection object or None
        """
        # read sql file into memory
        with open(sql_file, 'r', encoding='utf-8') as f:
            data = f.read().splitlines()

        query = ''  # place holder for current string
        query_concat = []  # list to collect all queries in the file
        for line in data:
            if line:
                if line.startswith('--'):  # skip comments in the query file
                    continue
                query += line.strip() + ' '
                if ';' in query:
                    query_concat.append(query.strip())
                    query = ''

        # combine queries in query_concat list into a string
        sql_string = ' '.join(query_concat)
        return sql_string

    def close_db_connection(self):
        self.conn.close()


class PopulateTables:
    """Generate data to populate database"""
    def __init__(self, conn):
        self.conn = conn

    def insert_into_tables(self, tablename, data_file):
        """Create multi query string to populate tables"""
        df = pd.read_csv(data_file)
        df.to_sql(tablename, self.conn, if_exists='append', index=False)


class SimulateData:
    def __init__(self, conn):
        self.conn = conn

    def simulate_patient_recruitment(self):
        """Simulate data for recruitment rate"""

        # Read base tables from database
        site_query = "SELECT DISTINCT * FROM sites;"
        df_sites = pd.read_sql_query(site_query, self.conn)

        trial_query = "SELECT DISTINCT * FROM trials;"
        df_trials = pd.read_sql_query(trial_query, self.conn)
        trials = df_trials.id.tolist()

        inv_query = "SELECT DISTINCT * FROM investigators;"
        df_inv = pd.read_sql_query(inv_query, self.conn)

        tsi_query = "SELECT DISTINCT * FROM investigators_to_site;"
        df_sit = pd.read_sql_query(tsi_query, self.conn)

        site_ids = []
        investigator_ids = []
        sim_averages = []
        trial_ids = []
        months = []

        for i, row in df_sit.iterrows():

            # get the number of months for the trial
            date1 = df_trials[df_trials.id == row.trial_id].begin_date.item()
            date2 = df_trials[df_trials.id == row.trial_id].end_date.item()
            num_of_months = self.date_difference(date1, date2)

            # simulate the recruitment rate for each month
            for j in range(num_of_months):

                months.append(j+1)
                sim_ave = self.monthly_average(row.average_rr, row.trial_weight)
                sim_averages.append(sim_ave)
                site_ids.append(row.site_id)
                investigator_ids.append(row.investigator_id)
                trial_ids.append(row.trial_id)

        df = pd.DataFrame({'site_id': site_ids,
                           'investigator_id': investigator_ids,
                           'trial_id': trial_ids,
                           'month': months,
                           'sim_average': sim_averages})

        tablename = 'recruitment_rate'
        print(df.head(30))
        df.to_sql(tablename, self.conn, if_exists='append', index=False)

    def date_difference(self, date1, date2):
        """Calculate number of months between two dates"""

        date1 = datetime.strptime(str(date1), "%d/%m/%Y")
        date2 = datetime.strptime(str(date2), "%d/%m/%Y")
        months = len([dt for dt in rrule(MONTHLY, dtstart=date1, until=date2)])

        return months

    def monthly_average(self, rr, trial_weight, alpha=2.0, beta=1.0/5, n_draws=1000, n_tune=2000, target_accept=0.8):
        """simulate monthly rate"""

        # define mcmc model
        model = pm.Model()
        with model:
            # trial weighting: to account for site effect in the recruitment
            alpha_s = pm.Gamma('alpha_w', alpha=alpha, beta=beta)
            beta_s = pm.Gamma('beta_w', alpha=alpha, beta=beta)
            sw = pm.Gamma('site_weight', alpha=alpha_s, beta=beta_s)

            # trial weighting: to account for trial effect in the recruitment
            alpha_t = pm.Gamma('alpha_t', alpha=alpha, beta=beta)
            beta_t = pm.Gamma('beta_t', alpha=alpha, beta=beta)
            tw = pm.Gamma('trial_weight', alpha=alpha_t, beta=beta_t, observed=trial_weight)

            # Prior distribution for lambda for investigators.
            alpha_i = pm.Gamma('alpha', alpha=alpha, beta=beta)
            beta_i = pm.Gamma('beta', alpha=alpha, beta=beta)
            lam = pm.Gamma('lambda', alpha=alpha_i, beta=beta_i)

            # Likelihood function for the data with weighted lambda.
            lam_weighted = lam * tw * sw
            y_obs = pm.Poisson('y_obs', mu=lam_weighted, observed=rr)
            trace = pm.sample(draws=n_draws, tune=n_tune, target_accept=target_accept)

        # predict enrolment rate using the simulated model
        y_pred = np.random.poisson(lam=trace['lambda'], size=len(trace['lambda']))

        return math.ceil(y_pred.astype(int).mean())


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-db', type=str, help='path to SQLite database file.', default='data/simulated_rates.db')
    parser.add_argument('-s', type=str, help='path to SQL file to populate table.', default='sql/')
    parser.add_argument('-d', type=str, help='path to csv files to populate table.', default='data/')

    args = parser.parse_args()

    ctb = CreateDatabase(args.db)

    # process sql file
    sql_file_path = args.s + '/create_tables.sql'
    ctb.create_database(sql_file_path)

    # Load tables into database
    load_data = PopulateTables(ctb.conn)
    table_name = ['sites', 'investigators', 'trials', 'investigators_to_site']
    for i in range(len(table_name)):
        file_path = args.d + "/" + table_name[i] + ".csv"
        load_data.insert_into_tables(table_name[i], file_path)

    # simulate the recruitment rate
    cts = SimulateData(ctb.conn)
    cts.simulate_patient_recruitment()
