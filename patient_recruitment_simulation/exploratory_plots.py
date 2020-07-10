import argparse
import logging
import sqlite3
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def create_connection(db_filename):
    """ create a database connection to db_file
    :return: conn: Connection object or None
    """
    conn = None
    try:
        conn = sqlite3.connect(db_filename)
    except sqlite3.Error as error:
        logging.CRITICAL("Error, connecting to sqlite database {}".format(error))

    return conn


def close_connection(conn):
    conn.close()


class PlotRates:
    """Plot recruitment rates for simulated data."""
    def __init__(self, conn, plot_dir):
        self.conn = conn
        self.plot_dir = plot_dir
        self.df = self.read_patient_table()
        self.site_names = self.get_name('site_id', 'center')
        self.inv_names = self.get_name('investigator_id', 'names')
        self.trial_names = self.get_name('trial_id', 'trial')

    def read_patient_table(self):
        """Extract dat into pandas dataframe"""

        query = "SELECT rr.site_id, rr.investigator_id, rr.trial_id, rr.month, rr.sim_average, s.center, ins.names, t.trial " \
                "FROM recruitment_rate rr " \
                "LEFT JOIN investigators ins ON rr.investigator_id = ins.id " \
                "LEFT JOIN sites s on rr.site_id = s.id left join trials t on rr.trial_id = t.id;"

        df = pd.read_sql_query(query, self.conn)

        return df

    def get_name(self, id_field, value_field):
        ids = sorted(self.df[id_field].unique())
        values = []
        for i in ids:
            sn = self.df[self.df[id_field] == i][value_field].tolist()
            values.append(sn[0])
        return values

    def get_site_names(self):
        """Read site table from sqlite"""
        df_site = pd.read_sql_query("SELECT id,center FROM sites;", self.conn)
        return df_site

    def get_investigators_names(self):
        """Read investigator table from sqlite"""
        df_inv = pd.read_sql_query("SELECT id,names FROM investigators;", self.conn)
        return df_inv

    def investigator_total(self):
        df_grp = self.df.groupby(['investigator_id'])['sim_average'].agg('sum').reset_index()

        # Draw a nested barplot to show survival for class and sex
        plt.figure(figsize=(16, 10))
        g = sns.catplot(x="investigator_id", y="sim_average", data=df_grp, height=6, kind="bar", palette="muted")
        g.despine(left=True)
        plt.title('Frequency of recruitment by Investigator', fontsize=16)
        g.set_ylabels("# Recruited", fontsize=16)
        g.set_xlabels("Investigators", fontsize=16)
        g.set_xticklabels(fontsize=12)
        g.set_xticklabels(self.inv_names, fontsize=11)
        plt.savefig(self.plot_dir + '/investigator_total.png')

    def investigator_recruitment_by_site(self):
        df_grp = self.df.groupby(['investigator_id', 'site_id'])['sim_average'].agg('sum').reset_index()

        # Draw a nested barplot to show survival for class and sex
        plt.figure(figsize=(16, 10))
        g = sns.catplot(x="investigator_id", y="sim_average", hue='site_id', data=df_grp, height=6, kind="bar",
                        palette="muted")

        g._legend.set_title('Center')
        for t, l in zip(g._legend.texts, self.site_names):
            t.set_text(l)

        g.despine(left=True)
        plt.title('Frequency of recruitment by site', fontsize=16)
        g.set_ylabels("# Recruited", fontsize=16)
        g.set_xlabels("Investigators", fontsize=16)
        g.set_xticklabels(fontsize=12)
        g.set_xticklabels(self.inv_names, fontsize=11)
        plt.savefig(self.plot_dir + '/investigator_per_site.png.png')

    def investigator_recruitment_per_month(self):
        df_grp = self.df.groupby(['investigator_id', 'month'])['sim_average'].agg('sum').reset_index()

        # Draw a nested barplot to show survival for class and sex
        plt.figure(figsize=(16, 10))
        g = sns.catplot(x="investigator_id", y="sim_average", hue='month', data=df_grp, height=6, kind="bar",
                        palette="muted")
        g.despine(left=True)
        plt.title('Recruites per Investigator', fontsize=16)
        g.set_ylabels("# Recruited", fontsize=16)
        g.set_xlabels("Investigators", fontsize=16)
        g.set_xticklabels(fontsize=12)
        g.set_xticklabels(self.inv_names, fontsize=11)
        plt.savefig(self.plot_dir + '/investigator_per_month.png')

    def investigator_by_trial(self):
        df_grp = self.df.groupby(['investigator_id', 'trial_id'])['sim_average'].agg('sum').reset_index()

        # Draw a nested barplot to show survival for class and sex
        plt.figure(figsize=(16, 10))
        g = sns.catplot(x="investigator_id", y="sim_average", hue='trial_id', data=df_grp, height=6, kind="bar",
                        palette="muted")
        g._legend.set_title('Trial')
        for t, l in zip(g._legend.texts, self.trial_names):
            t.set_text(l)

        g.despine(left=True)
        plt.title('frequency of recruitment by trials', fontsize=16)
        g.set_ylabels("# Recruited", fontsize=16)
        g.set_xlabels("Investigators", fontsize=16)
        g.set_xticklabels(fontsize=12)
        g.set_xticklabels(self.inv_names, fontsize=11)
        plt.savefig(self.plot_dir + '/investigator_per_trial.png')

    def site_recruitment(self):
        df_grp = self.df.groupby(['site_id'])['sim_average'].agg('sum').reset_index()

        # Draw a nested barplot to show survival for class and sex
        plt.figure(figsize=(16, 10))
        g = sns.catplot(x="site_id", y="sim_average", data=df_grp, height=6, kind="bar", palette="muted")
        g.despine(left=True)
        plt.title('Frequency of recruitment by Site', fontsize=16)
        g.set_ylabels("# Recruited", fontsize=16)
        g.set_xlabels("Site", fontsize=16)
        g.set_xticklabels(fontsize=12)
        g.set_xticklabels(self.site_names, fontsize=11)
        plt.savefig(self.plot_dir + '/recruit_per_site.png')

    def site_per_month(self):
        df_grp = self.df.groupby(['site_id', 'month'])['sim_average'].agg('sum').reset_index()

        # Draw a nested barplot to show survival for class and sex
        plt.figure(figsize=(16, 10))
        g = sns.catplot(x="site_id", y="sim_average", hue='month', data=df_grp, height=6, kind="bar", palette="muted")

        g.despine(left=True)
        plt.title('Frequency of recruitment per month grouped by Site', fontsize=16)
        g.set_ylabels("# Recruited", fontsize=16)
        g.set_xlabels("Site", fontsize=16)
        g.set_xticklabels(fontsize=12)
        g.set_xticklabels(self.site_names, fontsize=11)
        plt.savefig(self.plot_dir + '/site_per_month.png')

    def site_by_trial(self):
        df_grp = self.df.groupby(['site_id', 'trial_id'])['sim_average'].agg('sum').reset_index()

        # Draw a nested barplot to show survival for class and sex
        plt.figure(figsize=(16, 10))
        g = sns.catplot(x="site_id", y="sim_average", hue='trial_id', data=df_grp, height=6, kind="bar",
                        palette="muted")

        g._legend.set_title('Trial')
        for t, l in zip(g._legend.texts, self.trial_names):
            t.set_text(l)

        g.despine(left=True)
        plt.title('Recruites per Investigator', fontsize=16)
        g.set_ylabels("# Recruited", fontsize=16)
        g.set_xlabels("Site", fontsize=16)
        g.set_xticklabels(fontsize=12)
        g.set_xticklabels(self.site_names, fontsize=11)
        plt.savefig(self.plot_dir + '/site_per_trail.png')


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-db', type=str, help='path to SQLite database file.', default='data/simulated_rates_tmp.db')
    parser.add_argument('-p', type=str, help='folder to save plots.', default='plots/')

    args = parser.parse_args()
    conn = create_connection(args.db)

    plots = PlotRates(conn, args.p)
    plots.investigator_total()
    plots.investigator_recruitment_by_site()
    plots.investigator_recruitment_per_month()
    plots.investigator_by_trial()
    plots.site_recruitment()
    plots.site_per_month()
    plots.site_by_trial()
    plots.site_by_trial()

    close_connection(conn)

    logging.info('Plots created successfully')
