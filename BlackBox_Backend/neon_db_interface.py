# neon_db_interface.py
"""
A simple Neon database interface for caching and retrieving credit reports.
Replace connection details and queries as needed for your Neon/Postgres setup.
"""
import psycopg2
import os
from typing import Optional

class NeonDBInterface:
    def __init__(self, dsn: Optional[str] = None):
        self.dsn = dsn or os.getenv('NEON_DSN')
        self.conn = psycopg2.connect(self.dsn)
        self._ensure_table()

    def _ensure_table(self):
        with self.conn.cursor() as cur:
            cur.execute('''
                CREATE TABLE IF NOT EXISTS credit_reports (
                    ticker VARCHAR(32),
                    report JSONB,
                    timestamp TIMESTAMP,
                    PRIMARY KEY (ticker, timestamp)
                )
            ''')
            self.conn.commit()

    def get_latest_record(self, ticker: str):
        with self.conn.cursor() as cur:
            cur.execute('''
                SELECT report, timestamp FROM credit_reports
                WHERE ticker = %s
                ORDER BY timestamp DESC
                LIMIT 1
            ''', (ticker,))
            row = cur.fetchone()
            if row:
                return {'report': row[0], 'timestamp': row[1].isoformat()}
            return None

    def save_report(self, ticker: str, report_json: str, timestamp: str):
        with self.conn.cursor() as cur:
            cur.execute('''
                INSERT INTO credit_reports (ticker, report, timestamp)
                VALUES (%s, %s::jsonb, %s)
                ON CONFLICT (ticker, timestamp) DO UPDATE SET report = EXCLUDED.report
            ''', (ticker, report_json, timestamp))
            self.conn.commit()

    def update_latest_timestamp(self, ticker: str):
        """Update the timestamp of the latest record to current time (for testing cache)"""
        import datetime
        with self.conn.cursor() as cur:
            # Get the latest record
            cur.execute('''
                SELECT report FROM credit_reports
                WHERE ticker = %s
                ORDER BY timestamp DESC
                LIMIT 1
            ''', (ticker,))
            row = cur.fetchone()
            if row:
                # Delete old record and insert with new timestamp
                cur.execute('DELETE FROM credit_reports WHERE ticker = %s', (ticker,))
                cur.execute('''
                    INSERT INTO credit_reports (ticker, report, timestamp)
                    VALUES (%s, %s, %s)
                ''', (ticker, row[0], datetime.datetime.now()))
                self.conn.commit()
                return True
            return False

    def close(self):
        self.conn.close()
