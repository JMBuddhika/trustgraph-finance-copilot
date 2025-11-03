-- Fallback SQL seed (run with DuckDB CLI or any DuckDB client)
-- Generated: 2025-11-03T15:51:35.590750Z

CREATE OR REPLACE TABLE aapl_10_k_2024_tbl0 (
  Year TEXT, Segment TEXT, Revenue_USD_M INTEGER, GrossMargin_Pct DOUBLE
);
INSERT INTO aapl_10_k_2024_tbl0 VALUES
 ('2023','iPhone',205000,45.0),
 ('2023','Services',85000,71.0),
 ('2024','iPhone',212000,46.2),
 ('2024','Services',93000,72.1);

CREATE OR REPLACE TABLE msft_10_k_2024_tbl0 (
  Year TEXT, Segment TEXT, Revenue_USD_M INTEGER, GrossMargin_Pct DOUBLE
);
INSERT INTO msft_10_k_2024_tbl0 VALUES
 ('2023','Intelligent Cloud',91500,69.5),
 ('2023','More Personal Computing',54500,37.0),
 ('2024','Intelligent Cloud',105000,71.2),
 ('2024','More Personal Computing',56000,38.4);

CREATE OR REPLACE TABLE nvda_10_q_2024_tbl0 (
  Quarter TEXT, Segment TEXT, Revenue_USD_M INTEGER, GrossMargin_Pct DOUBLE
);
INSERT INTO nvda_10_q_2024_tbl0 VALUES
 ('2023Q3','Data Center',11000,71.0),
 ('2024Q2','Data Center',13800,72.5),
 ('2024Q3','Data Center',16000,73.3),
 ('2024Q3','Gaming',2800,52.0);
