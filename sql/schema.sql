CREATE TABLE dbo.product_type (
  type_id INT IDENTITY(1,1) PRIMARY KEY,
  type_code NVARCHAR(50) UNIQUE NOT NULL,
  description NVARCHAR(200)
);

CREATE TABLE dbo.thickness_spec (
  spec_id INT IDENTITY(1,1) PRIMARY KEY,
  type_id INT NOT NULL FOREIGN KEY REFERENCES dbo.product_type(type_id),
  point CHAR(1) NOT NULL CHECK (point IN ('A','B','C','D','E','F','G')),
  lower_um DECIMAL(9,3) NOT NULL,
  upper_um DECIMAL(9,3) NOT NULL,
  default_target_um DECIMAL(9,3) NOT NULL,
  weight DECIMAL(6,3) NOT NULL DEFAULT(1.0),
  CONSTRAINT uq_spec UNIQUE (type_id, point)
);

-- Data-driven feature ranges (used by optimizer/UI)
CREATE TABLE dbo.feature_bounds (
  bound_id INT IDENTITY(1,1) PRIMARY KEY,
  type_id INT NOT NULL FOREIGN KEY REFERENCES dbo.product_type(type_id),
  feature NVARCHAR(120) NOT NULL,
  q05 FLOAT NOT NULL,
  q50 FLOAT NOT NULL,
  q95 FLOAT NOT NULL,
  min_val FLOAT NULL,
  max_val FLOAT NULL,
  is_tunable BIT NOT NULL DEFAULT(1),
  CONSTRAINT uq_bounds UNIQUE (type_id, feature)
);

CREATE TABLE dbo.model_registry (
  model_id INT IDENTITY(1,1) PRIMARY KEY,
  model_name NVARCHAR(100) NOT NULL,
  version NVARCHAR(50) NOT NULL,
  artifact_uri NVARCHAR(400) NOT NULL,
  created_at DATETIME2 NOT NULL DEFAULT SYSUTCDATETIME(),
  notes NVARCHAR(MAX) NULL
);

CREATE TABLE dbo.prediction_log (
  log_id BIGINT IDENTITY(1,1) PRIMARY KEY,
  ts DATETIME2 NOT NULL DEFAULT SYSUTCDATETIME(),
  type_code NVARCHAR(50),
  inputs_json NVARCHAR(MAX),
  predictions_json NVARCHAR(MAX),
  model_version NVARCHAR(50),
  user_id NVARCHAR(100)
);
CREATE INDEX ix_pred_ts ON dbo.prediction_log(ts);

CREATE TABLE dbo.recommendation_log (
  log_id BIGINT IDENTITY(1,1) PRIMARY KEY,
  ts DATETIME2 NOT NULL DEFAULT SYSUTCDATETIME(),
  type_code NVARCHAR(50),
  targets_json NVARCHAR(MAX),
  recommendation_json NVARCHAR(MAX),
  objective_score FLOAT,
  trials INT,
  model_version NVARCHAR(50),
  user_id NVARCHAR(100)
);
CREATE INDEX ix_rec_ts ON dbo.recommendation_log(ts);