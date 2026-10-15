-- Step 6 – Prediction log. Runs once, when the PostgreSQL volume is created.
-- Every POST /predict adds a row; the real answer (actual) arrives later through POST /feedback.
CREATE TABLE predictions (
    prediction_id  uuid PRIMARY KEY,
    created_at     timestamptz      NOT NULL DEFAULT now(),
    passenger_id   bigint           NOT NULL,
    features       jsonb            NOT NULL,  -- the 22 model inputs, used for drift checks (step 7)
    probability    double precision NOT NULL,
    predicted      smallint         NOT NULL,  -- 1 = satisfied
    actual         smallint,                   -- NULL until feedback arrives
    labelled_at    timestamptz,
    model_version  text             NOT NULL,
    replica        text             NOT NULL,  -- which API copy answered
    latency_ms     double precision NOT NULL   -- time spent inside the API for this prediction
);

CREATE INDEX predictions_created_at ON predictions (created_at);
