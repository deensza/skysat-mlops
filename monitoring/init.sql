-- Prediction log of the complaint box. Runs once, when the PostgreSQL volume is created.
-- llm/serve.py adds one row per POST /score (and creates this table itself if the volume is older).
-- Only numbers and fixed labels: never the review, the title or the airline (the dataset is not ours to store).
-- true_score / true_recommended are filled when the caller (the replay) knows the real answer.
CREATE TABLE IF NOT EXISTS llm_scores (
    id                bigserial PRIMARY KEY,
    created_at        timestamptz NOT NULL DEFAULT now(),
    model_version     text,
    latency_ms        integer,
    score             integer,   -- NULL: the model's answer was malformed
    recommended       text,
    complaint         text,
    true_score        integer,
    true_recommended  text
);

CREATE INDEX IF NOT EXISTS llm_scores_created_at ON llm_scores (created_at);
