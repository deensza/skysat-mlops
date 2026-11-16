"""The review queue between the intake and the trainer, a table in the server's PostgreSQL.

    passengers / simulator -> POST /reviews (llm/intake.py) -> queue.reviews -> claim / done -> trainer (llm/trainer.py)

Each review moves pending -> claimed (by one training batch) -> trained. A batch that fails is put back to pending;
a claim older than STALE_HOURS (a trainer pod that died) is put back by the intake. Claims take the most recently
received reviews first (the model should learn what passengers say now) and use FOR UPDATE SKIP LOCKED, so two
trainers never get the same review.

The table is in its own schema `queue`: other database users (Grafana's read-only `grafana_reader`) get no USAGE on
it, so the dashboard can never read review text.
"""

STALE_HOURS = 3

SCHEMA = """
CREATE SCHEMA IF NOT EXISTS queue;
REVOKE ALL ON SCHEMA queue FROM PUBLIC;
CREATE TABLE IF NOT EXISTS queue.reviews (
    id            text PRIMARY KEY,           -- the sender's id (the simulator: kaggle128k:<row>), or a random one
    source        text NOT NULL,              -- simulator | passenger
    airline       text NOT NULL,
    cabin         text,
    travel_type   text,
    title         text,
    review        text NOT NULL,
    flown         date,
    score         integer NOT NULL CHECK (score BETWEEN 1 AND 10),
    recommended   text NOT NULL CHECK (recommended IN ('yes', 'no')),
    complaints    text[] NOT NULL,            -- the lowest sub-rating(s) if <= 2, else {none}
    received_at   timestamptz NOT NULL DEFAULT clock_timestamp(),
    status        text NOT NULL DEFAULT 'pending' CHECK (status IN ('pending', 'claimed', 'trained')),
    batch_id      text,
    claimed_at    timestamptz,
    trained_at    timestamptz,
    scored_at     timestamptz                 -- sent to POST /score as live traffic (with its true labels)
);
CREATE INDEX IF NOT EXISTS reviews_status ON queue.reviews (status, received_at);
"""

COLUMNS = ["id", "source", "airline", "cabin", "travel_type", "title", "review", "flown", "score", "recommended",
           "complaints", "scored_at"]  # scored_at set on insert: never sent as live traffic (training-only reviews)
INSERT = (f"INSERT INTO queue.reviews ({', '.join(COLUMNS)}) VALUES ({', '.join(f'%({c})s' for c in COLUMNS)})"
          " ON CONFLICT (id) DO NOTHING")
OUT = "r.id, r.airline, r.cabin, r.travel_type, r.title, r.review, r.flown, r.score, r.recommended, r.complaints"

CLAIM = f"""
WITH picked AS (
    SELECT id FROM queue.reviews WHERE status = 'pending'
    ORDER BY received_at DESC, id LIMIT %(n)s FOR UPDATE SKIP LOCKED
)
UPDATE queue.reviews r SET status = 'claimed', batch_id = %(batch_id)s, claimed_at = now()
FROM picked WHERE r.id = picked.id
RETURNING {OUT}
"""
# Already-trained reviews mixed into a batch, so a new adapter does not forget what the last one learned
REPLAY = f"SELECT {OUT} FROM queue.reviews r WHERE status = 'trained' ORDER BY random() LIMIT %(n)s"
DONE = "UPDATE queue.reviews SET status = 'trained', trained_at = now() WHERE batch_id = %(batch_id)s AND status = 'claimed'"
RELEASE = """
UPDATE queue.reviews SET status = 'pending', batch_id = NULL, claimed_at = NULL
WHERE batch_id = %(batch_id)s AND status = 'claimed'
"""
RELEASE_STALE = f"""
UPDATE queue.reviews SET status = 'pending', batch_id = NULL, claimed_at = NULL
WHERE status = 'claimed' AND claimed_at < now() - interval '{STALE_HOURS} hours'
"""
STATS = "SELECT status, count(*), max(received_at) FROM queue.reviews GROUP BY status"
# Live traffic: the oldest received reviews nobody has scored yet (first in, first scored) and nobody trained on
TO_SCORE = f"""
SELECT {OUT} FROM queue.reviews r WHERE scored_at IS NULL AND status = 'pending' ORDER BY received_at, id LIMIT %(n)s
"""
MARK_SCORED = "UPDATE queue.reviews SET scored_at = now() WHERE id = %(id)s"
UNSCORED = "SELECT count(*) FROM queue.reviews WHERE scored_at IS NULL AND status = 'pending'"


def _rows(cur) -> list[dict]:
    names = [d.name for d in cur.description]
    out = []
    for values in cur.fetchall():
        row = dict(zip(names, values))
        if row.get("flown") is not None:
            row["flown"] = row["flown"].isoformat()
        out.append(row)
    return out


def create(conn) -> None:
    conn.execute(SCHEMA)


def insert(conn, rows: list[dict]) -> int:
    """Adds the reviews not yet in the queue (by id); returns how many were new."""
    added = 0
    with conn.transaction():
        for row in rows:
            added += conn.execute(INSERT, {c: row.get(c) for c in COLUMNS}).rowcount
    return added


def claim(conn, batch_id: str, n: int, replay: int = 0) -> dict:
    """Up to n pending reviews for this batch (now claimed), newest first, plus up to `replay` already-trained ones."""
    with conn.transaction():
        new = _rows(conn.execute(CLAIM, {"n": n, "batch_id": batch_id}))
        old = _rows(conn.execute(REPLAY, {"n": replay})) if replay else []
    return {"batch_id": batch_id, "reviews": new, "replay": old}


def finish(conn, batch_id: str, ok: bool) -> int:
    """ok: the batch's reviews become trained; not ok: they go back to pending. Returns how many rows changed."""
    return conn.execute(DONE if ok else RELEASE, {"batch_id": batch_id}).rowcount


def release_stale(conn) -> int:
    return conn.execute(RELEASE_STALE).rowcount


def stats(conn) -> dict:
    """Counts per status, how many wait to be scored, the last arrival. Numbers and dates only, never text."""
    out = {"pending": 0, "claimed": 0, "trained": 0, "last_received": None}
    last = []
    for status, count, received in conn.execute(STATS).fetchall():
        out[status] = int(count)
        last += [received] if received else []
    out["last_received"] = max(last).isoformat() if last else None
    out["total"] = out["pending"] + out["claimed"] + out["trained"]
    out["unscored"] = int(conn.execute(UNSCORED).fetchone()[0])
    return out


def to_score(conn, n: int) -> list[dict]:
    return _rows(conn.execute(TO_SCORE, {"n": n}))


def mark_scored(conn, review_id: str) -> None:
    conn.execute(MARK_SCORED, {"id": review_id})
