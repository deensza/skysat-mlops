#!/usr/bin/env bash
# SkySat: the whole demo in one script, in under 5 minutes. Every step runs against the real server, the real
# registry and GitHub, prints the command it runs and what came back, and waits for Enter before the next one.
# It shows what the model can do (step 2) and why this is a level 2 MLOps project in small (the other steps).
# The longer live demo with a real training, by hand: demo/demo.md.
#
#   bash demo/demo.sh                 the walk-through; about 2 minutes of machine time, the rest is your talking
#   bash demo/demo.sh --timed         the same, but each pause ends by itself after 12 s: the whole run stays under 5 min
#   bash demo/demo.sh --no-pause      no pauses at all (rehearsal)
#   bash demo/demo.sh --quick         step 2 sends 4 reviews instead of 9 (saves about 35 s)
#   bash demo/demo.sh --review "..."  step 2 starts with your own review (someone in the room can give it)
#   bash demo/demo.sh --skip-post     do not add a review to the queue in step 3
#   bash demo/demo.sh --live          step 6 runs the loop now: drift check -> Train -> rented GPU -> new model
#                                     (needs AUTO_RETRAIN=true, drift on the server and 300 ready reviews; about 10
#                                     minutes and at most about $0.26; if a condition is missing it says which and
#                                     shows the recorded run instead). Not a 5-minute demo.
#   bash demo/demo.sh --train         starts a real training as the first thing (the Train workflow by hand: a rented
#                                     RunPod GPU, at most about $0.26), goes through the steps while it runs, and shows
#                                     its result as the last thing. A run takes 6 to 10 minutes; needs 300 ready
#                                     reviews in the queue. If it is still running at the end, the script waits
#                                     (q + Enter stops the waiting, the run goes on by itself).
#   bash demo/demo.sh --train-run ID  the same, for a Train run that is already started (in the browser: Actions ->
#                                     Train -> Run workflow): the script starts nothing and shows that run's result.
#
# It reads, and never prints: the server's address (SKYSAT_SERVER, or ~/.config/skysat/server-address) and the gateway
# logins team, github and trainer (SKYSAT_LOGIN_TEAM / _GITHUB / _TRAINER, or ~/.config/skysat/login-<name>).
# It changes nothing except: steps 2 and 3 send requests that are logged like any other, step 3 adds one made-up
# review to the queue, --live starts the drift check, and --train starts the Train workflow.
# Needs: curl, python3, gh (signed in), and .venv for the drift check (make install).

set -u
cd "$(dirname "$0")/.." || exit 1

REPO="${SKYSAT_REPO:-deensza/skysat-mlops}"
MLFLOW="${SKYSAT_MLFLOW:-https://dagshub.com/HawkOsm/skysat-mlops.mlflow}"
MODEL="skysat-reviews"
RECORDED_RUN="${SKYSAT_RECORDED_RUN:-44246174603}"   # 18 Nov 2026: the drift check that ended in version 7
                                                     # (17 Nov, version 6: 44111118456)
CFG="$HOME/.config/skysat"
PAUSE=1; LIVE=0; POST=1; AUTO=0; QUICK=0; TRAIN=0; T0=$(date +%s)
REVIEW=; TRAIN_RUN=
TRAIN_ARGS="${SKYSAT_TRAIN_ARGS:---replay 0 --epochs 3}"   # as the drift check calls it: the new reviews only, 3 passes
TRAIN_REASON="${SKYSAT_TRAIN_REASON:-by hand}"             # "by hand" skips the cooldown; any other reason waits for it
while [ $# -gt 0 ]; do
  case "$1" in
    --no-pause) PAUSE=0 ;;
    --timed) AUTO=12 ;;
    --quick) QUICK=1 ;;
    --live) LIVE=1 ;;
    --train) TRAIN=1 ;;
    --train-run) shift; TRAIN_RUN="${1:?--train-run needs the id of a Train run}" ;;
    --skip-post) POST=0 ;;
    --review) shift; REVIEW="${1:?--review needs a text}" ;;
    -h|--help) sed -n '2,29p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "unknown option $1 (see --help)"; exit 2 ;;
  esac
  shift
done

SERVER="${SKYSAT_SERVER:-$(cat "$CFG/server-address" 2>/dev/null)}"
[ -n "$SERVER" ] || { echo "Set SKYSAT_SERVER (the server's host name) or put it in $CFG/server-address"; exit 2; }
PY=python3; [ -x .venv/bin/python ] && PY=.venv/bin/python
TMP="$(mktemp -d)"; trap 'rm -rf "$TMP"' EXIT
if [ -t 1 ]; then B=$(tput bold 2>/dev/null); D=$(tput dim 2>/dev/null); N=$(tput sgr0 2>/dev/null); else B=; D=; N=; fi
RESULTS=()

pw() {  # the password of a gateway login, from the environment or the config folder
  local var="SKYSAT_LOGIN_$(echo "$1" | tr '[:lower:]' '[:upper:]')"
  if [ -n "${!var:-}" ]; then printf '%s' "${!var}"; else tr -d '\n' < "$CFG/login-$1" 2>/dev/null; fi
}
api() {  # api <login> <path> [curl options...]: the login travels in a config read from stdin, not on the command line
  local user="$1" path="$2"; shift 2
  printf 'user = "%s:%s"\n' "$user" "$(pw "$user")" | curl -sS -m 150 -K - "$@" "https://$SERVER$path"
}
clock() { local t=$(( $(date +%s) - T0 )); printf '%d:%02d' $((t / 60)) $((t % 60)); }
head_() { echo; echo "${B}== $1 ==${N}  ${D}[$(clock)]${N}"; [ -n "${2:-}" ] && echo "$2"; echo; }
cmd() { echo "${D}\$ $*${N}"; }
note() { echo "   $*"; }
done_() { RESULTS+=("$1|$2"); echo; echo "   ${B}$1${N}: $2"; }
pause() {  # Enter for the next step; with --timed it also goes on by itself
  [ "$PAUSE" = 1 ] || return 0
  echo
  if [ "$AUTO" -gt 0 ]; then read -r -t "$AUTO" -p "${D}[Enter, or ${AUTO} s]${N} " _ || echo; else read -r -p "${D}[Enter for the next step]${N} " _; fi
  return 0
}
json() { python3 -c "import json,sys
d=json.load(sys.stdin)
$1"; }
newest_run() { gh run list -R "$REPO" --workflow "$1" --limit 1 --json databaseId --jq '.[0].databaseId' 2>/dev/null; }
start_run() {  # start_run <workflow> [gh options...]: starts it on main and prints the new run's id
  # The new run is told apart by its id, not by time: this PC's clock and GitHub's need not agree.
  local wf="$1" last id; shift
  last=$(newest_run "$wf")
  gh workflow run "$wf" -R "$REPO" --ref main "$@" >/dev/null 2>&1 || return 1
  for _ in $(seq 1 12); do
    sleep 5; id=$(newest_run "$wf")
    if [ -n "$id" ] && [ "$id" != "$last" ]; then echo "$id"; return 0; fi
  done
  return 1
}
run_state() {  # one line: where a run is now, and the step it is in
  gh run view "$1" -R "$REPO" --json status,conclusion,jobs --jq \
    '.status + (if (.conclusion // "") != "" then " (" + .conclusion + ")" else "" end) + ([.jobs[].steps[]? | select(.status == "in_progress") | ", in step \"" + .name + "\""] | join(""))' 2>/dev/null
}
minutes_since() { echo $(( ($(date +%s) - $1) / 60 )); }
run_took() {  # how long a finished run took, by GitHub's own clock
  gh run view "$1" -R "$REPO" --json createdAt,updatedAt --jq '((.updatedAt | fromdate) - (.createdAt | fromdate)) as $s | "\($s / 60 | floor) min \($s % 60) s"' 2>/dev/null
}

echo "${B}SkySat${N}: what the model does, and what makes this a level 2 MLOps project in small, on the running system."
echo "Server: https://<server>   Registry: $MLFLOW   Repository: $REPO"

# ----------------------------------------------------------------------------------------------------------------
TRAIN_ID=
if [ -n "$TRAIN_RUN" ]; then
  head_ "0. A training was started before this demo, on a rented GPU; its result is the last thing shown" \
        "Renting the GPU, starting it and the gate take 6 to 10 minutes, so it runs while the other steps are shown."
  cmd "gh run view $TRAIN_RUN     (https://github.com/$REPO/actions/runs/$TRAIN_RUN)"
  state=$(run_state "$TRAIN_RUN")
  if [ -n "$state" ]; then TRAIN_ID=$TRAIN_RUN; note "run $TRAIN_ID is $state"
  else note "run $TRAIN_RUN could not be read (gh): the recorded run in step 6 stands in"; fi
  pause
elif [ "$TRAIN" = 1 ]; then
  head_ "0. A training starts now, on a rented GPU; its result is the last thing shown" \
        "Renting the GPU, starting it and the gate take 6 to 10 minutes, so it runs while the other steps are shown."
  ready=$(api trainer /queue/stats 2>/dev/null | json "print(d.get('ready', d['pending']))" 2>/dev/null); ready=${ready:-0}
  note "reviews in the queue that the model has scored and nobody has trained on: $ready (a training needs 300)"
  if [ "$ready" -ge 300 ]; then
    cmd "gh workflow run train.yml -f reason='$TRAIN_REASON' -f trainer_args='$TRAIN_ARGS'      (at most about \$0.26)"
    if TRAIN_ID=$(start_run train.yml -f "reason=$TRAIN_REASON" -f "trainer_args=$TRAIN_ARGS"); then
      note "run $TRAIN_ID started: https://github.com/$REPO/actions/runs/$TRAIN_ID"
      note "on that page the line \"Pod ... created on <GPU>\" appears after about a minute: a GPU is rented"
    else
      TRAIN_ID=; note "the run could not be started or found (gh): the recorded run in step 6 stands in"
    fi
  else
    note "too few, so nothing is started: the recorded run in step 6 stands in"
  fi
  pause
fi

# ----------------------------------------------------------------------------------------------------------------
head_ "1. A model is serving, and it is the one the registry says" \
      "Level 2 keeps models in a registry; the server serves what the registry names, not a file someone copied."
cmd "curl https://<server>/model                      (no login)"
code=$(curl -sS -o /dev/null -w '%{http_code}' -m 60 "https://$SERVER/model")
note "HTTP $code$([ "$code" = 401 ] && echo ': the gateway refuses a request without a login')"
cmd "curl -u team:*** https://<server>/model"
api team /model > "$TMP/model.json"
SERVING=$(json "print(d.get('version',''))" < "$TMP/model.json" 2>/dev/null)
json "print('   serving:', d.get('name'), 'version', d.get('version'), '| alias', d.get('alias'), '| base model', d.get('base_model'), '| loaded', str(d.get('loaded_at'))[:19], 'UTC')" < "$TMP/model.json" 2>/dev/null \
  || note "could not read /model: $(head -c 200 "$TMP/model.json")"
cmd "MLflow registry (REST, read-only): the versions of $MODEL, their gate numbers, and the champion alias"
python3 - "$MLFLOW" "$MODEL" "${SERVING:-?}" "$TMP/champion-complaint.txt" > "$TMP/registry.txt" 2>&1 <<'PYEOF'
import http.client, json, sys, urllib.parse
base, model, serving = sys.argv[1], sys.argv[2], sys.argv[3]
u = urllib.parse.urlparse(base)
conn = http.client.HTTPSConnection(u.netloc, timeout=60)  # one connection for all the reads
def get(path, **query):
    conn.request("GET", f"{u.path}/api/2.0/mlflow/{path}?{urllib.parse.urlencode(query)}")
    return json.loads(conn.getresponse().read())
champion = get("registered-models/alias", name=model, alias="champion")["model_version"]["version"]
versions = sorted(get("model-versions/search", filter=f"name='{model}'")["model_versions"], key=lambda v: int(v["version"]))
print(f"   {'version':>7}  {'MAE':>5}  {'recommended':>11}  {'complaint':>9}  {'combined':>8}  {'champion then':>13}  trained on            ")
for v in versions:
    data = get("runs/get", run_id=v["run_id"])["run"]["data"]
    m = {x["key"]: x["value"] for x in data.get("metrics", [])}
    p = {x["key"]: x["value"] for x in data.get("params", [])}
    num = lambda k, f: format(m[k], f) if k in m else "-"
    mark = "  <- champion" if v["version"] == champion else ""
    if v["version"] == champion and "complaint_accuracy" in m:
        open(sys.argv[4], "w").write(f"{m['complaint_accuracy']:.0%} on {p.get('n_gate', 'the')} held-back reviews")
    print(f"   {v['version']:>7}  {num('score_mae', '.2f'):>5}  {num('recommended_accuracy', '.1%'):>11}  "
          f"{num('complaint_accuracy', '.1%'):>9}  {num('combined', '.3f'):>8}  {num('champion_combined', '.3f'):>13}  "
          f"{p.get('device', '-').replace('NVIDIA ', '').replace('GeForce ', '')[:22]:<22}{mark}")
print(f"   champion alias -> version {champion}; the server serves version {serving}: "
      + ("the same" if champion == serving else "NOT the same"))
sys.exit(0 if champion == serving else 1)
PYEOF
rc=$?; cat "$TMP/registry.txt"
note "\"champion then\" is the serving model's score on the same held-back reviews: a version is promoted only if it beats it."
if [ "$rc" = 0 ] && [ "$code" = 401 ]; then done_ "1 registry and serving" "shown"; else done_ "1 registry and serving" "PROBLEM (see above)"; fi
pause

# ----------------------------------------------------------------------------------------------------------------
head_ "2. What the model can do" \
      "It reads a review and writes three things: a score out of 10, would the passenger recommend the airline, and
the main complaint (seat, service, food, entertainment, ground service, wifi, value, or none). One request each,
through the gateway, on the server's CPU. \"A person would say\" is the complaint label we expect; it is not sent."
cat > "$TMP/reviews.json" <<'EOF'
[
 {"what": "a happy passenger", "expect": "none", "cabin": "Economy Class", "travel_type": "Family Leisure", "title": "Lovely flight",
  "review": "Everything went smoothly from check-in to landing. The crew were warm and attentive, the meal was tasty and hot, and my seat had plenty of legroom. We even arrived twenty minutes early. I will fly with them again."},
 {"what": "bad food", "expect": "food", "cabin": "Economy Class", "travel_type": "Solo Leisure", "title": "Cold dinner",
  "review": "The lasagne was still frozen in the middle and the bread was stale. Boarding was on time and the seat was fine, but for a nine hour flight the food was simply not acceptable."},
 {"what": "rude cabin crew", "expect": "service", "cabin": "Economy Class", "travel_type": "Solo Leisure", "title": "Rude crew",
  "review": "The cabin crew ignored the call button for an hour and one of them rolled her eyes when I asked for water. The plane was clean and we landed on time, but I felt unwelcome the whole way."},
 {"what": "typos, no details, HTML", "expect": "ground service", "cabin": "", "travel_type": "", "title": "",
  "review": "worst flite ever!!! 3 hrs delay, no info, rude staff at gate.. NEVER again <br> <br>"},
 {"what": "wifi that never worked", "expect": "wifi", "cabin": "Economy Class", "travel_type": "Business", "title": "Offline for eleven hours",
  "review": "I bought the wifi package to work on board and it never connected once. The crew said it often fails. Otherwise an ordinary flight: decent meal, seat was okay, on time."},
 {"what": "broken screens", "expect": "entertainment", "cabin": "Economy Class", "travel_type": "Family Leisure", "title": "Broken screens",
  "review": "None of the screens in our row worked on a ten hour flight with two children, and there were no headphones left. Food and crew were good and the seats were fine."},
 {"what": "no legroom", "expect": "seat", "cabin": "Premium Economy", "travel_type": "Business", "title": "No room at all",
  "review": "I paid extra for premium economy and my knees touched the seat in front for seven hours. The seat would not recline and the cushion was worn through. Crew and food were fine."},
 {"what": "trouble on the ground", "expect": "ground service", "cabin": "Economy Class", "travel_type": "Couple Leisure", "title": "Lost at the airport",
  "review": "The flight itself was pleasant, but the check-in queue took ninety minutes, the gate changed twice without any announcement, and my suitcase arrived two days late with nobody at the desk able to help."},
 {"what": "too expensive", "expect": "value", "cabin": "Business Class", "travel_type": "Business", "title": "Not worth the money",
  "review": "A comfortable seat and polite crew, but at four times the economy fare I expected more than a small sandwich and a paper cup. Fine flight, far too expensive for what you get."}
]
EOF
OWN="$REVIEW" QUICK="$QUICK" python3 - "$TMP/reviews.json" > "$TMP/bodies.tsv" <<'PYEOF'
import json, os, sys
rows = json.load(open(sys.argv[1]))
if os.environ["QUICK"] == "1":
    rows = rows[:4]
if os.environ["OWN"]:
    rows.insert(0, {"what": "your own review", "expect": "?", "cabin": "", "travel_type": "", "title": "", "review": os.environ["OWN"]})
for r in rows:
    body = {"airline": "Imaginary Air", **{k: r[k] for k in ("cabin", "travel_type", "title", "review")}}
    print("\t".join([r["what"], r["expect"], r["review"][:74] + ("..." if len(r["review"]) > 74 else ""), json.dumps(body)]))
PYEOF
cmd "curl -u team:*** -X POST https://<server>/score -d '{\"airline\": ..., \"cabin\": ..., \"title\": ..., \"review\": ...}'    ($(wc -l < "$TMP/bodies.tsv") reviews, one after the other)"
printf '   %-25s %-6s %-12s %-15s %-19s %-8s %s\n' "the review is about" "score" "recommended" "main complaint" "a person would say" "" "time"
asked=0; agreed=0; failed=0; version=
while IFS=$'\t' read -r what expect short body; do
  printf '%s' "$body" > "$TMP/body.json"  # a file: curl's stdin is taken by the login
  out=$(api team /score -X POST -H 'Content-Type: application/json' --data-binary "@$TMP/body.json" -w '\n%{time_total}')
  line=$(printf '%s' "$out" | EXPECT="$expect" WHAT="$what" python3 -c "
import json, os, sys
body, _, seconds = sys.stdin.read().rpartition('\n')
d = json.loads(body); expect = os.environ['EXPECT']
mark = '' if expect == '?' else ('agrees' if d['complaint'] == expect else 'DIFFERS')
print('   %-25s %-6s %-12s %-15s %-19s %-8s %.1f s' % (os.environ['WHAT'], str(d['score']) + '/10', d['recommended'], d['complaint'], expect, mark, float(seconds)))
print('RESULT', d['model_version'], 'x' if expect == '?' else int(d['complaint'] == expect))" 2>/dev/null)
  if [ -n "$line" ]; then
    echo "$line" | grep -v '^RESULT'; echo "      ${D}\"$short\"${N}"
    r=$(echo "$line" | grep '^RESULT'); version=$(echo "$r" | cut -d' ' -f2)
    case "$(echo "$r" | cut -d' ' -f3)" in 1) asked=$((asked + 1)); agreed=$((agreed + 1)) ;; 0) asked=$((asked + 1)) ;; esac
  else
    failed=$((failed + 1)); printf '   %-25s no answer: %s\n' "$what" "$(printf '%s' "$out" | head -c 120)"
  fi
done < "$TMP/bodies.tsv"
measured=$(cat "$TMP/champion-complaint.txt" 2>/dev/null)
note "answered by model version ${version:-?}. The complaint agrees with a person on $agreed of $asked here${measured:+; at its gate it was $measured}."
note "It is a small model on a CPU: about 7 s a review, and it is wrong some of the time. That is what the gate measures."
cmd "curl -u team:*** -X POST https://<server>/score -d '{\"review\": \"\"}'      (an empty review)"
printf '{"airline": "Imaginary Air", "review": ""}' > "$TMP/body.json"
code=$(api team /score -X POST -H 'Content-Type: application/json' --data-binary "@$TMP/body.json" -o /dev/null -w '%{http_code}')
note "HTTP $code$([ "$code" = 422 ] && echo ': refused before it reaches the model')"
# POST /ask: the same model for a review typed as plain text; its answers are not logged
ASK="The screen at my seat was broken and so was the one next to it. No films and no music for ten hours."
cmd "curl -u team:*** -X POST https://<server>/ask -H 'Content-Type: text/plain' -d '$ASK'"
printf '%s' "$ASK" > "$TMP/body.txt"
api team /ask -X POST -H 'Content-Type: text/plain' --data-binary "@$TMP/body.txt" 2>/dev/null \
  | json "print('   ' + str(d['score']) + '/10 | recommended', d['recommended'], '| complaint', d['complaint'], '| version', d['model_version'], '| not logged')" 2>/dev/null \
  || note "no answer from /ask (the server may run an image from before 18 Nov 2026)"
note "The same in the browser: https://<server>/docs -> POST /ask -> Try it out: one text box, type any review, Execute."
note "What it cannot do: other languages were not tested, it reads only the first 250 words, it knows only these 8 complaints."
if [ "$failed" = 0 ] && [ "$asked" -gt 0 ]; then done_ "2 what the model can do" "shown ($agreed of $asked complaints as a person would say)"
else done_ "2 what the model can do" "PROBLEM ($failed requests without an answer)"; fi
pause

# ----------------------------------------------------------------------------------------------------------------
head_ "3. A passenger's review goes through the queue and becomes a measurement" \
      "The review arrives with the passenger's own score. The intake queues it, the model answers it as live traffic,
and prediction and truth are logged as numbers (never the text). That log is what monitoring reads."
qline() { api trainer /queue/stats 2>/dev/null | json "print('   queue: pending', d['pending'], '| ready to train on', d.get('ready'), '| not scored yet', d['unscored'], '| trained', d['trained'])" 2>/dev/null; }
cmd "curl -u trainer:*** https://<server>/queue/stats"
qline || note "(no trainer login here: the queue's numbers are skipped)"
before=$(api team "/scores/recent?hours=1" | json "print(d.get('labelled_score', 0))" 2>/dev/null); before=${before:-0}
if [ "$POST" = 1 ]; then
  rid="demo:$(date -u +%Y%m%dT%H%M%S)"
  cmd "curl -u team:*** -X POST https://<server>/reviews -d '{\"review_id\": \"$rid\", \"score\": 2, \"recommended\": \"no\", \"ratings\": {\"food\": 1, ...}, ...}'"
  cat > "$TMP/review.json" <<EOF
{"review_id": "$rid", "airline": "Imaginary Air", "cabin": "Economy Class", "travel_type": "Solo Leisure",
 "title": "Cold dinner", "review": "The lasagne was still frozen in the middle and the bread was stale. Boarding was on time and the seat was fine, but for a long flight the food was not acceptable.",
 "flown": "2026-11-01", "score": 2, "recommended": "no", "ratings": {"food": 1, "seat": 4, "service": 3}}
EOF
  api team /reviews -X POST -H 'Content-Type: application/json' --data-binary "@$TMP/review.json" \
    | json "print('   intake: queued', d.get('queued'), '| complaint label worked out from the sub-ratings:', d.get('complaints'))" 2>/dev/null \
    || note "the intake did not accept it"
  note "waiting for the model to answer it as live traffic (oldest first; up to 2 minutes) ..."
  after=$before
  for _ in $(seq 1 24); do
    sleep 5
    after=$(api team "/scores/recent?hours=1" | json "print(d.get('labelled_score', 0))" 2>/dev/null); after=${after:-$before}
    [ "$after" -gt "$before" ] && break
  done
  qline
  if [ "$after" -gt "$before" ]; then note "labelled requests in the last hour: $before -> $after"; done_ "3 intake -> queue -> scored -> logged" "shown"
  else note "labelled requests in the last hour still $before: reviews ahead of it in the queue are scored first"; done_ "3 intake -> queue -> scored -> logged" "not finished in 2 minutes"; fi
else
  done_ "3 intake -> queue -> scored -> logged" "skipped (--skip-post)"
fi
cmd "curl -u team:*** 'https://<server>/scores/recent?hours=1'      (totals only: this is all monitoring ever sees)"
api team "/scores/recent?hours=1" | json "print('   last hour:', d['count'], 'requests,', d['labelled_score'], 'with the true answer | MAE', d['mae'], '| recommended accuracy', d['recommended_accuracy'], '| by version', d['model_versions'])
if (d['labelled_score'] or 0) < 100: print('   (so few labelled requests say nothing yet: that is why the drift check in step 4 waits for 100)')" 2>/dev/null
note "The same log feeds the Grafana dashboard: https://<server>/grafana/"
pause

# ----------------------------------------------------------------------------------------------------------------
head_ "4. Monitoring decides: the drift check" \
      "It judges only the serving version, over 24 hours, against the gate's limits, and only calls it drift when the
miss is bigger than chance (one standard error) on at least 100 labelled requests."
cmd "python -m llm.drift_check --url https://<server> --hours 24"
API_USER=github API_PASSWORD="$(pw github)" "$PY" -m llm.drift_check --url "https://$SERVER" --hours 24 > "$TMP/drift.txt" 2>&1; rc=$?
sed "s/$SERVER/<server>/g" "$TMP/drift.txt" | sed '/^| Complaint/,$d' | grep -v '^$' | sed 's/^/   /' 
DRIFT_NOW=$(grep -c '^## Drift check: DRIFT' "$TMP/drift.txt")
if [ "$rc" = 0 ]; then done_ "4 drift check" "shown ($([ "$DRIFT_NOW" = 1 ] && echo 'verdict: DRIFT' || echo 'verdict: no drift'))"; else done_ "4 drift check" "PROBLEM (exit code $rc)"; fi
note "On GitHub the same check runs as the workflow \"Drift check\"; on drift it opens an issue and calls the training."
pause

# ----------------------------------------------------------------------------------------------------------------
head_ "5. The pipeline's own code is tested and packaged on every change (CI)" \
      "Level 2 is about the pipeline, not only the model: its code goes through tests and a build before it is used."
cmd "gh run list --workflow ci.yml --branch main --limit 1"
ci=$(gh run list -R "$REPO" --workflow ci.yml --branch main --limit 1 --json databaseId --jq '.[0].databaseId' 2>/dev/null)
if [ -n "$ci" ]; then
  gh run view "$ci" -R "$REPO" --json conclusion,headSha,createdAt,jobs \
    --jq '"   run '"$ci"' of \(.createdAt) on commit \(.headSha[0:7]): \(.conclusion)", (.jobs[] | "     job \(.name): \(.conclusion)")' 2>/dev/null
  gh run view "$ci" -R "$REPO" --log 2>/dev/null > "$TMP/ci.log"
  note "tests in that run: $(grep -a ' passed' "$TMP/ci.log" | tail -1 | sed 's/.*Z //; s/=//g; s/^ *//')"
  note "among them tests/test_pipeline.py: the real trainer against a real PostgreSQL, only the GPU training replaced."
  note "the image job built the Docker image the server runs; the deploy job is off (the server is updated by hand)."
  ci_ok=$(gh run view "$ci" -R "$REPO" --json conclusion --jq .conclusion 2>/dev/null)
  done_ "5 CI on main" "$([ "$ci_ok" = success ] && echo shown || echo "last run: $ci_ok")"
else
  done_ "5 CI on main" "PROBLEM (gh could not list the runs)"
fi
pause

# ----------------------------------------------------------------------------------------------------------------
show_run() {  # the steps and the telling log lines of one Drift check run
  local id="$1"
  gh run view "$id" -R "$REPO" --json conclusion,createdAt,event,jobs --jq \
    '"   run '"$id"' (\(.event)) of \(.createdAt): \(.conclusion)", (.jobs[] | "     job \(.name): \(.conclusion)", (.steps[] | select(.name | test("Drift check|Open an issue|Cooldown|Queue before|Train on a rented GPU|A GPU was rented|No pod may be left|Close the drift issue")) | "        \(.name): \(.conclusion)"))' 2>/dev/null
  gh run view "$id" -R "$REPO" --log 2>/dev/null > "$TMP/run.log"
  echo "   from its log:"
  grep -a 'Drift check: \|model version .* only\|| score MAE \|| recommended accuracy \|Pod [a-z0-9]* created on\|Pod [a-z0-9]* is gone\|From the server:\|Outcome: \|No GPU had capacity\|Training skipped\|Too few new reviews\|Trainer pods left behind\|Closed issue #\|No open drift issue' "$TMP/run.log" \
    | cut -f3- | sed 's/^[0-9-]*T\([0-9:]*\)[0-9.]*Z /\1  /' | grep -v '36;1m' | cut -c1-200 | sed 's/^/        /'
}
head_ "6. Training starts by itself and ends in a new serving model (continuous training)" \
      "Drift check -> issue -> Train workflow -> a GPU rented on RunPod -> trainer -> gate -> registry -> the server
loads the new champion. From the drift check on, nobody touches anything."
LIVE_DONE=0
if [ "$LIVE" = 1 ]; then
  auto=$(gh variable get AUTO_RETRAIN -R "$REPO" 2>/dev/null)
  ready=$(api trainer /queue/stats 2>/dev/null | json "print(d.get('ready', d['pending']))" 2>/dev/null); ready=${ready:-0}
  note "conditions for a live run: AUTO_RETRAIN=$auto (needs true) | drift now: $([ "$DRIFT_NOW" = 1 ] && echo yes || echo no) (needs yes) | ready reviews: $ready (needs 300)"
  if [ "$auto" = true ] && [ "$DRIFT_NOW" = 1 ] && [ "$ready" -ge 300 ]; then
    cmd "gh workflow run drift.yml          (the only command typed; a rented GPU costs at most about \$0.26)"
    live_id=$(start_run drift.yml) || live_id=
    if [ -n "$live_id" ]; then
      note "run $live_id started: https://github.com/$REPO/actions/runs/$live_id  (about 10 minutes; watching)"
      gh run watch "$live_id" -R "$REPO" --interval 20 >/dev/null 2>&1
      show_run "$live_id"
      api team /model > "$TMP/model-after.json"
      NOW=$(json "print(d.get('version',''))" < "$TMP/model-after.json" 2>/dev/null)
      note "the server served version ${SERVING:-?} before this run and serves version ${NOW:-?} now"
      if grep -aq "'registered': True" "$TMP/run.log"; then done_ "6 continuous training, live" "shown (run $live_id)"; LIVE_DONE=1
      else note "this run did not end in a new version (the reason is in the lines above): the recorded run follows"; fi
    else
      note "could not find the run that was started: the recorded run follows"
    fi
  else
    note "a condition is missing, so nothing is started: the recorded run follows"
  fi
fi
if [ -n "$TRAIN_ID" ]; then
  note "the training of step 0: run $TRAIN_ID is $(run_state "$TRAIN_ID")."
  note "Its result is the last thing shown. What a finished run looks like, with the drift check in front of it:"
fi
if [ "$LIVE_DONE" = 0 ]; then
  cmd "gh run view $RECORDED_RUN --log     (recorded: https://github.com/$REPO/actions/runs/$RECORDED_RUN)"
  show_run "$RECORDED_RUN"
  if grep -aq "'registered': True" "$TMP/run.log"; then done_ "6 continuous training" "shown from the recorded run $RECORDED_RUN"
  else done_ "6 continuous training" "PROBLEM (the recorded run could not be read)"; fi
fi
note "The pod's own log is not shown: RunPod refuses it to this project's API key, so the outcome is read from the server."
pause

# ----------------------------------------------------------------------------------------------------------------
head_ "7. What keeps it from costing money or serving a bad model" "Each guard is a step or a line in the run above."
cat <<EOF
   no GPU without 300 ready reviews ............. step "Queue before"
   no automatic run within 6 h of a rental ...... step "Cooldown" (a run that found no free GPU does not count)
   a pod never outlives its job ................. it deletes itself, a 45-minute timer, the launcher, and the last
                                                  step: "Trainer pods left behind: 0"
   cheapest GPU first ........................... 7 types from \$0.13 to \$0.34 an hour; at most about \$0.26 a run
   a new model must pass the gate ............... MAE <= 1.0, recommended >= 90%, complaint >= 65%, on held-back reviews
   and beat the serving one on the same reviews . the "champion then" column in step 1
   going back is one command .................... make llm-promote VERSION=<n> SHARED=yes (the server follows in a minute)
EOF
note "switch for the loop: AUTO_RETRAIN=$(gh variable get AUTO_RETRAIN -R "$REPO" 2>/dev/null || echo '?')   cooldown hours: $(gh variable get TRAIN_COOLDOWN_HOURS -R "$REPO" 2>/dev/null || echo '6 (default)')"
done_ "7 guards" "listed"
pause

# ----------------------------------------------------------------------------------------------------------------
head_ "8. What a full level 2 system has that this one does not" "Said before anyone asks."
cat <<EOF
   - GitHub's hourly schedule for the drift check fires only now and then; the runs shown were started by hand.
   - The deploy job is off: a new image reaches the server by hand (az vm run-command).
   - No orchestrator (GitHub Actions plays that part), no feature store, one server and no staging environment.
   - The loop has ended in a new serving model twice (17 and 18 Nov 2026), both times on simulated drift.
EOF

# ----------------------------------------------------------------------------------------------------------------
if [ -n "$TRAIN_ID" ]; then
  head_ "9. The training of step 0" \
        "The same workflow the drift check calls: queue check -> rented GPU -> trainer -> gate -> registry -> server."
  cmd "gh run view $TRAIN_ID     (https://github.com/$REPO/actions/runs/$TRAIN_ID)"
  [ "$PAUSE" = 1 ] && note "(it is checked every 20 s; q + Enter stops the waiting, the run goes on by itself)"
  st=
  while :; do
    st=$(gh run view "$TRAIN_ID" -R "$REPO" --json status --jq .status 2>/dev/null)
    [ "$st" = completed ] && break
    note "$(date +%H:%M:%S)  $(run_state "$TRAIN_ID")"
    key=
    if [ "$PAUSE" = 1 ]; then read -r -t 20 key || true; else sleep 20; fi
    [ "$key" = q ] && break
    [ "$(minutes_since "$T0")" -ge 30 ] && break   # a run stops its pod after 45 minutes; the script does not wait that long
  done
  if [ "$st" = completed ]; then
    show_run "$TRAIN_ID"
    steps=$(gh run view "$TRAIN_ID" -R "$REPO" --json jobs --jq '[.jobs[].steps[] | "\(.name)=\(.conclusion)"] | join(";")' 2>/dev/null)
    api team /model > "$TMP/model-after.json"
    NOW=$(json "print(d.get('version',''))" < "$TMP/model-after.json" 2>/dev/null)
    api trainer /queue/stats 2>/dev/null | json "print('   queue now: pending', d['pending'], '| trained', d['trained'])" 2>/dev/null
    note "the server served version ${SERVING:-?} when the demo began and serves version ${NOW:-?} now; the run took $(run_took "$TRAIN_ID")"
    if grep -aq "'registered': True" "$TMP/run.log"; then
      if grep -aq "'promoted': True" "$TMP/run.log"; then done_ "9 live training" "shown (run $TRAIN_ID: a new version, promoted, serving)"
      else done_ "9 live training" "shown (run $TRAIN_ID: a new version, registered; it did not beat the champion, which stays)"; fi
    elif grep -aq "No GPU had capacity" "$TMP/run.log"; then done_ "9 live training" "no GPU was free on RunPod: nothing was rented (run $TRAIN_ID)"
    else
      case "$steps" in
        *"Cooldown=success"*"Queue before=skipped"*) done_ "9 live training" "stopped by the cooldown: nothing was rented (run $TRAIN_ID)" ;;
        *"Queue before=success"*"Train on a rented GPU=skipped"*) done_ "9 live training" "too few ready reviews: nothing was rented (run $TRAIN_ID)" ;;
        *) done_ "9 live training" "ended without a new version (run $TRAIN_ID; the reason is in the lines above)" ;;
      esac
    fi
  else
    note "still running: https://github.com/$REPO/actions/runs/$TRAIN_ID   (the new version will be in step 1's table)"
    done_ "9 live training" "still running (run $TRAIN_ID)"
  fi
fi
echo; echo "${B}Summary${N}  ${D}[the whole run took $(clock)]${N}"
for r in "${RESULTS[@]}"; do printf '   %-42s %s\n' "${r%%|*}" "${r#*|}"; done
case "${RESULTS[*]}" in *PROBLEM*) exit 1 ;; esac
exit 0
