

#include <bits/stdc++.h>
using namespace std;

// ─── Types ───────────────────────────────────────────────────────────────────
using ll  = long long;
using ull = unsigned long long;

// ─── Globals ─────────────────────────────────────────────────────────────────
int N, M;
vector<ll>          skill;       // 1-indexed skill ratings
vector<unordered_set<int>> adj;  // adjacency sets (1-indexed)
vector<bool>        inTeam;      // current solution
ll                  bestScore = 0;
vector<bool>        bestTeam;

// ─── Timer ───────────────────────────────────────────────────────────────────
auto START_TIME = chrono::steady_clock::now();
double elapsed() {
    return chrono::duration<double>(chrono::steady_clock::now() - START_TIME).count();
}

void signalHandler(int sig) {
    ll finalScore = 0;
    vector<int> chosen;
    for (int i = 1; i <= N; i++) {
        if (bestTeam[i]) {
            finalScore += skill[i];
            chosen.push_back(i);
        }
    }
    cout << finalScore << "\n";
    for (int i = 0; i < (int)chosen.size(); i++) {
        cout << chosen[i];
        if (i+1 < (int)chosen.size()) cout << ' ';
    }
    cout << "\n";
    cout.flush();
    exit(0);
}


// ─── Score ───────────────────────────────────────────────────────────────────
ll computeScore(const vector<bool>& team) {
    ll s = 0;
    for (int i = 1; i <= N; i++) if (team[i]) s += skill[i];
    return s;
}

// ─── Validate ────────────────────────────────────────────────────────────────
bool isValid(const vector<bool>& team) {
    for (int u = 1; u <= N; u++) {
        if (!team[u]) continue;
        for (int v : adj[u]) if (team[v]) return false;
    }
    return true;
}

// ─── Save Best ───────────────────────────────────────────────────────────────
void tryUpdateBest(const vector<bool>& team, ll score) {
    if (score > bestScore) {
        bestScore = score;
        bestTeam  = team;
    }
}

// ════════════════════════════════════════════════════════════════════════════
// PHASE 1 — Greedy Construction
//   Sort by skill DESC. For each coder, add them if no conflict with current team.
//   Simple O(N log N + sum_of_degrees) solution.
// ════════════════════════════════════════════════════════════════════════════
void greedyConstruct() {
    // Sort indices by skill descending
    vector<int> order(N);
    iota(order.begin(), order.end(), 1);
    sort(order.begin(), order.end(), [](int a, int b){
        return skill[a] > skill[b];
    });

    fill(inTeam.begin(), inTeam.end(), false);

    for (int u : order) {
        bool ok = true;
        for (int v : adj[u]) {
            if (inTeam[v]) { ok = false; break; }
        }
        if (ok) inTeam[u] = true;
    }

    ll s = computeScore(inTeam);
    tryUpdateBest(inTeam, s);
}


void localSearch(double timeLimit) {
    inTeam  = bestTeam;
    ll score = bestScore;

    bool improved = true;
    while (improved && elapsed() < timeLimit) {
        improved = false;

        // Build out-list sorted by skill descending
        vector<int> outs;
        outs.reserve(N);
        for (int i = 1; i <= N; i++) if (!inTeam[i]) outs.push_back(i);
        sort(outs.begin(), outs.end(), [](int a, int b){
            return skill[a] > skill[b];
        });

        for (int u : outs) {
            if (elapsed() > timeLimit) break;

            // Collect blocking selected neighbors
            ll blockSum = 0;
            vector<int> block;
            block.reserve(adj[u].size());
            for (int v : adj[u]) {
                if (inTeam[v]) {
                    block.push_back(v);
                    blockSum += skill[v];
                }
            }

            ll gain = skill[u] - blockSum;
            if (gain > 0) {
                // Apply swap
                for (int v : block) inTeam[v] = false;
                inTeam[u] = true;
                score     += gain;
                improved   = true;
            }
        }

        tryUpdateBest(inTeam, score);
    }
}

// ════════════════════════════════════════════════════════════════════════════
// PHASE 3 — Simulated Annealing
//   Move types:
//     (A) Flip IN→OUT  : always valid, may hurt score
//     (B) Flip OUT→IN  : remove all blocking neighbors, add u
//   Accept bad moves with probability exp(delta/T)
//   Temperature decays geometrically.
// ════════════════════════════════════════════════════════════════════════════
void simulatedAnnealing(double timeLimit) {
    inTeam = bestTeam;
    ll score = bestScore;

    // SA parameters — tuned for 5-minute runs
    double T      = max(1.0, (double)bestScore * 0.01); // initial temperature
    double Tmin   = 0.1;
    double alpha  = 0.9999;   // cooling rate (slow decay for long runs)

    mt19937 rng(chrono::steady_clock::now().time_since_epoch().count());
    uniform_int_distribution<int> randNode(1, N);
    uniform_real_distribution<double> randProb(0.0, 1.0);

    long long iter = 0;

    while (elapsed() < timeLimit && T > Tmin) {
        iter++;

        int u = randNode(rng);

        if (inTeam[u]) {
            // Move A: remove u from team (always valid)
            ll delta = -skill[u];
            if (delta > 0 || randProb(rng) < exp((double)delta / T)) {
                inTeam[u] = false;
                score     += delta;
                tryUpdateBest(inTeam, score);
            }
        } else {
            // Move B: add u, evicting all blocking neighbors
            ll blockSum = 0;
            vector<int> block;
            for (int v : adj[u]) {
                if (inTeam[v]) {
                    block.push_back(v);
                    blockSum += skill[v];
                }
            }
            ll delta = skill[u] - blockSum;
            if (delta > 0 || randProb(rng) < exp((double)delta / T)) {
                for (int v : block) { inTeam[v] = false; score -= skill[v]; }
                inTeam[u]  = true;
                score     += skill[u];
                tryUpdateBest(inTeam, score);
            }
        }

        T *= alpha;

        // Periodically re-sync score (float drift guard)
        if (iter % 1000000 == 0) {
            score = computeScore(inTeam);
        }
    }
}

// ════════════════════════════════════════════════════════════════════════════
// PHASE 2b — Targeted Local Search with Priority Queue
//   More efficient version: only re-evaluates nodes whose neighborhood changed.
//   Good for sparse graphs with large N.
// ════════════════════════════════════════════════════════════════════════════
void priorityLocalSearch(double timeLimit) {
    inTeam = bestTeam;
    ll score = bestScore;

    // Max-heap: (skill, node) for OUT nodes
    priority_queue<pair<ll,int>> pq;
    vector<bool> inPQ(N+1, false);

    for (int i = 1; i <= N; i++) {
        if (!inTeam[i]) {
            pq.push({skill[i], i});
            inPQ[i] = true;
        }
    }

    while (!pq.empty() && elapsed() < timeLimit) {
        auto [sk, u] = pq.top(); pq.pop();
        inPQ[u] = false;

        if (inTeam[u]) continue; // already added by a previous swap

        ll blockSum = 0;
        vector<int> block;
        for (int v : adj[u]) {
            if (inTeam[v]) {
                block.push_back(v);
                blockSum += skill[v];
            }
        }

        ll gain = skill[u] - blockSum;
        if (gain > 0) {
            // Apply swap
            for (int v : block) {
                inTeam[v] = false;
                score -= skill[v];
                // Neighbors of evicted v become candidates again
                for (int w : adj[v]) {
                    if (!inTeam[w] && !inPQ[w]) {
                        pq.push({skill[w], w});
                        inPQ[w] = true;
                    }
                }
            }
            inTeam[u]  = true;
            score     += skill[u];
            tryUpdateBest(inTeam, score);
        }
    }
}

// ════════════════════════════════════════════════════════════════════════════
// MAIN
// ════════════════════════════════════════════════════════════════════════════
int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    signal(SIGTERM, signalHandler);
    signal(SIGINT, signalHandler);

    cin >> N >> M;

    skill.resize(N+1);
    adj.resize(N+1);
    inTeam.resize(N+1, false);
    bestTeam.resize(N+1, false);

    for (int i = 1; i <= N; i++) cin >> skill[i];
    for (int i = 0; i < M; i++) {
        int u, v; cin >> u >> v;
        adj[u].insert(v);
        adj[v].insert(u);
    }

    // ── Hard time limit: 300 seconds ─────────────────────────────────────
    const double GLOBAL_LIMIT = 299.9;

    // ── Phase 1: Greedy (instant) ─────────────────────────────────────────
    if (elapsed() < GLOBAL_LIMIT) greedyConstruct();

    // ── Phase 2a: Priority-queue local search (~30s) ──────────────────────
    if (elapsed() < GLOBAL_LIMIT) priorityLocalSearch(min(30.0, GLOBAL_LIMIT - elapsed()));

    // ── Phase 2b: Full local search sweeps (~60s) ─────────────────────────
    if (elapsed() < GLOBAL_LIMIT) localSearch(min(90.0, GLOBAL_LIMIT - elapsed()));

    // ── Phase 3: Simulated Annealing (remaining time) ─────────────────────
    if (elapsed() < GLOBAL_LIMIT) simulatedAnnealing(min(290.0, GLOBAL_LIMIT - elapsed()));

    // ── Output ────────────────────────────────────────────────────────────
    ll finalScore = 0;
    vector<int> chosen;
    for (int i = 1; i <= N; i++) {
        if (bestTeam[i]) {
            finalScore += skill[i];
            chosen.push_back(i);
        }
    }

    cout << finalScore << "\n";
    for (int i = 0; i < (int)chosen.size(); i++) {
        cout << chosen[i];
        if (i+1 < (int)chosen.size()) cout << ' ';
    }
    cout << "\n";

    return 0;
}