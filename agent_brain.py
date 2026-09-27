# agent_brain.py
# Q-Learning mozak agenta
# Koristi standardnu Q-tablicu (rječnik stanje → [q_gore, q_dolje, q_desno, q_lijevo])

import numpy as np
import pandas as pd


class QLearningAgent:

    def __init__(
        self,
        actions,
        learning_rate = 0.1,    # alpha  — koliko brzo agent uči
        gamma         = 0.9,    # gamma  — faktor popusta (važnost budućih nagrada)
        epsilon       = 1.0,    # epsilon — vjerovatnoća istraživanja (exploration)
        epsilon_min   = 0.01,   # minimalni epsilon (uvijek malo istražuje)
        epsilon_decay = 0.995,  # koliko se epsilon smanjuje po epizodi
    ):
        self.actions       = actions
        self.lr            = learning_rate
        self.gamma         = gamma
        self.epsilon       = epsilon
        self.epsilon_min   = epsilon_min
        self.epsilon_decay = epsilon_decay

        # Q-tablica: rječnik { 'stanje': [q_a0, q_a1, q_a2, q_a3] }
        self.q_table = {}

    # ── Inicijalizacija novog stanja u Q-tablici ─────────────────
    def _ensure_state(self, state):
        if state not in self.q_table:
            self.q_table[state] = [0.0] * len(self.actions)

    # ── Odabir akcije (ε-greedy) ──────────────────────────────────
    def choose_action(self, state):
        self._ensure_state(state)

        if np.random.uniform(0, 1) < self.epsilon:
            # Istraživanje — nasumična akcija
            return np.random.choice(self.actions)
        else:
            # Eksploatacija — best poznata akcija
            q_vals = self.q_table[state]
            return int(np.argmax(q_vals))

    # ── Bellmanova jednadžba — ažuriranje Q-tablice ───────────────
    def learn(self, state, action, reward, next_state):
        self._ensure_state(state)
        self._ensure_state(next_state)

        q_current  = self.q_table[state][action]
        q_next_max = max(self.q_table[next_state])

        # Q(s,a) ← Q(s,a) + α · [r + γ · max Q(s',a') − Q(s,a)]
        q_target   = reward + self.gamma * q_next_max
        td_error   = q_target - q_current

        self.q_table[state][action] += self.lr * td_error

        return abs(td_error)   # vraćamo grešku za praćenje

    # ── Smanjenje epsilona na kraju epizode ───────────────────────
    def decay_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            self.epsilon  = max(self.epsilon, self.epsilon_min)

    # ── Ispis Q-tablice (debug) ───────────────────────────────────
    def print_q_table(self):
        print('\n── Q-tablica ──')
        print(f'{"Stanje":<12} {"Gore":>8} {"Dolje":>8} {"Desno":>8} {"Lijevo":>8}')
        print('─' * 50)
        for state in sorted(self.q_table):
            vals = self.q_table[state]
            best = np.argmax(vals)
            row = f'{state:<12}'
            for i, v in enumerate(vals):
                marker = '*' if i == best else ' '
                row += f' {v:>7.3f}{marker}'
            print(row)

    # ── Export Q-tablice kao DataFrame ────────────────────────────
    def q_table_as_df(self):
        records = []
        for state, vals in self.q_table.items():
            records.append({
                'state'  : state,
                'q_gore' : vals[0],
                'q_dolje': vals[1],
                'q_desno': vals[2],
                'q_lijevo': vals[3],
                'best_action': ['gore','dolje','desno','lijevo'][int(np.argmax(vals))]
            })
        return pd.DataFrame(records).sort_values('state')
