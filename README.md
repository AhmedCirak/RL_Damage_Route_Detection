# Road Inspection RL — Q-Learning

Agent koji uči da pronađe najkraći put kroz mrežu ceste (10×10 grid), pri čemu
usput treba proći kroz što više oštećenih dionica (manje i veće štete) prije
nego stigne do cilja. Riješeno klasičnim **Q-Learningom** (bez neuronskih mreža),
uz interaktivnu vizualizaciju u Tkinteru.

## Kako radi

- **Grid 10×10** — svaka ćelija je jednog od pet tipova:
  - `FREE` — slobodna cesta (kazna -0.05 po koraku)
  - `ROAD_CLOSED` — neprohodna dionica (kazna -1.0, epizoda završava)
  - `MINOR_DAMAGE` — manje oštećenje (kazna -0.3, +0.2 bonus ako se prvi put otkrije)
  - `MAJOR_DAMAGE` — veće oštećenje (kazna -0.7, +0.5 bonus ako se prvi put otkrije)
  - `GOAL` — cilj, donji desni ugao (nagrada +100, epizoda završava)
- Agent bira akciju **ε-greedy** politikom (gore/dolje/desno/lijevo) i uči
  Bellmanovom jednačinom: `Q(s,a) ← Q(s,a) + α·[r + γ·max Q(s',a') − Q(s,a)]`
- Epsilon se eksponencijalno smanjuje kroz epizode (više istraživanja na
  početku, više eksploatacije na kraju treninga)

## Struktura projekta

```
env_v2.py             # Tkinter environment — grid, render, logika koraka i reward-a
agent_brain.py         # QLearningAgent — Q-tabela, ε-greedy, Bellman update
run_agent_v2.py        # Glavna trening petlja + ispis metrika + grafovi
plot_metrics_text.py   # Generiše PNG sa tekstualnim pregledom metrika
requirements.txt
```

> Napomena: `run_agent_v2.py` i `plot_metrics_text.py` importuju module po imenu
> (`env_v2`, `agent_brain`), pa fajlovi u repou moraju biti tačno tako nazvani.

## Pokretanje

```bash
pip install -r requirements.txt
python run_agent_v2.py
```

Trening se odvija u Tkinter prozoru (agent se vidi kako se kreće po mreži).
Nakon treninga (podrazumijevano 800 epizoda) program:

- ispisuje detaljne metrike u konzoli (uspješnost, TD greška, konvergencija...)
- čuva naučenu Q-tabelu u `q_table.npy`
- generiše `metrike_v2.png` — pregledna tabela sa svim ključnim brojkama
- generiše `rezultati_v2.png` — 6 grafova (koraci po epizodi, nagrada, TD
  greška, epsilon decay, stopa uspjeha, pronađena oštećenja)
- otvara poseban prozor sa vizualizacijom pronađene najkraće rute

## Konfiguracija

Parametri treninga se podešavaju u `CONFIG` rječniku na vrhu `run_agent_v2.py`:

| Parametar        | Podrazumijevano | Opis                                  |
|-------------------|-----------------|----------------------------------------|
| `n_episodes`      | 800             | broj epizoda treninga                 |
| `max_steps`       | 300             | maks. koraka po epizodi                |
| `learning_rate`   | 0.9             | α — brzina učenja                     |
| `gamma`           | 0.9             | γ — faktor popusta budućih nagrada    |
| `epsilon_decay`   | 0.995           | brzina opadanja istraživanja           |

## Rezultati

_Ovdje dodati `metrike_v2.png` i/ili `rezultati_v2.png` nakon treninga, kao i
opciono GIF/screenshot agenta dok se kreće kroz grid._

## Tehnologije

Python, NumPy, Pandas, Matplotlib, Tkinter
