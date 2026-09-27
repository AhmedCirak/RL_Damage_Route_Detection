# plot_metrics_text.py
# Generiše PNG sliku s metrikama — bijela pozadina, veliki fontovi, bez grida.

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from env_v2 import GRID_MAP, GRID_H, GRID_W, MINOR_DAMAGE, MAJOR_DAMAGE

# ═════════════════════════════════════════════════════════════════════
# PALETA — bijela pozadina
# ═════════════════════════════════════════════════════════════════════
BG_FIG   = '#FFFFFF'
BG_PANEL = '#F5F5F5'
C_HEAD   = '#1A1A2E'   # tamno — naslovi sekcija
C_KEY    = '#444444'   # tamnosiva — nazivi polja
C_VAL    = '#1565C0'   # plava — neutralne vrijednosti
C_GOOD   = '#2E7D32'   # zelena
C_WARN   = '#E65100'   # narandžasta
C_ERR    = '#C62828'   # crvena
C_MINOR  = '#F57F17'
C_MAJOR  = '#B71C1C'
C_BORDER = '#CCCCCC'


# ═════════════════════════════════════════════════════════════════════
# IZGRADNJA PODATAKA
# ═════════════════════════════════════════════════════════════════════

def _build_lines(history, env):
    successful = [h for h in history if h['success']]
    rewards    = [h['total_reward'] for h in history]
    errors     = [h['avg_td_error'] for h in history]
    total      = len(history)

    total_minor = int(np.sum(GRID_MAP == MINOR_DAMAGE))
    total_major = int(np.sum(GRID_MAP == MAJOR_DAMAGE))
    found_minor = history[-1]['minor_found']
    found_major = history[-1]['major_found']

    sections = {}

    succ_pct = 100 * len(successful) / total
    sections['Epizode'] = [
        ('Ukupno epizoda',   f'{total}',                              None),
        ('Uspješnih',        f'{len(successful)}  ({succ_pct:.1f}%)', C_GOOD if succ_pct > 50 else C_WARN),
        ('Neuspješnih',      f'{total - len(successful)}',            C_ERR if len(successful) == 0 else None),
    ]

    if successful:
        succ_steps = [h['steps'] for h in successful]
        sections['Koraci (uspješne epizode)'] = [
            ('Min / Max / Avg', f'{min(succ_steps)} / {max(succ_steps)} / {sum(succ_steps)/len(succ_steps):.1f}', None),
        ]
    else:
        sections['Koraci (uspješne epizode)'] = [
            ('—', 'Nema uspješnih epizoda', C_ERR),
        ]

    avg50 = sum(rewards[-50:]) / 50
    sections['Nagrade'] = [
        ('Prva epizoda',      f'{rewards[0]:.2f}',  None),
        ('Zadnja epizoda',    f'{rewards[-1]:.2f}', C_GOOD if rewards[-1] > 0 else C_ERR),
        ('Avg zadnjih 50 ep', f'{avg50:.2f}',       C_GOOD if avg50 > 0 else C_WARN),
    ]

    err_end = sum(errors[-50:]) / 50
    sections['TD greška'] = [
        ('Početak',          f'{errors[0]:.4f}',  None),
        ('Kraj (avg 50 ep)', f'{err_end:.4f}',    C_GOOD if err_end < errors[0] else C_WARN),
    ]

    minor_pct = 100 * found_minor // max(total_minor, 1)
    major_pct = 100 * found_major // max(total_major, 1)
    sections['Inspekcija oštećenja'] = [
        ('Minor pronađeno', f'{found_minor} / {total_minor}  ({minor_pct}%)',
         C_GOOD if minor_pct == 100 else (C_WARN if minor_pct >= 50 else C_ERR)),
        ('Major pronađeno', f'{found_major} / {total_major}  ({major_pct}%)',
         C_GOOD if major_pct == 100 else (C_WARN if major_pct >= 50 else C_ERR)),
    ]

    route_lines = []
    if env.shortest_steps < float('inf'):
        route_lines.append(('Pronađena',       'DA',                   C_GOOD))
        route_lines.append(('Dužina (koraci)', f'{env.shortest_steps}', None))
        if env.shortest_route:
            rm = sum(1 for (r, c) in env.shortest_route if GRID_MAP[r, c] == MINOR_DAMAGE)
            rM = sum(1 for (r, c) in env.shortest_route if GRID_MAP[r, c] == MAJOR_DAMAGE)
            route_lines.append(('Minor na ruti', f'{rm}', C_MINOR))
            route_lines.append(('Major na ruti', f'{rM}', C_MAJOR))
        first_s = next((h['episode'] for h in history if h['success']), None)
        if first_s:
            route_lines.append(('Prva uspješna ep.', f'{first_s}', None))
    else:
        route_lines.append(('Pronađena', 'NE',                      C_ERR))
        route_lines.append(('Razlog',    'Agent nije dostigao cilj', C_WARN))
    sections['Optimalna ruta'] = route_lines

    window  = 20
    conv_ep = None
    for i in range(len(history) - window):
        if all(h['success'] for h in history[i:i + window]):
            conv_ep = history[i]['episode']
            break

    conv_lines = []
    if conv_ep:
        stable_steps = [h['steps'] for h in history[conv_ep - 1:]]
        conv_lines += [
            ('Stabilna od ep.',       f'{conv_ep}',                              C_GOOD),
            ('Ep. stabilno ', f'{total - conv_ep}',                      None),
            ('Avg koraci ', f'{sum(stable_steps)/len(stable_steps):.1f}', None),
        ]
    else:
        succ_count = len(successful)
        conv_lines.append(('Status', f'Nije detektovana  (treba {window} uzast.)', C_WARN))
        if succ_count == 0:
            conv_lines.append(('Razlog', 'Nula uspješnih epizoda', C_ERR))
        else:
            max_streak = cur_streak = 0
            for h in history:
                if h['success']:
                    cur_streak += 1; max_streak = max(max_streak, cur_streak)
                else:
                    cur_streak = 0
            conv_lines += [
                ('Ukupno uspješnih',    f'{succ_count}  ({100*succ_count/total:.1f}%)', None),
                ('Najveći niz uspjeha', f'{max_streak} uzast.  (treba {window})',
                 C_GOOD if max_streak >= window else C_WARN),
                ('Komentar',
                 'Blizu konvergencije' if max_streak >= window // 2 else 'Daleko — prilagodi parametre',
                 C_WARN if max_streak >= window // 2 else C_ERR),
            ]
    sections['Konvergencija'] = conv_lines

    return sections


# ═════════════════════════════════════════════════════════════════════
# GLAVNI EXPORT
# ═════════════════════════════════════════════════════════════════════

def plot_metrics_png(history, env, config, filename='metrike_v2.png'):

    sections = _build_lines(history, env)

    fig = plt.figure(figsize=(13, 11))
    fig.patch.set_facecolor(BG_FIG)

    ax = fig.add_axes([0.03, 0.03, 0.94, 0.88])
    ax.set_facecolor(BG_PANEL)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    for spine in ax.spines.values():
        spine.set_edgecolor(C_BORDER)
        spine.set_linewidth(1.2)

    # Naslov
    fig.text(
        0.5, 0.975,
        f'Road Inspection RL v2  |  10×10  |  '
        f'α={config["learning_rate"]}  γ={config["gamma"]}  '
        f'ε-decay={config["epsilon_decay"]}  ep={config["n_episodes"]}',
        ha='center', va='top',
        color=C_HEAD, fontsize=14, fontweight='bold',
        fontfamily='monospace',
    )

    # ── Layout: dvije kolone ─────────────────────────────────────────
    sec_list  = list(sections.items())
    half      = (len(sec_list) + 1) // 2
    left_secs  = sec_list[:half]
    right_secs = sec_list[half:]

    FS_HEAD = 20     # font sekcije
    FS_KEY  = 20   # font ključa
    FS_VAL  = 20     # font vrijednosti
    DY_HEAD = 0.062
    DY_ROW  = 0.048
    DY_GAP  = 0.022

    def draw_sections(secs, x_key, x_val):
        y = 0.95
        for sec_title, rows in secs:
            ax.text(x_key, y, f'▸  {sec_title}',
                    transform=ax.transAxes,
                    color=C_HEAD, fontsize=FS_HEAD, fontweight='bold',
                    fontfamily='monospace', va='top')
            ax.plot([x_key, x_key + 0.44], [y - 0.006, y - 0.006],
                    transform=ax.transAxes,
                    color=C_BORDER, linewidth=1.0)
            y -= DY_HEAD

            for key, val, color in rows:
                c = color if color else C_VAL
                ax.text(x_key + 0.015, y, key,
                        transform=ax.transAxes,
                        color=C_KEY, fontsize=FS_KEY,
                        fontfamily='monospace', va='top')
                ax.text(x_val, y, val,
                        transform=ax.transAxes,
                        color=c, fontsize=FS_VAL,
                        fontfamily='monospace', va='top', fontweight='bold')
                y -= DY_ROW

            y -= DY_GAP

    # Lijeva kolona
    draw_sections(left_secs,  x_key=0.02, x_val=0.30)

    # Vertikalni separator
    ax.plot([0.50, 0.50], [0.02, 0.97],
            transform=ax.transAxes,
            color=C_BORDER, linewidth=1.2, linestyle='--')

    # Desna kolona
    draw_sections(right_secs, x_key=0.52, x_val=0.80)

    plt.savefig(filename, dpi=140, bbox_inches='tight', facecolor=BG_FIG)
    plt.close(fig)
    print(f'\nMetrike PNG snimljene u: {filename}')


# ═════════════════════════════════════════════════════════════════════
# Standalone test
# ═════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    import random
    random.seed(42)

    class FakeEnv:
        shortest_steps   = 18
        shortest_route   = [(0,0),(0,1),(0,2),(1,2),(2,2),(3,2),(4,2),
                            (5,2),(6,2),(7,2),(8,2),(9,2),(9,3),(9,4),(9,5),
                            (9,6),(9,7),(9,8),(9,9)]
        damage_found_log = []

    fake_history = []
    eps = 1.0
    for ep in range(1, 801):
        eps = max(0.01, eps * 0.995)
        succ = random.random() < (0.8 if ep > 400 else ep / 1000)
        fake_history.append({
            'episode':      ep,
            'steps':        random.randint(10, 50) if succ else 300,
            'total_reward': random.uniform(5, 15) if succ else random.uniform(-30, -5),
            'avg_td_error': max(0.001, 2.0 - ep * 0.002 + random.uniform(-0.1, 0.1)),
            'epsilon':      eps,
            'success':      succ,
            'minor_found':  min(3, ep // 200),
            'major_found':  min(2, ep // 300),
        })

    fake_config = {
        'learning_rate': 0.1, 'gamma': 0.9,
        'epsilon_decay': 0.995, 'n_episodes': 800, 'max_steps': 300,
    }

    plot_metrics_png(fake_history, FakeEnv(), fake_config, 'metrike_v2_test.png')
    print('Test PNG kreiran.')
