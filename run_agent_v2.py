
# run_agent_v2.py
# Q-Learning za inspekciju ceste — verzija 2
# Grid: 10x10  |  Tipovi: FREE, ROAD_CLOSED, MINOR_DAMAGE, MAJOR_DAMAGE, GOAL
# agent_brain.py ostaje nepromijenjen

from env_v2      import Environment, GRID_MAP, GRID_H, GRID_W, MINOR_DAMAGE, MAJOR_DAMAGE
from agent_brain import QLearningAgent
from plot_metrics_text import plot_metrics_png
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import sys
sys.stdout.reconfigure(encoding='utf-8')
# ════════════════════════════════════════════════════════════════
# KONFIGURACIJA
# ════════════════════════════════════════════════════════════════

CONFIG = {
    'n_episodes'    : 800,     # više epizoda — veći i gušći grid
    'max_steps'     : 300,     # veći grid → više koraka
    'learning_rate' : 0.9,
    'gamma'         : 0.9,
    'epsilon'       : 1.0,
    'epsilon_min'   : 0.01,
    'epsilon_decay' : 0.995,
    'render_speed'  : 0.0,
    'render_every'  : 100,
}

# ════════════════════════════════════════════════════════════════


def run():
    env = Environment(render_speed=CONFIG['render_speed'])
    agent = QLearningAgent(
        actions       = list(range(env.n_actions)),
        learning_rate = CONFIG['learning_rate'],
        gamma         = CONFIG['gamma'],
        epsilon       = CONFIG['epsilon'],
        epsilon_min   = CONFIG['epsilon_min'],
        epsilon_decay = CONFIG['epsilon_decay'],
    )

    history = []
    _print_header()

    def update():
        for episode in range(1, CONFIG['n_episodes'] + 1):

            state        = env.reset()
            total_reward = 0.0
            total_error  = 0.0
            steps        = 0
            done         = False

            while not done and steps < CONFIG['max_steps']:
                if CONFIG['render_every'] == 0 or episode % CONFIG['render_every'] == 0:
                    env.render()

                action                   = agent.choose_action(state)
                next_state, reward, done = env.step(action)
                td_error                 = agent.learn(state, action, reward, next_state)

                total_reward += reward
                total_error  += td_error
                state         = next_state
                steps        += 1

            agent.decay_epsilon()

            minor_found = sum(1 for d in env.damage_found_log if d['type'] == 'minor')
            major_found = sum(1 for d in env.damage_found_log if d['type'] == 'major')

            history.append({
                'episode'      : episode,
                'steps'        : steps,
                'total_reward' : round(total_reward, 3),
                'avg_td_error' : round(total_error / max(steps, 1), 4),
                'epsilon'      : round(agent.epsilon, 4),
                'success'      : done and reward >= 0.9,
                'minor_found'  : minor_found,
                'major_found'  : major_found,
            })

            if episode % 100 == 0 or episode == 1:
                print(
                    f'Ep {episode:>4}  '
                    f'koraci={steps:>3}  '
                    f'reward={total_reward:>7.2f}  '
                    f'e={agent.epsilon:.3f}  '
                    f'minor={minor_found}  major={major_found}'
                )

        # ── Nakon treninga ────────────────────────────────────────
        _print_metrics(history, env)
        _print_route(env)
        agent.print_q_table()
        np.save('q_table.npy', agent.q_table)    # sprema Q-tablicu u fajl
        print('Q-tablica snimljena u: q_table.npy')
        plot_metrics_png(history, env, CONFIG)   # sprema metrike_v2.png
        env.show_optimal_route_window()
        _plot_results(history, CONFIG)           # sprema rezultati_v2.png
        env.mainloop()

    env.after(100, update)
    env.mainloop()


# ════════════════════════════════════════════════════════════════
# ISPIS HEADERA
# ════════════════════════════════════════════════════════════════

def _print_header():
    print('=' * 65)
    print('  Road Inspection RL v2 — Q-Learning  |  10×10 grid')
    print(f'  alpha={CONFIG["learning_rate"]}  '
          f'gamma={CONFIG["gamma"]}  '
          f'epsilon_decay={CONFIG["epsilon_decay"]}')
    print(f'  Epizode: {CONFIG["n_episodes"]}  '
          f'Maks. koraka: {CONFIG["max_steps"]}')
    print('  Tipovi: FREE | ROAD_CLOSED | MINOR_DAMAGE | MAJOR_DAMAGE | GOAL')
    print('=' * 65)


# ════════════════════════════════════════════════════════════════
# METRIKE
# ════════════════════════════════════════════════════════════════

def _print_metrics(history, env):
    successful = [h for h in history if h['success']]
    rewards    = [h['total_reward'] for h in history]
    errors     = [h['avg_td_error'] for h in history]

    total_minor = int(np.sum(GRID_MAP == MINOR_DAMAGE))
    total_major = int(np.sum(GRID_MAP == MAJOR_DAMAGE))
    found_minor = history[-1]['minor_found']
    found_major = history[-1]['major_found']

    total = len(history)

    print('\n' + '=' * 55)
    print('  METRIKE — v2')
    print('=' * 55)

    print(f'Ukupno epizoda            : {total}')
    print(f'Uspješnih                 : {len(successful)} ({100*len(successful)/total:.1f}%)')
    print(f'Neuspješnih               : {total - len(successful)}')

    if successful:
        succ_steps = [h['steps'] for h in successful]
        print(f'\nKoraci (uspješne):')
        print(f'  Min / Max / Avg       : {min(succ_steps)} / {max(succ_steps)} / {sum(succ_steps)/len(succ_steps):.1f}')

    print(f'\nNagrade:')
    print(f'  Prva epizoda            : {rewards[0]:.2f}')
    print(f'  Zadnja epizoda          : {rewards[-1]:.2f}')
    print(f'  Avg zadnjih 50          : {sum(rewards[-50:])/50:.2f}')

    print(f'\nTD greška:')
    print(f'  Pocetak                 : {errors[0]:.4f}')
    print(f'  Kraj (avg zadnjih 50)   : {sum(errors[-50:])/50:.4f}')

    print(f'\nInspekcija oštećenja:')
    print(f'  Minor damage — pronađeno: {found_minor} / {total_minor}  '
          f'({100*found_minor//max(total_minor,1)}%)')
    print(f'  Major damage — pronađeno: {found_major} / {total_major}  '
          f'({100*found_major//max(total_major,1)}%)')

    print(f'\nOptimalna ruta:')
    if env.shortest_steps < float('inf'):
        print(f'  Pronađena              : DA')
        print(f'  Dužina (koraci)        : {env.shortest_steps}')
        if env.shortest_route:
            route_minor = sum(1 for (r, c) in env.shortest_route if GRID_MAP[r, c] == MINOR_DAMAGE)
            route_major = sum(1 for (r, c) in env.shortest_route if GRID_MAP[r, c] == MAJOR_DAMAGE)
            print(f'  Minor damage na ruti   : {route_minor}')
            print(f'  Major damage na ruti   : {route_major}')
        first_success = next((h['episode'] for h in history if h['success']), None)
        if first_success:
            print(f'  Prva uspješna epizoda  : {first_success}')
    else:
        print(f'  Pronađena              : NE — agent nikad nije dostigao cilj')
        print(f'  Mogući razlozi         :')
        avg_last = sum(rewards[-50:]) / 50
        if avg_last < -5:
            print(f'    - Agent se zaglavio (avg nagrada zadnjih 50ep: {avg_last:.2f})')
        if history[-1]['epsilon'] > 0.3:
            print(f'    - Epsilon još visok ({history[-1]["epsilon"]:.3f}) — premalo eksploatacije')
        if CONFIG['n_episodes'] < 300:
            print(f'    - Premalo epizoda ({CONFIG["n_episodes"]}) za ovaj grid')
        print(f'  Preporuka              : povećaj n_episodes ili smanji epsilon_decay')

    print(f'\nKonvergencija:')
    window    = 20
    conv_ep   = None
    for i in range(len(history) - window):
        if all(h['success'] for h in history[i:i + window]):
            conv_ep = history[i]['episode']
            break

    if conv_ep:
        print(f'  Stabilna od epizode    : {conv_ep}')
        print(f'  Epizoda stabilno do kraja : {total - conv_ep}')
        stable_steps = [h['steps'] for h in history[conv_ep - 1:]]
        print(f'  Avg koraci (stabilno)  : {sum(stable_steps)/len(stable_steps):.1f}')
    else:
        succ_count = len(successful)
        print(f'  Nije detektovana       : (treba {window} uzastopnih uspjeha)')
        if succ_count == 0:
            print(f'  Razlog                 : Nula uspješnih epizoda')
        else:
            max_streak = cur_streak = 0
            for h in history:
                if h['success']:
                    cur_streak += 1
                    max_streak = max(max_streak, cur_streak)
                else:
                    cur_streak = 0
            print(f'  Ukupno uspješnih       : {succ_count} ({100*succ_count/total:.1f}%)')
            print(f'  Najveći niz uspjeha    : {max_streak} uzastopnih (treba {window})')
            if max_streak >= window // 2:
                print(f'  Komentar               : Blizu konvergencije — pokušaj više epizoda')
            else:
                print(f'  Komentar               : Daleko od konvergencije — prilagodi parametre')

    print('=' * 55)


# ════════════════════════════════════════════════════════════════
# PRIKAZ RUTE U KONZOLI
# ════════════════════════════════════════════════════════════════

def _print_route(env):
    print('\n=== OPTIMALNA RUTA (*) ===')

    if env.shortest_route is None:
        print('Ruta nije pronađena — agent nije dostigao cilj.')
        return

    route_set = set(map(tuple, env.shortest_route))
    symbols   = {0: '.', 1: 'X', 2: 'm', 3: 'M', 4: 'G'}

    for r in range(GRID_H):
        row_str = ''
        for c in range(GRID_W):
            if (r, c) == (0, 0):
                row_str += 'S '
            elif (r, c) in route_set:
                row_str += '* '
            else:
                row_str += symbols.get(int(GRID_MAP[r, c]), '?') + ' '
        print(row_str)

    print(f'\nLegenda: S=start  *=ruta  G=cilj  X=zatvorena  m=minor  M=major  .=slobodno')
    print(f'Dužina najkraće rute: {env.shortest_steps} koraka')


# ════════════════════════════════════════════════════════════════
# MATPLOTLIB GRAFOVI — 6 panela
# ════════════════════════════════════════════════════════════════

def _plot_results(history, config):
    episodes     = [h['episode']       for h in history]
    steps        = [h['steps']         for h in history]
    rewards      = [h['total_reward']  for h in history]
    td_errors    = [h['avg_td_error']  for h in history]
    epsilons     = [h['epsilon']       for h in history]
    minor_found  = [h['minor_found']   for h in history]
    major_found  = [h['major_found']   for h in history]
    successes    = [1 if h['success'] else 0 for h in history]

    def moving_avg(data, window=20):
        return np.convolve(data, np.ones(window) / window, mode='valid')

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.patch.set_facecolor('#FFFFFF')
    fig.suptitle(
        f'Road Inspection RL v2  |  10×10  |  '
        f'alpha={config["learning_rate"]}  gamma={config["gamma"]}  '
        f'eps-decay={config["epsilon_decay"]}',
        fontsize=12, fontweight='bold', color='#111111'
    )

    BG = '#F5F5F5'
    for ax in axes.flat:
        ax.set_facecolor(BG)
        ax.tick_params(colors='#222222', labelsize=8)
        for spine in ax.spines.values():
            spine.set_edgecolor('#CCCCCC')
        ax.grid(True, alpha=0.5, color='#CCCCCC')

    def style_ax(ax, title, xlabel='Epizoda', ylabel=''):
        ax.set_title(title, color='#111111', fontsize=9)
        ax.set_xlabel(xlabel, color='#333333', fontsize=8)
        ax.set_ylabel(ylabel, color='#333333', fontsize=8)

    # 1. Koraci po epizodi
    ax = axes[0, 0]
    ax.plot(episodes, steps, color='#85B8E8', alpha=0.4, linewidth=0.7)
    if len(steps) >= 20:
        ax.plot(episodes[19:], moving_avg(steps), color='#0D47A1', linewidth=2, label='avg 20ep')
    ax.legend(fontsize=7, facecolor='#FFFFFF', labelcolor='#222222')
    style_ax(ax, 'Koraci po epizodi', ylabel='Koraci')

    # 2. Ukupna nagrada
    ax = axes[0, 1]
    ax.plot(episodes, rewards, color='#5CC4A0', alpha=0.4, linewidth=0.7)
    if len(rewards) >= 20:
        ax.plot(episodes[19:], moving_avg(rewards), color='#0A6B4B', linewidth=2, label='avg 20ep')
    ax.axhline(0, color='#888888', linestyle='--', linewidth=0.8)
    ax.legend(fontsize=7, facecolor='#FFFFFF', labelcolor='#222222')
    style_ax(ax, 'Ukupna nagrada po epizodi', ylabel='Nagrada')

    # 3. TD greška
    ax = axes[0, 2]
    ax.plot(episodes, td_errors, color='#E8956D', alpha=0.4, linewidth=0.7)
    if len(td_errors) >= 20:
        ax.plot(episodes[19:], moving_avg(td_errors), color='#7B2500', linewidth=2, label='avg 20ep')
    ax.legend(fontsize=7, facecolor='#FFFFFF', labelcolor='#222222')
    style_ax(ax, 'TD greška (konvergencija)', ylabel='Avg TD greška')

    # 4. Epsilon decay
    ax = axes[1, 0]
    ax.plot(episodes, epsilons, color='#5048C8', linewidth=2)
    ax.fill_between(episodes, epsilons, alpha=0.15, color='#5048C8')
    style_ax(ax, 'Epsilon decay', ylabel='Epsilon')

    # 5. Stopa uspjeha (rolling 50)
    ax = axes[1, 1]
    window = 50
    if len(successes) >= window:
        rolling = [
            sum(successes[i:i + window]) / window * 100
            for i in range(len(successes) - window + 1)
        ]
        ax.plot(episodes[window - 1:], rolling, color='#3A6B00', linewidth=2)
        ax.fill_between(episodes[window - 1:], rolling, alpha=0.15, color='#3A6B00')
    ax.set_ylim(0, 105)
    style_ax(ax, f'Stopa uspjeha (rolling {window} ep)', ylabel='Uspjeh (%)')

    # 6. Pronađena oštećenja — minor vs major
    ax = axes[1, 2]
    ax.plot(episodes, minor_found, color='#EF9F27', linewidth=2, label='Minor')
    ax.plot(episodes, major_found, color='#E24B4A', linewidth=2, label='Major')
    ax.fill_between(episodes, minor_found, alpha=0.12, color='#EF9F27')
    ax.fill_between(episodes, major_found, alpha=0.12, color='#E24B4A')
    total_minor = int(np.sum(GRID_MAP == MINOR_DAMAGE))
    total_major = int(np.sum(GRID_MAP == MAJOR_DAMAGE))
    ax.axhline(total_minor, color='#EF9F27', linestyle='--',
               linewidth=1.0, alpha=0.6, label=f'Ukupno minor ({total_minor})')
    ax.axhline(total_major, color='#E24B4A', linestyle='--',
               linewidth=1.0, alpha=0.6, label=f'Ukupno major ({total_major})')
    ax.legend(fontsize=7, facecolor='#FFFFFF', labelcolor='#222222')
    style_ax(ax, 'Pronađena oštećenja — Minor vs Major', ylabel='Broj')

    plt.tight_layout()
    plt.savefig('rezultati_v2.png', dpi=120, bbox_inches='tight', facecolor='#FFFFFF')
    print('\nGrafovi snimljeni u: rezultati_v2.png')
    plt.show()


# ════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    run()
